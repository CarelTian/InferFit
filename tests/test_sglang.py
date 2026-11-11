"""SGLang-specific contracts and regression coverage; no SGLang/Torch install."""
from __future__ import annotations

import json
import math
import unittest
from dataclasses import replace
from importlib.resources import files
from pathlib import Path

from inferfit import (GiB, Hardware, Parallelism, RuntimeProfile, Workload,
                                check, from_config, kv_per_token, load_glm51,
                                optimize_pipeline, search)
from inferfit.cli import evaluate_document


class SglangTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_glm51()
        cls.runtime = RuntimeProfile.sglang()
        cls.plan = Parallelism(8, 1, 8, 2, 16)
        cls.contract = json.loads(files("inferfit").joinpath("data/sglang-contract.json").read_text())

    def test_pinned_factory_and_unknown_version(self):
        self.assertEqual(self.runtime.engine, "sglang")
        self.assertEqual(self.runtime.version, self.contract["version"])
        self.assertEqual(check(self.model, runtime=RuntimeProfile.sglang(version="latest")).status, "unsupported")
        self.assertEqual(check(self.model, runtime=RuntimeProfile(engine="sglang")).status, "unsupported")

    def test_kv_matches_pinned_pool_contract(self):
        for quant, field in [("bf16", "mla_bf16_bytes_per_layer_token"), ("fp8", "mla_fp8_bytes_per_layer_token")]:
            parts = kv_per_token(self.model, runtime=self.runtime, kv_quantization=quant)
            self.assertEqual(parts["total"], 78 * (self.contract[field] + self.contract["index_bytes_per_layer_token"]))
        self.assertEqual(kv_per_token(self.model, runtime=self.runtime, tp=8)["total"], 100152)
        with self.assertRaises(ValueError):
            kv_per_token(self.model, runtime=self.runtime, kv_quantization="fp8_ds_mla")

    def test_eight_cards_fail_and_sixteen_pass(self):
        self.assertEqual(check(self.model, runtime=self.runtime).status, "fail")
        result = check(self.model, parallelism=self.plan, runtime=self.runtime)
        self.assertTrue(result.memory_pass)
        self.assertEqual(result.summary["engine"], "sglang")
        self.assertEqual(result.summary["engine_version"], "0.5.23")
        self.assertFalse(result.summary["deployment_verified"])

    def test_fp8_alias_and_backend_flags(self):
        result = check(self.model, parallelism=self.plan, runtime=self.runtime, kv_quantization="fp8")
        self.assertEqual(result.kv_quantization, "fp8_e4m3")
        self.assertEqual(result.ranks[0].kv_bytes_per_token, 30732)
        args = result.summary["engine_args"]
        self.assertEqual(args["kv_cache_dtype"], "fp8_e4m3")
        self.assertEqual(args["dsa_decode_backend"], "flashmla_kv")
        self.assertEqual(args["dsa_prefill_backend"], "flashmla_auto")

    def test_native_arguments_exist_in_pinned_server(self):
        for kv in ("bf16", "fp8"):
            result = check(self.model, parallelism=self.plan, runtime=self.runtime, kv_quantization=kv)
            args = result.summary["engine_args"]
            self.assertLessEqual(set(args), set(self.contract["cli_argument_fields"]))
            self.assertTrue(args["disable_prefill_cuda_graph"])
            self.assertTrue(args["disable_decode_cuda_graph"])
            self.assertTrue(args["disable_overlap_schedule"])
            self.assertTrue(args["disable_radix_cache"])
            self.assertTrue(args["disable_shared_experts_fusion"])
            self.assertEqual(args["max_total_tokens"], 8192)
            self.assertNotIn("gpu_memory_utilization", args)
            self.assertNotIn("VLLM_PP_LAYER_PARTITION", result.summary["engine_env"])

    def test_partial_ep_uses_residual_expert_tp(self):
        # Native TP8/EP2 means 128 experts per rank, each matrix TP-sharded by 4.
        partial = check(self.model, parallelism=replace(self.plan, ep=2), runtime=self.runtime)
        full = check(self.model, parallelism=self.plan, runtime=self.runtime)
        self.assertTrue(partial.memory_pass)
        self.assertEqual(partial.ranks[0].weight_bytes, full.ranks[0].weight_bytes)
        self.assertEqual(check(self.model, parallelism=replace(self.plan, ep=16), runtime=self.runtime).status, "unsupported")
        self.assertEqual(check(self.model, parallelism=replace(self.plan, ep=2)).status, "unsupported")

    def test_ordinary_multi_node_dp_is_rejected(self):
        plan = Parallelism(8, 2, 8, 1, 16)
        result = check(self.model, parallelism=plan, runtime=self.runtime)
        self.assertEqual(result.status, "unsupported")
        self.assertTrue(any("multi-node ordinary DP" in reason for reason in result.reasons))

    def test_dp_attention_maps_portable_mesh_to_native_tp(self):
        runtime = RuntimeProfile.sglang(sglang_dp_attention=True)
        plan = Parallelism(8, 2, 16, 1, 16)
        result = check(self.model, parallelism=plan, runtime=runtime,
                       workload=Workload(concurrency_per_dp=3))
        self.assertTrue(result.memory_pass)
        args = result.summary["engine_args"]
        self.assertEqual((args["tp_size"], args["attn_dp_size"], args["ep_size"]), (16, 2, 16))
        self.assertEqual(args["dp_size"], 1)
        self.assertNotIn("enable_dp_attention", args)
        self.assertEqual(args["max_running_requests"], 6)
        self.assertEqual(args["chunked_prefill_size"] // args["attn_dp_size"], 2048)
        self.assertEqual(args["max_total_tokens"], 3 * 8192)
        self.assertEqual(result.summary["target_resident_tokens_per_dp"], 3 * 8192)
        self.assertEqual(result.ranks[0].kv_bytes_per_token, 100152)

    def test_dp_attention_dense_and_shared_sharding_matches_native_width(self):
        plan = Parallelism(8, 2, 16, 1, 16)
        a = check(self.model, parallelism=plan, runtime=RuntimeProfile.sglang(sglang_dp_attention=True))
        b = check(self.model, parallelism=plan)
        # SGLang none-A2A shared MLPs shard by native TP16; vLLM duplicates them.
        self.assertLess(a.ranks[0].weight_bytes, b.ranks[0].weight_bytes)
        self.assertEqual(a.summary["placed_weight_bytes_total"], 16 * a.ranks[0].weight_bytes)

    def test_dp_attention_rank_mapping(self):
        result = check(self.model, parallelism=Parallelism(4, 2, 8, 2, 16),
                       runtime=RuntimeProfile.sglang(sglang_dp_attention=True))
        self.assertTrue(result.memory_pass)
        self.assertEqual((result.ranks[4].pp_rank, result.ranks[4].dp_rank), (0, 1))
        self.assertEqual((result.ranks[8].pp_rank, result.ranks[8].dp_rank), (1, 0))

    def test_equal_worker_topology_and_no_cross_attention_tp(self):
        optimized = optimize_pipeline(self.model, parallelism=Parallelism(4, 1, 4, 3, 12), runtime=self.runtime)
        self.assertTrue(optimized.memory_pass)
        self.assertEqual(optimized.summary["stage_layers"], [28, 25, 25])
        self.assertEqual(optimized.summary["nodes_required"], 3)
        self.assertEqual(optimized.summary["workers_per_node"], 4)
        self.assertEqual(optimized.summary["engine_args"]["nnodes"], 3)
        self.assertEqual([r.node for r in optimized.ranks], [0]*4+[1]*4+[2]*4)
        allowed = optimize_pipeline(self.model, parallelism=Parallelism(4, 1, 4, 3, 12), runtime=self.runtime,
                                    hardware=Hardware(allow_cross_node_tp=True))
        self.assertEqual(allowed.summary["nodes_required"], 3)
        self.assertEqual(allowed.summary["workers_per_node"], 4)

    def test_two_by_six_cannot_launch_three_pipeline_stages(self):
        h=Hardware(gpus_per_node=6,available_nodes=2,allow_cross_node_tp=True)
        p=Parallelism(4,1,4,3,12,stage_layers=(28,25,25))
        r=check(self.model,hardware=h,parallelism=p,runtime=self.runtime)
        self.assertEqual(r.status,"unsupported")
        self.assertTrue(any("native SGLang" in reason for reason in r.reasons))
        # A rank cannot be made launchable by an apparent byte-capacity pass.
        self.assertIsNone(r.memory_pass)
        # A rectangular TP2/PP6 plan is launchable on 2x6; capacity, not ranks,
        # becomes the limiting factor. Nominal decimal GB and GiB differ.
        alternate=Parallelism(2,1,2,6,12)
        thin=optimize_pipeline(self.model,hardware=h,parallelism=alternate,runtime=self.runtime)
        self.assertEqual(thin.status,"fail")
        enough=optimize_pipeline(self.model,hardware=replace(h,memory_bytes=80*GiB),parallelism=alternate,runtime=self.runtime)
        self.assertEqual(enough.status,"pass")
        self.assertEqual(enough.summary["nodes_required"],2)
        h=replace(h,available_nodes=3)
        self.assertEqual(check(self.model,hardware=h,parallelism=p,runtime=self.runtime).status,"pass")

    def test_actual_upcasts_and_materialized_bf16_projection_copies(self):
        v = check(self.model, parallelism=self.plan)
        s = check(self.model, parallelism=self.plan, runtime=self.runtime)
        self.assertEqual(s.ranks[0].weight_bytes-v.ranks[0].weight_bytes, 39*128*2*2)
        v16 = check(self.model, parallelism=self.plan, weight_quantization="bf16")
        s16 = check(self.model, parallelism=self.plan, weight_quantization="bf16", runtime=self.runtime)
        self.assertEqual(v16.ranks[0].runtime_breakdown["mla_bf16_projection_copies"], 0)
        self.assertGreater(s16.ranks[0].runtime_breakdown["mla_bf16_projection_copies"], 0)

    def test_request_mapping_allocation_and_dummy_page(self):
        w = Workload(context_tokens=65, concurrency_per_dp=3)
        result = check(self.model, parallelism=self.plan, runtime=self.runtime, workload=w)
        rank = result.ranks[0]
        self.assertEqual(rank.runtime_breakdown["block_table_metadata"], (3+1)*(65+4)*4)
        self.assertEqual(rank.kv_allocated_tokens, 3*128+64)
        self.assertEqual(result.summary["engine_args"]["max_total_tokens"], 384)

    def test_default_pp_remainder_goes_to_last_stages(self):
        plan = Parallelism(8, 1, 8, 4, 32)
        result = check(self.model, parallelism=plan, runtime=self.runtime)
        self.assertEqual(result.summary["stage_layers"], [19, 19, 20, 20])
        self.assertEqual(result.summary["engine_env"]["SGLANG_PP_LAYER_PARTITION"], "19,19,20,20")

    def test_chunk_alignment_envelope_and_boolean_rejection(self):
        for kwargs in [dict(workload=Workload(max_batch_tokens=65)),
                       dict(runtime=RuntimeProfile.sglang(sglang_dp_attention=True), workload=Workload(max_batch_tokens=8192), parallelism=Parallelism(4, 2, 8, 2, 16))]:
            kwargs.setdefault("runtime", self.runtime)
            self.assertEqual(check(self.model, **kwargs).status, "unsupported")
        self.assertEqual(check(self.model, runtime=RuntimeProfile.sglang(sglang_dp_attention="false")).status, "invalid")

    def test_search_filters_by_engine_contract_and_optimizes_pp(self):
        result = search(self.model, runtime=self.runtime, gpu_counts=[8, 12, 16], max_gpus=16,
                        tp_values=[4, 8], pp_values=[1, 2, 3], dp_values=[1, 2])
        self.assertEqual(result.minimum_num_gpus, 12)
        self.assertGreater(result.counts["unsupported"], 0)
        self.assertTrue(all(r.summary["engine"]=="sglang" for r in result.candidates))
        self.assertTrue(all(r.parallelism.dp==1 for r in result.candidates))
        self.assertTrue(any(r.parallelism.ep==2 for r in result.candidates))

    def test_json_engine_selection_uses_sglang_defaults(self):
        document = {"model":"glm-5.1", "runtime":{"engine":"sglang"},
                    "parallelism":{"tp":8,"dp":1,"ep":8,"pp":2,"num_gpus":16}}
        result = evaluate_document(document)
        self.assertTrue(result["memory_pass"])
        self.assertEqual(result["summary"]["engine_version"], "0.5.23")
        self.assertEqual(result["summary"]["runtime_profile"], self.runtime.name)

    def test_generic_gqa_and_unknown_feature_boundary(self):
        model = from_config(dict(model_type="llama", hidden_size=512, intermediate_size=1024,
                                 num_hidden_layers=4,num_attention_heads=8,num_key_value_heads=2,
                                 head_dim=64,vocab_size=1024,max_position_embeddings=32768))
        self.assertEqual(kv_per_token(model, runtime=self.runtime,kv_quantization="fp8")["total"],1024)
        result = check(model, runtime=self.runtime, parallelism=Parallelism(2,2,1,1,4),weight_quantization="bf16")
        self.assertTrue(result.memory_pass)
        self.assertEqual(check(model, runtime=RuntimeProfile.sglang(sglang_dp_attention=True),
                               parallelism=Parallelism(2,2,1,1,4)).status,"unsupported")


if __name__ == "__main__":
    unittest.main()
