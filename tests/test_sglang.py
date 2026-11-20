"""SGLang-specific contracts and regression coverage; no SGLang/Torch install."""
from __future__ import annotations

import json
import math
import unittest
from dataclasses import replace
from importlib.resources import files
from pathlib import Path

from inferfit import (GiB, Hardware, Parallelism, RuntimeProfile, Workload,
                                check, from_config, kv_per_token, load_model,
                                load_hardware, optimize_pipeline, PDPool, check_pd, search)
from inferfit.cli import evaluate_document


class SglangTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model("glm-5.1")
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

    def test_qwen3_moe_dp_attention_check_and_search(self):
        model = load_model("qwen3-235b-a22b")
        runtime = RuntimeProfile.sglang(sglang_dp_attention=True)
        plan = Parallelism(4, 2, 8, 1, 8)
        result = check(model, parallelism=plan, runtime=runtime, weight_quantization="bf16")
        self.assertEqual(result.status, "pass")
        args = result.summary["engine_args"]
        self.assertEqual((args["tp_size"], args["attn_dp_size"], args["dp_size"], args["ep_size"]),
                         (8, 2, 1, 8))
        self.assertEqual(result.ranks[0].kv_bytes_per_token, 94 * 128 * 2 * 2)
        self.assertEqual(args["page_size"], 1)
        self.assertEqual(result.ranks[0].kv_allocated_tokens, 8192 + 1)
        self.assertEqual(result.ranks[0].kv_required_bytes, (8192 + 1) * 48128)
        self.assertFalse(result.summary["deployment_verified"])
        found = search(model, runtime=runtime, weight_quantization="bf16", gpu_counts=[8],
                       tp_values=[4], dp_values=[2], pp_values=[1], ep_values=[8], max_gpus=8)
        self.assertEqual(found.minimum_num_gpus, 8)
        self.assertEqual(found.candidates[0].parallelism, replace(plan, stage_layers=(94,)))

    def test_ordinary_attention_applies_alignment_to_the_effective_page_size(self):
        model = load_model("qwen3-235b-a22b")
        workload = Workload(max_batch_tokens=65)
        options = dict(parallelism=Parallelism(8, 1, 8, 1, 8), weight_quantization="bf16")
        sglang = check(model, workload=workload, runtime=self.runtime, **options)
        self.assertEqual(sglang.summary["engine_args"]["page_size"], 1)
        self.assertEqual(sglang.summary["engine_args"]["chunked_prefill_size"], 65)
        vllm = check(model, workload=workload, **options)
        self.assertEqual(vllm.summary["engine_args"]["block_size"], 16)
        self.assertEqual(vllm.summary["engine_args"]["max_num_batched_tokens"], 65)
        misaligned = check(model, workload=replace(workload, block_size=64), runtime=self.runtime, **options)
        self.assertEqual(misaligned.status, "unsupported")
        self.assertIn("SGLang chunked prefill size must be divisible by page_size", misaligned.reasons)

    def test_auto_search_keeps_standard_dp_opt_in(self):
        model = load_model("qwen3-235b-a22b")
        options = dict(runtime=self.runtime, tp_values=[4], pp_values=[1],
                       ep_values=[4], max_gpus=8, total_concurrency=7)
        auto = search(model, **options)
        self.assertEqual([c.parallelism.dp for c in auto.candidates], [1, 2])
        self.assertEqual([c.sglang_dp_attention for c in auto.candidates], [False, True])
        self.assertEqual([c.concurrency_per_dp for c in auto.candidates], [7, 4])
        explicit = search(model, dp_values=[2], **options)
        self.assertEqual(len(explicit.candidates), 1)
        self.assertEqual(explicit.candidates[0].parallelism.dp, 2)
        self.assertFalse(explicit.candidates[0].sglang_dp_attention)
        self.assertEqual(explicit.candidates[0].concurrency_per_dp, 4)

    def test_total_concurrency_does_not_override_per_group_input(self):
        with self.assertRaisesRegex(ValueError, "total_concurrency must be a positive integer"):
            search(self.model, runtime=self.runtime, total_concurrency=True)
        with self.assertRaisesRegex(ValueError, "not both"):
            search(self.model, runtime=self.runtime, total_concurrency=5,
                   workload=Workload(concurrency_per_dp=2))

    def test_dense_qwen3_dp_attention_weight_and_cache_geometry(self):
        model = from_config(dict(model_type="qwen3", hidden_size=512, intermediate_size=1024,
                                 num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2,
                                 head_dim=64, vocab_size=1024, max_position_embeddings=32768))
        runtime = RuntimeProfile.sglang(sglang_dp_attention=True)
        result = check(model, runtime=runtime, parallelism=Parallelism(2, 2, 1, 1, 4),
                       weight_quantization="bf16")
        self.assertEqual(result.status, "pass")
        # Attention and embedding use TP2; dense FFN and untied head use native TP4.
        embedding = 1024 * 512 * 2 // 2
        output = 1024 * 512 * 2 // 4
        attention = (512 * 512 + 2 * 128 * 512 + 512 * 512) * 2 // 2
        ffn = 3 * 512 * 1024 * 2 // 4
        norms = 2 * 512 * 2 + 2 * 64 * 2
        self.assertEqual(result.ranks[0].weight_bytes,
                         embedding + output + 512 * 2 + 4 * (attention + ffn + norms))
        self.assertEqual(result.ranks[0].kv_bytes_per_token, 4 * 64 * 2 * 2)
        # With four KV heads, attention TP2 stores two heads per rank, not native TP4's one.
        sharded = from_config({**model.config, "num_key_value_heads": 4})
        result = check(sharded, runtime=runtime, parallelism=Parallelism(2, 2, 1, 1, 4),
                       weight_quantization="bf16")
        self.assertEqual(result.status, "pass")
        self.assertEqual(result.ranks[0].kv_bytes_per_token, 4 * 2 * 64 * 2 * 2)
        replicated = check(model, runtime=runtime, parallelism=Parallelism(4, 2, 1, 1, 8),
                           weight_quantization="bf16")
        self.assertEqual(replicated.status, "pass")
        self.assertEqual(replicated.ranks[0].kv_bytes_per_token, 4 * 64 * 2 * 2)

    def test_dp_attention_keeps_hybrid_mapping_boundary(self):
        result = check(load_model("qwen3.8-2.4t-a95b"),
                       runtime=RuntimeProfile.sglang(sglang_dp_attention=True),
                       parallelism=Parallelism(4, 2, 8, 1, 8), weight_quantization="bf16")
        self.assertEqual(result.status, "unsupported")
        self.assertTrue(any("layer-boundary mappings" in reason for reason in result.reasons))
        self.assertFalse(any("only for GLM MLA/DSA" in reason for reason in result.reasons))

    def test_blackwell_dsa_backends_choose_matching_fp8_kv_rows(self):
        runtime = RuntimeProfile.sglang(sglang_dp_attention=True)
        plan = Parallelism(4, 2, 8, 1, 8)
        cases = [
            ("h200-141gb", "flashmla_auto", "flashmla_kv", 656, "pass"),
            ("b200-180gb", "trtllm", "trtllm", 576, "pass"),
            ("gb300-279gb", "trtllm", "trtllm", 576, "pass"),
            ("rtx5090-32gb", "flashinfer_sparse_mla", "flashinfer_sparse_mla", 656, "fail"),
        ]
        for name, prefill, decode, row_bytes, status in cases:
            with self.subTest(hardware=name):
                h = load_hardware(name)
                result = check(self.model, hardware=h, parallelism=plan, runtime=runtime,
                               kv_quantization="fp8")
                self.assertEqual(result.status, status)
                rank = result.ranks[0]
                args = result.summary["engine_args"]
                self.assertEqual((args["dsa_prefill_backend"], args["dsa_decode_backend"]),
                                 (prefill, decode))
                self.assertEqual((args["tp_size"], args["attn_dp_size"], args["dp_size"]), (8, 2, 1))
                expected = 78 * (row_bytes + 132)
                self.assertEqual(rank.kv_bytes_per_token, expected)
                self.assertEqual(rank.kv_required_bytes, (8192 + 64) * expected)
                rate = kv_per_token(self.model, tp=plan.tp, runtime=runtime, hardware=h,
                                    kv_quantization="fp8")
                self.assertEqual(rate["total"], expected)
                if decode == "trtllm":
                    self.assertEqual(rank.runtime_breakdown["dsa_flashinfer_workspace"], 384 * 1024**2)
                    self.assertEqual(rank.runtime_breakdown["dsa_trtllm_kv_counters"], 8192 * 16 * 4)
                if decode == "flashinfer_sparse_mla":
                    self.assertEqual(result.summary["engine_env"]["SGLANG_FP8_PAGED_MQA_LOGITS_TORCH"], "true")

    def test_blackwell_bf16_and_unsupported_dsa_paths(self):
        runtime = RuntimeProfile.sglang(sglang_dp_attention=True)
        plan = Parallelism(4, 2, 8, 1, 8)
        result = check(self.model, hardware=load_hardware("b300-270gb"),
                       parallelism=plan, runtime=runtime)
        self.assertEqual(result.status, "pass")
        args = result.summary["engine_args"]
        self.assertEqual((args["dsa_prefill_backend"], args["dsa_decode_backend"]),
                         ("flashmla_sparse", "trtllm"))
        self.assertEqual(result.ranks[0].kv_bytes_per_token, 78 * (1152 + 132))
        for name in ("a100-80gb", "rtx5090-32gb"):
            self.assertEqual(check(self.model, hardware=load_hardware(name), parallelism=plan,
                                   runtime=runtime).status, "unsupported")
        self.assertEqual(check(self.model, hardware=load_hardware("b200-180gb")).status, "unsupported")

    def test_public_dp_attention_search_keeps_blackwell_kv_layout_during_pp_optimization(self):
        path = Path(__file__).resolve().parents[1] / "examples/hardware/b200-glm51-dp-attention-search.json"
        document = json.loads(path.read_text())
        result = evaluate_document(document, mode="search")
        self.assertEqual(result["minimum_num_gpus"], 6)
        candidate = result["candidates"][0]
        self.assertEqual(candidate["parallelism"]["dp"], 2)
        document.pop("search")
        checked = evaluate_document({**document, "parallelism": candidate["parallelism"]})
        self.assertEqual(checked["summary"]["kv_bytes_per_token_logical"], 78 * (576 + 132))
        self.assertEqual(checked["summary"]["engine_args"]["dsa_decode_backend"], "trtllm")
        for rank in checked["ranks"]:
            self.assertEqual(rank["kv_bytes_per_token"],
                             (rank["layer_end"] - rank["layer_start"]) * (576 + 132))

    def test_pd_rejects_hopper_to_blackwell_fp8_cache_without_conversion(self):
        runtime = RuntimeProfile.sglang(sglang_dp_attention=True)
        plan = Parallelism(4, 2, 8, 1, 8)
        p = PDPool(tuple(f"p/gpu{i}" for i in range(8)), load_hardware("h200-141gb"),
                   plan, runtime, weight_quantization="fp8_block", kv_quantization="fp8")
        d = replace(p, gpu_ids=tuple(f"d/gpu{i}" for i in range(8)), hardware=load_hardware("b200-180gb"))
        result = check_pd(self.model, prefill=p, decode=d)
        self.assertEqual(result.status, "unsupported")
        self.assertIn("different KV storage layouts", result.reasons[0])
        same_layout = check_pd(self.model, prefill=replace(p, hardware=load_hardware("gb200-186gb")), decode=d)
        self.assertEqual(same_layout.status, "pass")
        self.assertEqual(same_layout.summary["handoff_logical_payload_bytes_per_request"],
                         8192 * 78 * (576 + 132))

    def test_pd_uses_common_page_size_across_hopper_and_blackwell(self):
        model = load_model("qwen3-235b-a22b")
        prefill = PDPool(tuple(f"p/gpu{i}" for i in range(8)), hardware=load_hardware("h800-80gb"),
                         runtime=self.runtime, parallelism=Parallelism(tp=8, ep=8, num_gpus=8))
        decode = replace(prefill, gpu_ids=tuple(f"d/gpu{i}" for i in range(8)),
                         hardware=load_hardware("b200-180gb"))
        result = check_pd(model, prefill=prefill, decode=decode)
        self.assertEqual(result.status, "pass")
        self.assertEqual(result.prefill.summary["engine_args"]["page_size"], 64)
        self.assertEqual(result.decode.summary["engine_args"]["page_size"], 64)

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
        self.assertGreater(result.compatibility_counts["unsupported"], 0)
        self.assertTrue(all(r.static_candidate for r in result.candidates))
        self.assertTrue(all(r.parallelism.dp==1 for r in result.candidates))
        self.assertTrue(any(r.parallelism.ep==2 for r in result.candidates))

    def test_json_engine_selection_uses_sglang_defaults(self):
        document = {"model":"glm-5.1", "runtime":{"engine":"sglang"},
                    "parallelism":{"tp":8,"dp":1,"ep":8,"pp":2,"num_gpus":16}}
        result = evaluate_document(document)
        self.assertTrue(result["memory_pass"])
        self.assertEqual(result["summary"]["engine_version"], "0.5.23")
        self.assertEqual(result["summary"]["runtime_profile"], self.runtime.name)

    def test_dense_gqa_dp_attention_requires_inspected_model_mapping(self):
        model = from_config(dict(model_type="llama", hidden_size=512, intermediate_size=1024,
                                 num_hidden_layers=4,num_attention_heads=8,num_key_value_heads=2,
                                 head_dim=64,vocab_size=1024,max_position_embeddings=32768))
        self.assertEqual(kv_per_token(model, runtime=self.runtime,kv_quantization="fp8")["total"],1024)
        result = check(model, runtime=self.runtime, parallelism=Parallelism(2,2,1,1,4),weight_quantization="bf16")
        self.assertTrue(result.memory_pass)
        for model_type in ("llama", "qwen2"):
            with self.subTest(model_type=model_type):
                dense = from_config({**model.config, "model_type": model_type})
                result = check(dense, runtime=RuntimeProfile.sglang(sglang_dp_attention=True),
                               parallelism=Parallelism(2, 2, 1, 1, 4), weight_quantization="bf16")
                self.assertEqual(result.status, "unsupported")
                self.assertTrue(any("dense model_type=" in reason for reason in result.reasons))
                self.assertFalse(any("only for GLM MLA/DSA" in reason for reason in result.reasons))
        found = search(model, runtime=self.runtime, tp_values=[2], pp_values=[1],
                       max_gpus=8, weight_quantization="bf16")
        self.assertEqual([c.parallelism.dp for c in found.candidates], [1])
        self.assertGreater(found.compatibility_counts.get("unsupported", 0), 0)


if __name__ == "__main__":
    unittest.main()
