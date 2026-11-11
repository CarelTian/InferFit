from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from importlib.resources import files
from pathlib import Path

from inferfit import (GiB, Hardware, Parallelism, RuntimeProfile, Workload,
                                check, from_config, glm_profile, kv_per_token,
                                load_glm51, logical_weight_bytes, optimize_pipeline, search)
from inferfit.cli import evaluate_document

ROOT = Path(__file__).resolve().parents[1]


def small_model(**updates):
    config = dict(model_type="llama", hidden_size=512, intermediate_size=1024,
                  num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2,
                  head_dim=64, vocab_size=1024, max_position_embeddings=32768,
                  tie_word_embeddings=False)
    config.update(updates)
    return from_config(config)


class OfficialMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_glm51()

    def test_architecture_inventory_matches_every_official_template(self):
        full = glm_profile(self.model.config, include_mtp=True)
        for mode in ("bf16", "fp8"):
            meta = json.loads(files("inferfit").joinpath(f"data/{mode}-tensor-templates.json").read_text())
            counted = {}
            for tensor in full.tensors:
                name = re.sub(r"\.\d+(?=\.)", ".#", tensor.name).replace(".*.", ".#.")
                self.assertEqual(tuple(meta["templates"][name]["shape"]), tensor.shape)
                counted[name] = counted.get(name, 0) + tensor.multiplicity
            # All checkpoint tensors must be accounted for, including FP8 scales.
            for name, count in meta["counts"].items():
                if name.endswith("weight_scale_inv"):
                    weight_name = name.removesuffix("weight_scale_inv") + "weight"
                    self.assertEqual(counted[weight_name], count)
                else:
                    self.assertEqual(counted[name], count)

    def test_inventory_matches_official_serialized_byte_counts(self):
        full = glm_profile(self.model.config, include_mtp=True)
        self.assertEqual(logical_weight_bytes(full, "bf16"), 1_507_728_316_928)
        self.assertEqual(logical_weight_bytes(full, "fp8_block"), 756_162_687_872)
        self.assertEqual(logical_weight_bytes(self.model, "bf16"), 1_487_822_475_264)
        self.assertEqual(logical_weight_bytes(self.model, "fp8_block"), 746_130_054_912)

    def test_official_header_bytes_independently_sum_to_index(self):
        for mode in ("bf16", "fp8"):
            meta = json.loads(files("inferfit").joinpath(f"data/{mode}-tensor-templates.json").read_text())
            widths = {"BF16": 2, "F32": 4, "F8_E4M3": 1}
            total = sum(math.prod(meta["templates"][n]["shape"]) * widths[meta["templates"][n]["dtype"]] * count
                        for n, count in meta["counts"].items())
            self.assertEqual(total, meta["checkpoint_total_bytes"])

    def test_dsa_bf16_token_includes_index(self):
        self.assertEqual(kv_per_token(self.model)["total"], 100_152)
        self.assertEqual(kv_per_token(self.model)["index_scales"], 78 * 4)

    def test_dsa_fp8_retains_bf16_rope_and_scale_overhead(self):
        result = kv_per_token(self.model, kv_quantization="fp8")
        self.assertEqual(result["total"], 61_464)
        self.assertEqual(result["rope"], 78 * 128)
        self.assertEqual(result["mla_scales"], 78 * 16)
        self.assertNotEqual(result["total"], 100_152 // 2)

    def test_mla_replicates_cache_across_tp(self):
        self.assertEqual(kv_per_token(self.model, tp=1), kv_per_token(self.model, tp=8))
        self.assertEqual(kv_per_token(self.model, tp=8, layers=39)["total"], 50_076)


class CapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_glm51()
        cls.p16 = Parallelism(tp=8, dp=1, ep=8, pp=2, num_gpus=16)

    def test_h800_eight_cards_cannot_even_hold_fp8_weights(self):
        result = check(self.model)
        self.assertEqual(result.status, "fail")
        self.assertFalse(result.memory_pass)
        self.assertEqual(result.summary["weight_capacity_lower_bound_gpus"], 10)

    def test_h800_sixteen_card_candidate_and_pp_bottleneck(self):
        result = check(self.model, parallelism=self.p16)
        self.assertTrue(result.memory_pass)
        self.assertEqual(len(result.ranks), 16)
        self.assertGreater(result.ranks[8].weight_bytes, result.ranks[0].weight_bytes)
        self.assertEqual(result.summary["bottleneck_rank"], 8)
        self.assertFalse(result.summary["deployment_verified"])
        self.assertEqual(result.summary["engine_env"]["VLLM_PP_LAYER_PARTITION"], "39,39")
        self.assertGreater(result.ranks[0].runtime_breakdown["mla_bf16_projection_copies"], 0)

    def test_bf16_needs_more_than_sixteen_cards(self):
        self.assertFalse(check(self.model, parallelism=self.p16, weight_quantization="bf16").memory_pass)
        p32 = Parallelism(tp=8, dp=1, ep=8, pp=4, num_gpus=32)
        self.assertTrue(check(self.model, parallelism=p32, weight_quantization="bf16").memory_pass)

    def test_ep_reuses_mesh_not_extra_cards(self):
        bad = replace(self.p16, num_gpus=128)
        self.assertEqual(check(self.model, parallelism=bad).status, "invalid")
        bad_ep = replace(self.p16, ep=2)
        self.assertEqual(check(self.model, parallelism=bad_ep).status, "unsupported")

    def test_dp_replicates_weights_without_ep(self):
        a = check(self.model, parallelism=replace(self.p16, ep=1))
        b = check(self.model, parallelism=replace(self.p16, ep=1, dp=2, num_gpus=32))
        self.assertEqual(a.ranks[0].weight_bytes, b.ranks[0].weight_bytes)
        self.assertEqual(b.summary["placed_weight_bytes_total"], a.summary["placed_weight_bytes_total"] * 2)
        self.assertEqual(a.ranks[0].kv_required_bytes, b.ranks[0].kv_required_bytes)

    def test_ep_can_shard_experts_across_dp(self):
        no_ep = check(self.model, parallelism=Parallelism(8, 2, 1, 1, 16))
        ep = check(self.model, parallelism=Parallelism(8, 2, 16, 1, 16))
        self.assertLess(ep.ranks[0].weight_bytes, no_ep.ranks[0].weight_bytes)
        # Attention KV is still private to each attention replica.
        self.assertEqual(ep.ranks[0].kv_bytes_per_token, no_ep.ranks[0].kv_bytes_per_token)
        self.assertTrue(ep.memory_pass)

    def test_block_rounding_and_null_block(self):
        w = Workload(context_tokens=65, concurrency_per_dp=3)
        result = check(self.model, parallelism=self.p16, workload=w)
        self.assertEqual(result.ranks[0].kv_allocated_tokens, 3 * 128 + 64)
        self.assertEqual(result.summary["block_rounded_target_tokens_per_dp"], 384)

    def test_increasing_context_or_concurrency_reduces_headroom(self):
        a = check(self.model, parallelism=self.p16)
        b = check(self.model, parallelism=self.p16, workload=Workload(context_tokens=32768, concurrency_per_dp=4))
        self.assertLess(b.summary["minimum_headroom_bytes"], a.summary["minimum_headroom_bytes"])

    def test_lower_precision_kv_releases_budget(self):
        workload = Workload(context_tokens=32768, concurrency_per_dp=4)
        a = check(self.model, parallelism=self.p16, workload=workload)
        b = check(self.model, parallelism=self.p16, workload=workload, kv_quantization="fp8")
        self.assertGreater(b.summary["minimum_headroom_bytes"], a.summary["minimum_headroom_bytes"])
        self.assertEqual(b.ranks[0].kv_bytes_per_token, 30_732)

    def test_kv_precision_does_not_reduce_loading_peak(self):
        a = check(self.model, parallelism=self.p16, runtime=RuntimeProfile(load_workspace_bytes=10*GiB))
        b = check(self.model, parallelism=self.p16, runtime=RuntimeProfile(load_workspace_bytes=10*GiB), kv_quantization="fp8")
        self.assertLess(b.ranks[0].kv_required_bytes, a.ranks[0].kv_required_bytes)
        self.assertEqual(a.ranks[0].load_peak_bytes, b.ranks[0].load_peak_bytes)
        self.assertEqual(a.summary["minimum_headroom_bytes"], b.summary["minimum_headroom_bytes"])

    def test_external_usage_and_utilization_not_double_counted(self):
        a = check(self.model, parallelism=self.p16)
        b = check(self.model, parallelism=self.p16, hardware=Hardware(external_used_bytes_per_gpu=1_000_000_000))
        self.assertEqual(a.ranks[0].budget_bytes, b.ranks[0].budget_bytes)
        c = check(self.model, parallelism=self.p16, hardware=Hardware(external_used_bytes_per_gpu=20_000_000_000))
        self.assertEqual(c.ranks[0].budget_bytes, 60_000_000_000 - GiB)

    def test_loading_peak_checked_separately(self):
        profile = RuntimeProfile(load_workspace_bytes=100 * GiB)
        result = check(self.model, parallelism=self.p16, runtime=profile)
        self.assertFalse(result.memory_pass)
        self.assertGreater(result.ranks[0].load_peak_bytes, result.ranks[0].serving_peak_bytes)

    def test_illegal_inputs_and_unsupported_features(self):
        cases = [dict(parallelism=replace(self.p16, tp=0)),
                 dict(workload=Workload(context_tokens=202753)),
                 dict(hardware=Hardware(utilization=float("nan"))),
                 dict(hardware=Hardware(memory_bytes=-1)),
                 dict(parallelism=replace(self.p16, stage_layers=(20, 20))),
                 dict(workload=Workload(concurrency_per_dp=128, max_batch_tokens=64))]
        for kwargs in cases:
            self.assertEqual(check(self.model, **kwargs).status, "invalid")
        for kwargs in [dict(kv_quantization="int4"), dict(weight_quantization="nvfp4"),
                       dict(runtime=RuntimeProfile(enforce_eager=False)),
                       dict(workload=Workload(block_size=16)),
                       dict(hardware=Hardware(inter_node_network=False), parallelism=self.p16)]:
            self.assertEqual(check(self.model, **kwargs).status, "unsupported")

    def test_quantized_shard_alignment_rejected(self):
        p = Parallelism(tp=32, pp=1, dp=1, ep=1, num_gpus=32)
        result = check(self.model, parallelism=p, hardware=Hardware(allow_cross_node_tp=True))
        self.assertEqual(result.status, "unsupported")
        self.assertIn("128-element", result.reasons[0])

    def test_exact_capacity_boundary(self):
        model = small_model()
        parallel = Parallelism(1, 1, 1, 1, 1)
        runtime = RuntimeProfile(cuda_context_bytes=0, communication_bytes=0, allocator_bytes=0,
                                 activation_fixed_bytes=0, activation_copies=0, load_workspace_bytes=0)
        a = check(model, parallelism=parallel, runtime=runtime, weight_quantization="bf16")
        peak = a.ranks[0].peak_bytes
        h = Hardware(memory_bytes=peak, utilization=1, safety_margin_bytes=0)
        b = check(model, hardware=h, parallelism=parallel, runtime=runtime, weight_quantization="bf16")
        self.assertTrue(b.memory_pass)
        self.assertEqual(b.ranks[0].headroom_bytes, 0)
        self.assertFalse(check(model, hardware=replace(h, memory_bytes=peak-1), parallelism=parallel,
                               runtime=runtime, weight_quantization="bf16").memory_pass)


class GenericAndSearchTests(unittest.TestCase):
    def test_gqa_cache_sharding_and_replication(self):
        model = small_model()
        self.assertEqual(kv_per_token(model)["total"], 2048)
        self.assertEqual(kv_per_token(model, tp=2)["total"], 1024)
        self.assertEqual(kv_per_token(model, tp=4)["total"], 1024)

    def test_tied_embeddings_replicate_at_pp_endpoints(self):
        model = small_model(tie_word_embeddings=True)
        a = check(model, parallelism=Parallelism(1, 1, 1, 1, 1), weight_quantization="bf16")
        b = check(model, parallelism=Parallelism(1, 1, 1, 2, 2), weight_quantization="bf16")
        self.assertEqual(b.summary["placed_weight_bytes_total"] - a.summary["placed_weight_bytes_total"], 1024*512*2)

    def test_fail_closed_on_unknown_or_hybrid_models(self):
        with self.assertRaises(ValueError):
            from_config({"model_type": "unknown"})
        with self.assertRaises(ValueError):
            small_model(sliding_window=4096)

    def test_search_returns_only_valid_conditional_passes(self):
        result = search(load_glm51(), gpu_counts=[8, 16, 24, 32], max_gpus=32, limit=10)
        self.assertEqual(result.minimum_num_gpus, 16)
        self.assertGreater(result.counts["fail"], 0)
        self.assertTrue(all(r.memory_pass and r.status == "pass" for r in result.candidates))
        self.assertEqual(result.evaluated, sum(result.counts.values()))
        self.assertEqual([r.parallelism.num_gpus for r in result.candidates], sorted(r.parallelism.num_gpus for r in result.candidates))

    def test_pipeline_optimizer_finds_twelve_card_candidate(self):
        model = load_glm51()
        plan = Parallelism(4, 1, 4, 3, 12)
        self.assertFalse(check(model, parallelism=plan).memory_pass)
        optimized = optimize_pipeline(model, parallelism=plan)
        self.assertTrue(optimized.memory_pass)
        self.assertEqual(optimized.summary["stage_layers"], [28, 25, 25])
        self.assertEqual(optimized.summary["engine_env"]["VLLM_PP_LAYER_PARTITION"], "28,25,25")
        result = search(model, gpu_counts=[8, 12, 16], max_gpus=16)
        self.assertEqual(result.minimum_num_gpus, 12)

    def test_partition_dp_matches_exhaustive_small_model(self):
        model = small_model(num_hidden_layers=8)
        plan = Parallelism(2, 1, 1, 3, 6)
        best = math.inf
        for first in range(1, 7):
            for second in range(1, 8 - first):
                third = 8 - first - second
                result = check(model, parallelism=replace(plan, stage_layers=(first, second, third)), weight_quantization="bf16")
                best = min(best, max(rank.peak_bytes for rank in result.ranks))
        optimized = optimize_pipeline(model, parallelism=plan, weight_quantization="bf16")
        self.assertEqual(max(rank.peak_bytes for rank in optimized.ranks), best)

    def test_sequence_parallel_shared_experts_are_replicated(self):
        model = load_glm51()
        no_ep = check(model, parallelism=Parallelism(8, 2, 1, 1, 16))
        ep = check(model, parallelism=Parallelism(8, 2, 16, 1, 16))
        # 75 MoE layers: saved routed shards minus newly replicated shared MLPs.
        self.assertEqual(no_ep.ranks[0].weight_bytes - ep.ranks[0].weight_bytes, 42_831_676_800)

    def test_rank_coordinates_and_boolean_validation(self):
        result = check(load_glm51(), parallelism=Parallelism(8, 2, 16, 2, 32))
        self.assertEqual((result.ranks[8].dp_rank, result.ranks[8].pp_rank), (0, 1))
        self.assertEqual((result.ranks[16].dp_rank, result.ranks[16].pp_rank), (1, 0))
        self.assertEqual(check(load_glm51(), hardware=Hardware(inter_node_network="false")).status, "invalid")

    def test_search_no_candidate_and_bad_degrees(self):
        result = search(load_glm51(), max_gpus=8)
        self.assertEqual(result.minimum_num_gpus, None)
        self.assertEqual(result.candidates, [])
        with self.assertRaises(ValueError):
            search(load_glm51(), tp_values=[0])

    def test_json_examples_and_reject_unknown_options(self):
        doc = json.loads((ROOT / "examples/h800-glm51-check.json").read_text())
        self.assertTrue(evaluate_document(doc)["memory_pass"])
        with self.assertRaises(ValueError):
            evaluate_document({**doc, "typo": 1})
        with self.assertRaises(ValueError):
            evaluate_document({**doc, "runtime": {"typo": 1}})

    def test_cli_exit_status_and_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenario.json"
            path.write_text(json.dumps({"model": "glm-5.1"}))
            done = subprocess.run([sys.executable, "-m", "inferfit", "check", str(path)],
                                  capture_output=True, text=True)
            self.assertEqual(done.returncode, 2)
            self.assertFalse(json.loads(done.stdout)["memory_pass"])
            path.write_text('{"typo": 1}')
            done = subprocess.run([sys.executable, "-m", "inferfit", "check", str(path)],
                                  capture_output=True, text=True)
            self.assertEqual(done.returncode, 1)
            self.assertEqual(json.loads(done.stderr)["status"], "input_error")


if __name__ == "__main__":
    unittest.main()
