"""Exact cost/rounding parity and deterministic PP optimization regression checks."""
from dataclasses import replace
from itertools import combinations
import unittest
from unittest.mock import patch

from inferfit import (Hardware, Parallelism, RuntimeProfile, Workload, check,
                      from_config, load_glm51, load_model, optimize_pipeline)
from inferfit.states import get_state_model
from inferfit.states.base import StandardStateModel, StateAllocation
from inferfit.states.deepseek_v4 import DeepSeekV4StateModel


def dense_model(layers=8):
    return from_config(dict(model_type="llama", hidden_size=32, num_hidden_layers=layers,
        vocab_size=128, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, intermediate_size=64, max_position_embeddings=32768,
        tie_word_embeddings=False, attention_bias=False))


class PartitionCostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dense = dense_model()
        cls.cases = [
            (cls.dense, RuntimeProfile(), "bf16", 64),
            (load_glm51(), RuntimeProfile.sglang(), "fp8", 64),
            (load_model("qwen3.8-2.4t-a95b"), RuntimeProfile.sglang(), "bf16", 64),
            (load_model("qwen3.8-flash-next"), RuntimeProfile.sglang(), "bf16", 64),
            (load_model("deepseek-v4-flash"), RuntimeProfile.sglang(), "fp8", 256),
            (load_model("deepseek-v4-pro"), RuntimeProfile.sglang(), "fp8", 256),
        ]

    def test_all_intervals_match_allocator_with_fractional_overheads(self):
        plan = Parallelism(tp=2, ep=2, pp=2, num_gpus=4)
        for model, runtime, kv, block in self.cases:
            state = get_state_model(model)
            workload = Workload(context_tokens=8193, concurrency_per_dp=3, block_size=block)
            for ratio in (1.0, 1.0000001, 1.03):
                with self.subTest(model=model.model_id, overhead=ratio):
                    r = replace(runtime, kv_pool_overhead_ratio=ratio)
                    cost = state.prepare_interval_cost(model, plan, workload, r, kv)
                    n = model.config["num_hidden_layers"]
                    for start in range(n):
                        for end in range(start + 1, n + 1):
                            expected = state.allocate(model, plan, workload, r, kv, start, end-start).total_bytes
                            self.assertEqual(cost(start, end), expected, (model.model_id, ratio, start, end))

    def test_prepared_hot_path_does_not_allocate_state_objects(self):
        # A call-count invariant, rather than a flaky elapsed-time threshold.
        for model, runtime, kv, block in self.cases:
            if model.architecture == "gdn_qsa_moe":
                continue  # QSA has a deliberately general memoized fallback.
            with self.subTest(model=model.model_id):
                state = get_state_model(model)
                n = model.config["num_hidden_layers"]
                with patch.object(state, "allocate", wraps=state.allocate) as allocate:
                    cost = state.prepare_interval_cost(model, Parallelism(tp=2, ep=2, num_gpus=2),
                                                       Workload(block_size=block), runtime, kv)
                    setup_calls = allocate.call_count
                    self.assertLessEqual(setup_calls, n)
                    for start in range(n):
                        for end in range(start+1, n+1):
                            cost(start, end)
                    self.assertEqual(allocate.call_count, setup_calls)

    def test_rounding_is_per_combined_block_not_per_layer(self):
        state = get_state_model(self.dense)
        p, w = Parallelism(tp=1, num_gpus=1), Workload(context_tokens=129)
        r = RuntimeProfile(kv_pool_overhead_ratio=1.0000001)
        cost = state.prepare_interval_cost(self.dense, p, w, r, "bf16")
        combined = state.allocate(self.dense, p, w, r, "bf16", 0, 2).total_bytes
        individual = sum(state.allocate(self.dense, p, w, r, "bf16", i, 1).total_bytes for i in range(2))
        self.assertEqual(cost(0, 2), combined)
        self.assertLess(combined, individual)

    def test_optimizer_matches_exhaustive_partitions_and_peak_phases(self):
        n = self.dense.config["num_hidden_layers"]
        for pp in (2, 3, 4, n):
            plan = Parallelism(tp=1, ep=1, pp=pp, num_gpus=pp)
            for runtime in (RuntimeProfile(kv_pool_overhead_ratio=1.03),
                            RuntimeProfile(load_workspace_bytes=100 * 1024**3)):
                options = dict(hardware=Hardware(memory_bytes=10**15), runtime=runtime,
                               workload=Workload(context_tokens=8193), weight_quantization="bf16")
                peaks = []
                for cuts in combinations(range(1, n), pp-1):
                    bounds = (0,) + cuts + (n,)
                    counts = tuple(b-a for a, b in zip(bounds, bounds[1:]))
                    result = check(self.dense, parallelism=replace(plan, stage_layers=counts), **options)
                    peaks.append(max(rank.peak_bytes for rank in result.ranks))
                optimized = optimize_pipeline(self.dense, parallelism=plan, **options)
                self.assertEqual(max(rank.peak_bytes for rank in optimized.ranks), min(peaks))

    def test_equal_cost_ties_keep_earliest_predecessor(self):
        for layers, pp, expected in ((7, 2, [3, 4]), (8, 3, [2, 3, 3]), (6, 6, [1]*6)):
            model = dense_model(layers)
            # Symmetric stages with no endpoint weights deliberately create ties.
            model = replace(model, tensors=tuple(t for t in model.tensors if t.layer is not None))
            result = optimize_pipeline(model, parallelism=Parallelism(tp=1, pp=pp, num_gpus=pp),
                                       weight_quantization="bf16")
            self.assertEqual(result.summary["stage_layers"], expected)

    def test_future_nonadditive_layout_fallback_is_scoped_and_memoized(self):
        class FutureState(StandardStateModel):
            def allocate(self, model, p, w, r, kv, start, count, *, tokens=None):
                return StateAllocation({"total": count}, 0, {"nonlinear": count**2 + start},
                                       {"request_workspace": w.context_tokens})
        state = FutureState()
        model = replace(self.dense, architecture="future-nonadditive")
        p, r = Parallelism(), RuntimeProfile()
        with patch.object(state, "allocate", wraps=state.allocate) as allocate:
            a = state.prepare_interval_cost(model, p, Workload(context_tokens=1), r, "bf16")
            self.assertEqual((a(1, 4), a(1, 4)), (11, 11))
            self.assertEqual(allocate.call_count, 1)
            b = state.prepare_interval_cost(model, p, Workload(context_tokens=2), r, "bf16")
            self.assertEqual(b(1, 4), 12)
            self.assertEqual(allocate.call_count, 2)

    def test_new_v4_pool_component_uses_exact_fallback(self):
        class FutureV4(DeepSeekV4StateModel):
            def allocate(self, model, p, w, r, kv, start, count, *, tokens=None):
                base = super().allocate(model, p, w, r, kv, start, count, tokens=tokens)
                return replace(base, cache_parts={**base.cache_parts, "future_nonlinear_pool": count**2})
        state = FutureV4()
        model, runtime, kv, block = self.cases[-2]
        p, w = Parallelism(tp=2, ep=2, num_gpus=2), Workload(block_size=block)
        cost = state.prepare_interval_cost(model, p, w, runtime, kv)
        for start, end in ((0, 2), (0, 43), (2, 19), (20, 21)):
            self.assertEqual(cost(start, end), state.allocate(model, p, w, runtime, kv, start, end-start).total_bytes)


if __name__ == "__main__":
    unittest.main()
