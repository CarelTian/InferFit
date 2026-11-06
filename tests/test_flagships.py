import copy
from dataclasses import replace
import json
import subprocess
import sys
import unittest

from inferfit import (Hardware, Parallelism, RuntimeProfile, Workload, check,
    flagship_catalog, from_config, kv_per_token, load_model, logical_weight_bytes,
    model_profile_to_dict, load_model_profile, optimize_pipeline,
    recurrent_state_bytes, search, UnsupportedModelError)


class FlagshipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qwen = load_model("qwen3.8-2.4t-a95b")
        cls.glm = load_model("glm-5.3")
        cls.runtime = RuntimeProfile.sglang_flagship()

    def test_catalog_separates_recognition_from_capacity_support(self):
        entries = flagship_catalog()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[2]["support"], "recognized_only")
        with self.assertRaisesRegex(UnsupportedModelError, "FP4"):
            load_model(entries[2]["id"])
        with self.assertRaisesRegex(ValueError, "Unknown"):
            load_model("future-model")
        result = subprocess.run([sys.executable, "-m", "inferfit", "models"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout), entries)

    def test_qwen_full_kv_and_fixed_state_independent_oracles(self):
        # Official 92 layers: 23 GQA + 69 GDN. BF16 K/V, FP32 temporal state.
        self.assertEqual(kv_per_token(self.qwen, runtime=self.runtime)["total"], 23 * 2 * 4 * 256 * 2)
        per_gdn = (2 * 16 * 128 + 128 * 128) * 3 * 2 + 128 * 128 * 128 * 4
        self.assertEqual(recurrent_state_bytes(self.qwen), 69 * per_gdn)
        self.assertEqual(recurrent_state_bytes(self.qwen, tp=8), 69 * per_gdn // 8)
        self.assertEqual(kv_per_token(self.qwen, runtime=self.runtime, layers=3)["total"], 0)
        self.assertEqual(kv_per_token(self.qwen, runtime=self.runtime, layer_start=3, layers=1)["total"], 4096)

    def test_qwen_gate_and_shared_experts_reside_in_inventory(self):
        ts = {t.name: t for t in self.qwen.tensors}
        self.assertEqual(ts["model.layers.3.self_attn.q_proj.weight"].shape, (32768, 8192))
        self.assertEqual(ts["model.layers.0.linear_attn.A_log"].dtype, "F32")
        self.assertIn("model.layers.0.mlp.shared_expert_gate.weight", ts)
        self.assertEqual(sum(t.multiplicity for t in self.qwen.tensors if t.name.endswith("experts.*.gate_proj.weight")), 92 * 512)
        self.assertGreater(logical_weight_bytes(self.qwen), 4_800_000_000_000)

    def test_glm_shared_index_has_only_twenty_one_owners(self):
        owners = [t.layer for t in self.glm.tensors if t.name.endswith("indexer.wk.weight")]
        self.assertEqual(owners, [0, 1, 2] + list(range(6, 78, 4)))
        self.assertEqual(kv_per_token(self.glm, runtime=self.runtime)["total"], 78 * 1152 + 21 * 132)
        self.assertEqual(kv_per_token(self.glm, runtime=self.runtime, layer_start=3, layers=3)["index_keys"], 0)

    def test_old_engines_and_unverified_paths_reject(self):
        for model in (self.glm, self.qwen):
            for runtime in (RuntimeProfile(), RuntimeProfile.sglang()):
                r = check(model, runtime=runtime, weight_quantization="bf16")
                self.assertEqual(r.status, "unsupported")
                with self.assertRaisesRegex(ValueError, "0.5.23"):
                    kv_per_token(model, runtime=runtime)
        r = check(self.qwen, runtime=self.runtime, weight_quantization="fp8")
        self.assertEqual(r.status, "unsupported")
        r = check(self.qwen, runtime=self.runtime, parallelism=Parallelism(tp=32, ep=32, num_gpus=32), weight_quantization="bf16")
        self.assertEqual(r.status, "unsupported")
        r = check(self.glm, runtime=self.runtime, parallelism=Parallelism(tp=8, pp=2, num_gpus=16))
        self.assertEqual(r.status, "unsupported")

    def test_context_growth_does_not_multiply_recurrent_state(self):
        plan = Parallelism(tp=8, ep=8, num_gpus=8)
        kwargs = dict(hardware=Hardware(memory_bytes=10**15), parallelism=plan,
                      runtime=self.runtime, weight_quantization="bf16")
        a = check(self.qwen, workload=Workload(context_tokens=1024), **kwargs)
        b = check(self.qwen, workload=Workload(context_tokens=2048), **kwargs)
        self.assertEqual(a.status, "pass")
        self.assertEqual(a.ranks[0].runtime_breakdown["recurrent_state_pool"], 2 * recurrent_state_bytes(self.qwen, tp=8))
        self.assertEqual(a.ranks[0].runtime_breakdown["recurrent_state_pool"], b.ranks[0].runtime_breakdown["recurrent_state_pool"])
        self.assertEqual(b.ranks[0].kv_required_bytes - a.ranks[0].kv_required_bytes, 1024 * 23552)
        self.assertEqual(a.summary["engine_args"]["max_mamba_cache_size"], 1)

    def test_hybrid_pp_optimizer_matches_exhaustive_partitions(self):
        c = copy.deepcopy(self.qwen.config)
        c["num_hidden_layers"] = 8
        c["layer_types"] = c["layer_types"][:8]
        model = from_config(c)
        plan = Parallelism(tp=2, ep=2, pp=2, num_gpus=4)
        kwargs = dict(hardware=Hardware(memory_bytes=10**15), runtime=self.runtime, weight_quantization="bf16")
        peaks = []
        for split in range(1, 8):
            result = check(model, parallelism=replace(plan, stage_layers=(split, 8-split)), **kwargs)
            self.assertEqual(result.status, "pass")
            peaks.append(max(r.peak_bytes for r in result.ranks))
        best = optimize_pipeline(model, parallelism=plan, **kwargs)
        self.assertEqual(max(r.peak_bytes for r in best.ranks), min(peaks))
        self.assertEqual(load_model_profile(model_profile_to_dict(model)).architecture, "gdn_moe")

    def test_glm_fp8_candidate_and_search(self):
        kwargs = dict(hardware=Hardware(allow_cross_node_tp=True), runtime=self.runtime, weight_quantization="fp8")
        result = search(self.glm, tp_values=[8, 16], dp_values=[1], pp_values=[1], ep_values=[8], max_gpus=16, **kwargs)
        self.assertEqual(result.minimum_num_gpus, 16)
        self.assertFalse(result.candidates[0].summary["deployment_verified"])


if __name__ == "__main__":
    unittest.main()
