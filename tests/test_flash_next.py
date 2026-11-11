import copy
from dataclasses import replace
from importlib.resources import files
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from inferfit import (Hardware, Parallelism, RuntimeProfile, Workload, check,
    from_config, kv_per_token, load_model, load_model_profile, logical_weight_bytes,
    model_profile_to_dict, recurrent_state_bytes, search, UnsupportedModelError)
from inferfit.pd import check_pd, PDPool
from inferfit.states.qwen_flash_next import flash_next_fixed_state


class FlashNextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model("qwen3.8-flash-next")
        cls.runtime = RuntimeProfile.sglang()
        cls.plan = Parallelism(tp=8, ep=8, num_gpus=8)
        cls.meta = json.loads(files("inferfit").joinpath("data/qwen38-flash-next-metadata.json").read_text())

    def result(self, **overrides):
        kwargs = dict(runtime=self.runtime, parallelism=self.plan, weight_quantization="bf16")
        kwargs.update(overrides)
        return check(self.model, **kwargs)

    def test_all_checkpoint_headers_match_official_total(self):
        rows = self.meta["tensors"]
        self.assertEqual(len(rows), 1658)
        self.assertEqual(self.meta["shard_count"], 131)
        self.assertFalse(self.meta["tensor_payload_downloaded"])
        size = lambda row: math.prod(row["shape"]) * {"BF16": 2, "I64": 8}[row["dtype"]]
        self.assertEqual(sum(map(size, rows)), 359_999_963_128)
        self.assertEqual(sum(size(row) for row in rows if row["name"].startswith("mtp.")), 5_214_301_696)
        # Every base tensor, including vision, HC, PLE and I64 buffers; one regenerated BF16 scalar.
        self.assertEqual(logical_weight_bytes(self.model), 354_785_661_432 + 2)

    def test_ngram_hyperconnection_and_vision_are_resident(self):
        tensors = {t.name: t for t in self.model.tensors}
        ngram = [t for t in tensors.values() if t.name.endswith("ngram_embedding.weight")]
        self.assertEqual(len(ngram), 1)
        self.assertEqual(ngram[0].shape, (320_001_536, 160))
        self.assertEqual(math.prod(ngram[0].shape) * 2, 102_400_491_520)
        self.assertEqual(ngram[0].layer, 1)  # Config PLE layer IDs are one-based.
        vision = [t for t in tensors.values() if t.name.startswith("model.visual.")]
        self.assertTrue(vision)
        self.assertTrue(all(t.placement == "replicated" for t in vision))
        self.assertTrue(any("hyper_connection" in name for name in tensors))
        self.assertFalse(any(name.startswith("mtp.") for name in tensors))

    def test_qsa_main_kv_and_compressed_index_are_separate(self):
        for tp, expected in [(1, 25_344), (2, 13_056), (4, 13_056), (8, 13_056)]:
            result = kv_per_token(self.model, runtime=self.runtime, tp=tp)
            self.assertEqual(result["qsa_compressed_keys"], 12 * 128 * 2 // 4)
            self.assertEqual(result["total"], expected)
        self.assertEqual(kv_per_token(self.model, runtime=self.runtime, layers=3)["total"], 0)

    def test_fixed_state_has_gdn_ple_and_pending_ring(self):
        self.assertEqual(recurrent_state_bytes(self.model), 36 * ((2 * 16 * 128 + 48 * 128) * 3 * 2 + 48 * 128 * 128 * 4))
        fixed = flash_next_fixed_state(self.model)
        self.assertEqual(fixed["ple_short_conv_state_pool"], 10240 * 9 * 2)
        self.assertEqual(fixed["ple_ngram_context_pool"], 16)
        self.assertEqual(fixed["qsa_pending_key_pool"], 12 * 4 * 128 * 2)
        self.assertEqual(fixed["qsa_pending_position_pool"], 4 * 3 * 8)
        self.assertEqual(flash_next_fixed_state(self.model, layer_start=2, layers=1)["ple_short_conv_state_pool"], 0)

    def test_longer_context_grows_history_not_fixed_state(self):
        a = self.result(workload=Workload(context_tokens=4096))
        b = self.result(workload=Workload(context_tokens=8192))
        self.assertEqual((a.status, b.status), ("pass", "pass"))
        for key in ("recurrent_state_pool", *flash_next_fixed_state(self.model)):
            self.assertEqual(a.ranks[0].runtime_breakdown[key], b.ranks[0].runtime_breakdown[key])
        self.assertEqual(b.ranks[0].kv_required_bytes - a.ranks[0].kv_required_bytes, 4096 * 13056)
        # Sparse budget 2048 does not cap retained main KV history to 2048 tokens.
        self.assertEqual(b.ranks[0].kv_allocated_tokens, 8192 + 64)

    def test_boundary_and_native_pool_hints(self):
        initial = self.result()
        self.assertEqual(initial.status, "pass")
        peak = initial.ranks[0].peak_bytes
        h = Hardware(memory_bytes=peak, utilization=1, safety_margin_bytes=0)
        self.assertEqual(self.result(hardware=h).status, "pass")
        self.assertEqual(self.result(hardware=replace(h, memory_bytes=peak-1)).status, "fail")
        args = initial.summary["engine_args"]
        self.assertEqual(args["max_mamba_cache_size"], 1)
        self.assertEqual(args["qsa_indexer_dtype"], "bfloat16")
        self.assertTrue(args["disable_radix_cache"])
        self.assertEqual(initial.summary["vision_weight_placement"], "replicated_upper_bound")

    def test_search_is_available_and_rejects_unverified_layouts(self):
        result = search(self.model, runtime=self.runtime, weight_quantization="bf16",
                        tp_values=[2, 4, 8], dp_values=[1], pp_values=[1], max_gpus=8)
        self.assertEqual(result.minimum_num_gpus, 8)
        self.assertTrue(all(c.parallelism.pp == 1 and c.parallelism.dp == 1 for c in result.candidates))
        for kwargs in (
            dict(weight_quantization="fp8"), dict(kv_quantization="fp8"),
            dict(runtime=RuntimeProfile.sglang(version="0.5.10")),
            dict(parallelism=Parallelism(tp=4, pp=2, num_gpus=8)),
            dict(parallelism=Parallelism(tp=4, dp=2, num_gpus=8)),
            dict(workload=Workload(block_size=2)),
        ):
            self.assertEqual(self.result(**kwargs).status, "unsupported")
        with self.assertRaisesRegex(ValueError, "BF16"):
            logical_weight_bytes(self.model, "fp8")

    def test_config_overrides_fail_closed_and_profiles_roundtrip(self):
        c = json.loads(files("inferfit").joinpath("data/qwen38-flash-next-config.json").read_text())
        self.assertEqual(from_config(c).architecture, "gdn_qsa_moe")
        modified = copy.deepcopy(c)
        modified["text_config"]["ple_offload_embedding"] = True
        with self.assertRaisesRegex(UnsupportedModelError, "pinned"):
            from_config(modified)
        profile = model_profile_to_dict(self.model)
        self.assertEqual(logical_weight_bytes(load_model_profile(profile)), logical_weight_bytes(self.model))

    def test_cli_pinned_model_check_and_pd_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            scenario = Path(directory) / "scenario.json"
            scenario.write_text(json.dumps(dict(model="qwen3.8-flash-next", runtime=dict(engine="sglang", version="0.5.23"), weight_quantization="bf16", parallelism=dict(tp=8, ep=8, num_gpus=8))))
            result = subprocess.run([sys.executable, "-m", "inferfit", "check", str(scenario)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["status"], "pass")
        pool = PDPool(tuple(f"gpu{i}" for i in range(8)))
        self.assertEqual(check_pd(self.model, prefill=pool, decode=pool).status, "unsupported")


if __name__ == "__main__":
    unittest.main()
