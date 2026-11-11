"""Independent capacity/compatibility axes, physical identity, and fallback safety."""
from dataclasses import replace
from itertools import combinations
import json
import subprocess
import sys
from unittest.mock import patch

import pytest

from inferfit import (Hardware, Parallelism, RuntimeProfile, Workload, MemoryAssumptions,
    analyze, search_plans, check, search, from_config, load_hardware, hardware_catalog,
    load_glm51, load_model, load_qwen3_235b)
from inferfit.backends import resolve_backend
from inferfit.cli import evaluate_document


def dense():
    return from_config(dict(model_type="llama", hidden_size=512, intermediate_size=1024,
        num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2, head_dim=64,
        vocab_size=1024, max_position_embeddings=32768, tie_word_embeddings=False))


P = Parallelism(2, 1, 1, 1, 2)
REF = MemoryAssumptions("runtime_reference", "sglang", "0.5.23")


def custom_hardware(gb=144):
    return Hardware(name=f"custom-test-{gb}gb", memory_bytes=gb * 10**9,
                    platform="custom", vendor="example", compute_capability=None)


def test_named_nvidia_catalog_and_capacity_override():
    assert set(hardware_catalog()["variants"]) == {
        "h800-80gb", "h20-96gb", "h100-sxm-80gb", "a100-80gb", "a100-40gb"}
    h = load_hardware("h100-sxm-80gb", memory_bytes=79 * 10**9)
    assert h.memory_bytes == 79 * 10**9
    assert h.memory_provenance == "caller_override"
    assert h.vendor == "nvidia"
    catalog = hardware_catalog()
    catalog["variants"].clear()
    assert len(hardware_catalog()["variants"]) == 5


@pytest.mark.parametrize("weight", ["bf16", "fp8"])
@pytest.mark.parametrize("kv,expected", [("bf16", 1024), ("fp8", 512)])
def test_unknown_device_still_prices_weights_and_per_token_kv(weight, kv, expected):
    result = analyze(dense(), hardware=Hardware(platform="new-device", compute_capability=None),
                     parallelism=P, weight_quantization=weight, kv_quantization=kv)
    assert result.capacity.status == "pass"
    assert result.capacity.summary["kv_bytes_per_token_per_rank_max"] == expected
    assert result.compatibility.status == "unknown"
    assert not result.static_candidate
    assert not result.to_dict()["deployment_verified"]
    assert result.capacity.summary["engine_args"] == {}


def test_generic_fp8_weights_count_unquantized_tensors_and_scales():
    model = dense()
    a, b = [analyze(model, hardware=custom_hardware(96), parallelism=P,
                     weight_quantization=fmt) for fmt in ("bf16", "fp8")]
    assert b.capacity.ranks[0].weight_bytes < a.capacity.ranks[0].weight_bytes
    assert b.capacity.ranks[0].weight_bytes > a.capacity.ranks[0].weight_bytes // 2
    assert b.compatibility.status == "unknown"


def test_negative_headroom_and_invalid_parallelism_are_not_hidden_by_backend_miss():
    h = replace(custom_hardware(96), memory_bytes=1000)
    result = analyze(dense(), hardware=h, parallelism=P, weight_quantization="bf16")
    assert result.capacity.status == "fail"
    assert result.capacity.summary["minimum_headroom_bytes"] < 0
    for p in (replace(P, num_gpus=3), replace(P, tp=0), replace(P, pp=2, num_gpus=4, stage_layers=(4, 1))):
        result = analyze(dense(), hardware=h, parallelism=p, weight_quantization="bf16")
        assert result.capacity.status == "invalid"
        assert not result.capacity.ranks


def test_known_unsupported_fp8_gpu_does_not_become_supported_after_fallback():
    result = analyze(dense(), hardware=load_hardware("a100-80gb"), parallelism=P,
                     weight_quantization="bf16", kv_quantization="fp8")
    assert result.capacity.status == "pass"
    assert result.compatibility.status == "unsupported"
    assert not result.static_candidate
    options = dict(hardware=load_hardware("a100-80gb"), weight_quantization="bf16",
                   kv_quantization="fp8", tp_values=[2], dp_values=[1], pp_values=[1])
    assert not search_plans(dense(), **options).candidates
    assert len(search_plans(dense(), **options, policy="capacity").candidates) == 1


def test_backend_match_requires_upstream_identity_not_cuda_api():
    h, r = load_hardware("h100-sxm-80gb"), RuntimeProfile.sglang()
    assert resolve_backend(h, r).kind == "inspected"
    for key in ("version", "distribution", "sdk_version", "build_revision", "image"):
        assert resolve_backend(h, replace(r, **{key: "other"})).kind == "unknown"
    assert resolve_backend(replace(h, vendor="example", device_api="cuda"), r).kind == "unknown"
    result = analyze(dense(), hardware=custom_hardware(), runtime=r, parallelism=P, weight_quantization="bf16")
    assert result.capacity.status == "pass"
    assert result.compatibility.backend_id is None
    assert result.compatibility.status == "unknown"


def test_model_specific_state_is_never_silently_replaced_by_gqa():
    model, h = load_glm51(), custom_hardware()
    result = analyze(model, hardware=h)
    assert result.capacity.status == "unknown"
    assert result.capacity.missing_data
    assert not result.capacity.ranks
    result = analyze(model, hardware=h, parallelism=Parallelism(8, 1, 8, 1, 8),
                     assumptions=MemoryAssumptions("runtime_reference", "sglang", "0.5.23"))
    assert result.capacity.status == "pass"
    assert result.capacity.summary["kv_bytes_per_token_logical"] == 100152
    assert result.capacity.basis == "runtime_reference"
    assert result.compatibility.status == "unknown"
    assert h.compute_capability is None


def test_native_v4_reference_keeps_compressor_states_and_scale_layouts():
    result = analyze(load_model("deepseek-v4-flash"), hardware=custom_hardware(),
        runtime=RuntimeProfile.sglang(), assumptions=REF, weight_quantization="native",
        kv_quantization="fp8", workload=Workload(block_size=256), parallelism=Parallelism(4, 1, 4, 1, 4))
    assert result.capacity.status == "pass"
    assert result.capacity.summary["kv_bytes_per_token_logical"] == 29055
    assert result.capacity.ranks[0].runtime_breakdown["c4_compressor_state"] > 0
    assert result.capacity.ranks[0].kv_allocation_breakdown["c128_compressed_kv"] > 0
    assert not result.static_candidate
    assert result.compatibility.status == "unknown"


def test_reference_is_explicit_pinned_and_preserves_missing_layout_constraints():
    with pytest.raises(ValueError):
        analyze(dense(), assumptions=MemoryAssumptions("runtime_reference", "sglang", "latest"))
    with pytest.raises(ValueError):
        analyze(dense(), assumptions=MemoryAssumptions("typo"))
    result = analyze(load_model("deepseek-v4-flash"), hardware=custom_hardware(),
        assumptions=REF, weight_quantization="native", kv_quantization="fp8", parallelism=Parallelism(4, 1, 4, 1, 4))
    assert result.capacity.status == "unknown"  # Default page=64 is not this layout.
    assert "page_size=256" in " ".join(result.capacity.missing_data)


def test_capacity_pp_optimization_matches_exhaustive_partitions():
    model = dense()
    options = dict(hardware=custom_hardware(96), weight_quantization="bf16")
    p = Parallelism(2, 1, 1, 3, 6)
    result = analyze(model, parallelism=p, optimize_pp=True, **options)
    peaks = []
    for cuts in combinations(range(1, 4), 2):
        bounds = (0, *cuts, 4)
        counts = tuple(b - a for a, b in zip(bounds, bounds[1:]))
        raw = analyze(model, parallelism=replace(p, stage_layers=counts), **options)
        peaks.append(max(x.peak_bytes for x in raw.capacity.ranks))
    assert max(x.peak_bytes for x in result.capacity.ranks) == min(peaks)
    assert result.capacity.summary["partition_optimization"]["predicted_peak_bytes"] == min(peaks)


@pytest.mark.parametrize("gb,minimum", [(96, 8), (144, 4)])
def test_qwen_capacity_search_has_conditional_minimum_and_diagnostics(gb, minimum):
    options = dict(hardware=custom_hardware(gb), runtime=RuntimeProfile.sglang(),
        weight_quantization="bf16", tp_values=[1, 2, 4, 8], dp_values=[1], pp_values=[1])
    capacity = search_plans(load_qwen3_235b(), policy="capacity", **options)
    assert capacity.minimum_num_gpus == minimum
    assert all(x.capacity.status == "pass" and x.compatibility.status == "unknown" for x in capacity.candidates)
    static = search_plans(load_qwen3_235b(), **options)
    assert static.candidates == []
    assert static.rejection_reasons
    assert static.capacity_counts["pass"] > 0
    assert static.compatibility_counts == {"unknown": static.evaluated}


def test_inspected_cuda_results_and_search_order_are_retained():
    model, r = load_glm51(), RuntimeProfile.sglang()
    options = dict(runtime=r, weight_quantization="fp8", tp_values=[4, 8],
                   dp_values=[1], pp_values=[1, 2, 4], max_gpus=16)
    legacy, modern = search(model, **options), search_plans(model, **options)
    assert legacy.minimum_num_gpus == modern.minimum_num_gpus
    assert len(legacy.candidates) == len(modern.candidates)
    for a, b in zip(legacy.candidates, modern.candidates):
        assert a.parallelism == b.parallelism
        assert a.ranks == b.capacity.ranks
        assert a.summary == b.capacity.summary
        assert b.static_candidate


def test_search_resolves_backend_once_and_explains_an_empty_degree_space():
    with patch("inferfit.analysis.resolve_backend", wraps=resolve_backend) as resolve:
        result = search_plans(dense(), hardware=custom_hardware(96), policy="capacity",
            weight_quantization="bf16", tp_values=[1, 2], dp_values=[1], pp_values=[1, 2])
        assert result.candidates
        assert resolve.call_count == 1
    result = search_plans(dense(), gpu_counts=[3], tp_values=[1, 2], dp_values=[1], pp_values=[1])
    assert result.evaluated == 0
    assert result.candidates == []


def test_custom_device_and_vendor_build_cannot_bypass_strict_checks():
    assert check(dense(), hardware=custom_hardware(96), parallelism=P).status == "unsupported"
    api_compatible = replace(custom_hardware(96), platform="cuda", device_api="cuda", compute_capability=90)
    assert check(dense(), hardware=api_compatible, parallelism=P).status == "unsupported"
    r = replace(RuntimeProfile(), distribution="uninspected-fork")
    assert check(dense(), runtime=r, parallelism=P).status == "unsupported"


def test_capacity_does_not_require_cuda_compute_capability():
    result = analyze(dense(), hardware=Hardware(compute_capability=None),
                     parallelism=P, weight_quantization="bf16")
    assert result.capacity.status == "pass"
    assert result.compatibility.status == "unknown"


def test_cli_scenario_and_exit_status_keep_capacity_separate_from_deployment(tmp_path):
    doc = {"model": "qwen3-235b-a22b", "hardware": {"catalog": "h20-96gb"},
           "runtime": {"engine": "sglang", "distribution": "custom-fork"},
           "parallelism": {"tp": 8, "dp": 1, "ep": 8, "pp": 1, "num_gpus": 8},
           "weight_quantization": "bf16"}
    result = evaluate_document(doc, mode="analyze")
    assert result["capacity"]["status"] == "pass"
    assert not result["static_candidate"]
    assert result["compatibility"]["backend_id"] is None
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(doc))
    run = subprocess.run([sys.executable, "-m", "inferfit", "analyze", str(path)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["compatibility"]["status"] == "unknown"
    doc.pop("parallelism")
    doc["search"] = {"tp_values": [4, 8], "dp_values": [1], "pp_values": [1], "policy": "capacity"}
    assert evaluate_document(doc, mode="search-plans")["minimum_num_gpus"] == 8
