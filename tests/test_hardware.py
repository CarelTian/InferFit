"""Published per-GPU capacities, architecture boundaries and deployment planning."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from inferfit import (MemoryAssumptions, Parallelism, RuntimeProfile, analyze,
                      check, hardware_catalog, load_hardware, load_model,
                      load_qwen3_235b, search, search_plans)
from inferfit.cli import evaluate_document


ROOT = Path(__file__).resolve().parents[1]
NEW_GPUS = [
    ("h200-141gb", 141, 90),
    ("b200-180gb", 180, 100),
    ("gb200-186gb", 186, 100),
    ("b300-270gb", 270, 103),
    ("b300-288gb", 288, 103),
    ("gb300-279gb", 279, 103),
    ("gb300-288gb", 288, 103),
    ("rtx5090-32gb", 32, 120),
]


@pytest.mark.parametrize("name,gb,cc", NEW_GPUS)
def test_published_capacity_identity_and_source(name, gb, cc):
    catalog = hardware_catalog()
    h = load_hardware(name)
    assert h.memory_bytes == gb * 10**9
    assert h.compute_capability == cc
    assert h.platform == "cuda" and h.vendor == "nvidia"
    assert h.spec_source == catalog["sources"][name]
    assert h.memory_provenance == "published_nominal"
    assert catalog["variant_notes"][name]
    actual = load_hardware(name, memory_bytes=gb * 10**9 - 12345,
                           gpus_per_node=1, available_nodes=1)
    assert actual.memory_bytes == h.memory_bytes - 12345
    assert actual.memory_provenance == "caller_override"
    assert actual.compute_capability == cc
    assert actual.gpus_per_node == actual.available_nodes == 1


def test_a100_40gb_preset_removed_but_80gb_retained():
    with pytest.raises(ValueError, match="Unknown hardware variant"):
        load_hardware("a100-40gb")
    assert load_hardware("a100-80gb").memory_bytes == 80 * 10**9


@pytest.fixture(scope="module")
def qwen():
    return load_qwen3_235b()


@pytest.mark.parametrize("runtime", [RuntimeProfile.sglang(), RuntimeProfile()])
@pytest.mark.parametrize("name,gb,cc", NEW_GPUS)
def test_real_qwen_capacity_and_fp8_kv_across_architectures(qwen, runtime, name, gb, cc):
    plan = Parallelism(tp=4, ep=4, num_gpus=4)
    results = [check(qwen, hardware=load_hardware(name), runtime=runtime,
                     parallelism=plan, weight_quantization="bf16", kv_quantization=kv)
               for kv in ("bf16", "fp8")]
    bf16, fp8 = results
    assert bf16.status == ("fail" if cc == 120 else "pass")
    assert fp8.status == bf16.status
    assert bf16.ranks[0].kv_bytes_per_token == 48128
    assert fp8.ranks[0].kv_bytes_per_token == 24064
    assert fp8.ranks[0].weight_bytes == bf16.ranks[0].weight_bytes
    assert fp8.ranks[0].kv_required_bytes * 2 == bf16.ranks[0].kv_required_bytes
    # The overall peak may be the loading peak, which KV precision cannot reduce.
    assert fp8.ranks[0].serving_peak_bytes < bf16.ranks[0].serving_peak_bytes
    assert fp8.ranks[0].headroom_bytes >= bf16.ranks[0].headroom_bytes
    assert not bf16.summary["deployment_verified"]


def test_b300_capacity_choice_changes_two_gpu_feasibility(qwen):
    results = [check(qwen, hardware=load_hardware(name), runtime=RuntimeProfile.sglang(),
                     parallelism=Parallelism(tp=2, ep=2, num_gpus=2), weight_quantization="bf16")
               for name in ("b300-270gb", "b300-288gb")]
    lower, higher = results
    assert lower.status == "fail" and higher.status == "pass"
    assert lower.ranks[0].peak_bytes == higher.ranks[0].peak_bytes
    assert higher.ranks[0].headroom_bytes - lower.ranks[0].headroom_bytes == 16_200_000_000


@pytest.mark.parametrize("name,gb,cc", NEW_GPUS)
def test_fp8_weights_reduce_residency_without_changing_kv(qwen, name, gb, cc):
    kwargs = dict(hardware=load_hardware(name), runtime=RuntimeProfile.sglang(),
                  parallelism=Parallelism(tp=4, ep=4, num_gpus=4))
    bf16 = check(qwen, weight_quantization="bf16", **kwargs)
    fp8 = check(qwen, weight_quantization="fp8", **kwargs)
    assert fp8.status in {"pass", "fail"}
    assert bf16.ranks[0].weight_bytes // 2 < fp8.ranks[0].weight_bytes < bf16.ranks[0].weight_bytes
    assert fp8.ranks[0].kv_required_bytes == bf16.ranks[0].kv_required_bytes
    assert fp8.ranks[0].headroom_bytes > bf16.ranks[0].headroom_bytes


@pytest.mark.parametrize("name,expected", [("h200-141gb", 4), ("b200-180gb", 4),
                                         ("gb200-186gb", 4), ("b300-270gb", 4),
                                         ("b300-288gb", 2), ("gb300-279gb", 2),
                                         ("gb300-288gb", 2)])
def test_qwen_sglang_search_uses_actual_capacity(qwen, name, expected):
    result = search(qwen, hardware=load_hardware(name), runtime=RuntimeProfile.sglang(),
                    weight_quantization="bf16", tp_values=[1, 2, 4], dp_values=[1],
                    pp_values=[1], max_gpus=4)
    assert result.minimum_num_gpus == expected
    assert all(c.memory_pass for c in result.candidates)


@pytest.fixture(scope="module")
def flash_next():
    return load_model("qwen3.8-flash-next-fp8")


def test_h200_reuses_hopper_native_contract(flash_next):
    result = check(flash_next, hardware=load_hardware("h200-141gb"),
                   runtime=RuntimeProfile.sglang(), weight_quantization="native",
                   parallelism=Parallelism(tp=2, ep=2, num_gpus=2))
    assert result.status == "pass"


@pytest.mark.parametrize("name", ["b200-180gb", "gb200-186gb", "b300-288gb",
                                 "gb300-279gb", "rtx5090-32gb"])
def test_new_architecture_does_not_bypass_hopper_contract(flash_next, name):
    result = analyze(flash_next, hardware=load_hardware(name),
                     runtime=RuntimeProfile.sglang(), weight_quantization="native",
                     parallelism=Parallelism(tp=2, ep=2, num_gpus=2),
                     assumptions=MemoryAssumptions("runtime_reference", "sglang", "0.5.23"))
    assert result.compatibility.status == "unsupported"
    assert any("SM90" in reason for reason in result.compatibility.reasons)
    assert result.capacity.status == ("fail" if name == "rtx5090-32gb" else "pass")
    assert not result.static_candidate
    assert result.capacity.summary["engine_args"] == {}


def test_capacity_search_keeps_blackwell_compatibility_restriction(flash_next):
    kwargs = dict(hardware=load_hardware("b200-180gb"), runtime=RuntimeProfile.sglang(),
                  weight_quantization="native", tp_values=[2], ep_values=[2],
                  pp_values=[1], dp_values=[1], max_gpus=2,
                  assumptions=MemoryAssumptions("runtime_reference", "sglang", "0.5.23"))
    assert not search_plans(flash_next, **kwargs).candidates
    result = search_plans(flash_next, policy="capacity", **kwargs)
    assert result.minimum_num_gpus == 2
    assert result.candidates[0].compatibility.status == "unsupported"


@pytest.mark.parametrize("path", sorted((ROOT / "examples/hardware").glob("*.json")),
                         ids=lambda p: p.stem)
def test_hardware_examples_are_runnable(path):
    mode = "pd-check" if path.stem.endswith("-pd") else "search" if path.stem.endswith("-search") else "check"
    result = evaluate_document(json.loads(path.read_text()), mode=mode, base=path.parent)
    if mode == "search":
        assert result["minimum_num_gpus"] == 4
    else:
        assert result["status"] == "pass"


def test_cli_hardware_includes_sources_and_new_devices():
    done = subprocess.run([sys.executable, "-m", "inferfit", "hardware"],
                          capture_output=True, text=True, check=True)
    catalog = json.loads(done.stdout)
    assert catalog["variants"]["rtx5090-32gb"]["compute_capability"] == 120
    assert catalog["variants"]["gb300-279gb"]["memory_bytes"] == 279_000_000_000
    assert "a100-40gb" not in catalog["variants"]
    assert catalog["variant_notes"]["b300-288gb"]
