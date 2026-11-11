"""Maintain one public API and a current, executable-argument contract."""
import importlib.util
import json
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
import tomllib

import pytest

import inferfit
from inferfit import Parallelism, RuntimeProfile, Workload, check, load_glm51, load_model


def test_fixed_version_has_one_build_source():
    config = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    assert "version" not in config["project"]
    assert "version" in config["project"]["dynamic"]
    assert config["tool"]["setuptools"]["dynamic"]["version"]["attr"] == "inferfit._version.__version__"
    assert inferfit.__version__ == version("inferfit") == "0.1.0"


def test_backend_catalog_lists_maintained_upstream_contracts():
    entries = inferfit.backend_catalog()["backends"]
    assert {entry["engine"] for entry in entries} == {"sglang", "vllm"}
    for entry in entries:
        assert entry["vendor"] == "nvidia"
        assert entry["platform"] == "cuda"
        assert entry["distribution"] == "upstream"


def test_public_exports_are_unique_and_bound():
    assert len(inferfit.__all__) == len(set(inferfit.__all__))
    assert all(hasattr(inferfit, name) for name in inferfit.__all__)
    assert not hasattr(inferfit, "__all_new__")
    for module in ("inferfit.adapters", "inferfit.qwen_flash_next"):
        assert importlib.util.find_spec(module) is None


@pytest.mark.parametrize("version", ["0.5.10", "0.5.17", "latest"])
def test_removed_or_vendor_versions_are_not_upstream_aliases(version):
    result = check(load_glm51(), runtime=RuntimeProfile.sglang(version=version))
    assert result.status == "unsupported"
    assert result.memory_pass is None


@pytest.mark.parametrize("name,tp,pp,page,kv,fmt", [
    (None, 8, 2, 64, "bf16", "fp8"),
    (None, 8, 2, 64, "fp8", "fp8"),
    ("qwen3.8-2.4t-a95b", 8, 2, 64, "bf16", "bf16"),
    ("glm-5.3", 8, 1, 64, "bf16", "fp8"),
    ("qwen3.8-flash-next-fp8", 4, 1, 64, "bf16", "native"),
    ("deepseek-v4-flash", 4, 1, 256, "fp8", "native"),
])
def test_all_families_emit_current_cli_fields(name, tp, pp, page, kv, fmt):
    contract = json.loads(files("inferfit").joinpath("data/sglang-contract.json").read_text())
    model = load_model(name) if name else load_glm51()
    result = check(model, runtime=RuntimeProfile.sglang(),
                   parallelism=Parallelism(tp, 1, tp, pp, tp * pp),
                   workload=Workload(block_size=page), weight_quantization=fmt, kv_quantization=kv)
    assert result.status in {"pass", "fail"}, result.reasons
    args = result.summary["engine_args"]
    assert set(args) <= set(contract["cli_argument_fields"])
    assert not {"enable_dp_attention", "disable_cuda_graph", "disable_piecewise_cuda_graph",
                "nsa_prefill_backend", "nsa_decode_backend"} & set(args)
    assert "SGL_USE_DEEPGEMM_BMM" not in result.summary["engine_env"]
