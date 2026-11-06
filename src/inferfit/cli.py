"""CLI: JSON in/out with no runtime network access."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

from . import check, from_config, load_glm51, load_qwen3_235b, search, check_pd, PDPool, PDDemand, UnsupportedModelError
from .profiles import load_model_profile, load_hardware_profile, load_runtime_profile
from .types import Hardware, Parallelism, RuntimeProfile, Workload


def _object(cls: type, values: Any, base: Path = Path(".")):
    if isinstance(values,dict) and "profile_path" in values:
        if set(values)!={"profile_path"}:raise ValueError("profile_path cannot be mixed with inline fields")
        loader={Hardware:load_hardware_profile,RuntimeProfile:load_runtime_profile}.get(cls)
        if loader is None:raise ValueError("No profile loader for this object")
        return loader(base / values["profile_path"])
    if not isinstance(values, dict):
        raise ValueError(f"{cls.__name__} must be a JSON object")
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    if cls is Parallelism and "stage_layers" in values and values["stage_layers"] is not None:
        values = {**values, "stage_layers": tuple(values["stage_layers"])}
    if cls is RuntimeProfile and values.get("engine") == "sglang":
        return RuntimeProfile.sglang(**values)
    return cls(**values)


def evaluate_document(document: dict[str, Any], *, mode: str = "check", base: Path = Path(".")) -> dict[str, Any]:
    allowed = {"model", "hardware", "parallelism", "workload", "runtime",
               "weight_quantization", "kv_quantization", "search", "prefill", "decode", "demand"}
    if not isinstance(document, dict) or set(document) - allowed:
        raise ValueError("Scenario must be an object with only documented top-level fields")
    ref = document.get("model", "glm-5.1")
    if ref == "glm-5.1":
        model = load_glm51()
    elif ref == "qwen3-235b-a22b":
        model = load_qwen3_235b()
    elif isinstance(ref,dict) and "profile_path" in ref:
        if set(ref)!={"profile_path"}:raise ValueError("model profile_path must be exclusive")
        model=load_model_profile(base / ref["profile_path"])
    elif isinstance(ref, dict):
        if set(ref) - {"config", "config_path", "id", "revision"}:
            raise ValueError("Unknown model fields")
        if ("config" in ref) == ("config_path" in ref):
            raise ValueError("model must provide exactly one of config or config_path")
        config = ref.get("config") if "config" in ref else base / ref["config_path"]
        model = from_config(config, model_id=ref.get("id", "custom"), revision=ref.get("revision", "user-config"))
    else:
        raise ValueError("model must be glm-5.1 or a config reference")
    if mode == "pd-check":
        if set(document)-{"model","prefill","decode","demand"}:
            raise ValueError("PD scenarios use only model, prefill, decode and demand")
        def pool(values):
            if not isinstance(values,dict) or set(values)-{"gpu_ids","hardware","parallelism","runtime","weight_quantization","kv_quantization","max_batch_tokens","transfer_buffer_bytes_per_gpu"}:
                raise ValueError("Invalid PD pool fields")
            return PDPool(gpu_ids=tuple(values["gpu_ids"]),
                hardware=_object(Hardware,values.get("hardware",{}),base),
                parallelism=_object(Parallelism,values.get("parallelism",{})),
                runtime=_object(RuntimeProfile,values.get("runtime",{}),base),
                **{k:v for k,v in values.items() if k not in {"gpu_ids","hardware","parallelism","runtime"}})
        return check_pd(model,prefill=pool(document["prefill"]),decode=pool(document["decode"]),
                        demand=_object(PDDemand,document.get("demand",{}))).to_dict()
    if set(document)&{"prefill","decode","demand"}:
        raise ValueError("Use pd-check for PD fields")
    kwargs = {
        "hardware": _object(Hardware, document.get("hardware", {}),base),
        "workload": _object(Workload, document.get("workload", {})),
        "runtime": _object(RuntimeProfile, document.get("runtime", {}),base),
        "weight_quantization": document.get("weight_quantization", "fp8_block"),
        "kv_quantization": document.get("kv_quantization", "bf16"),
    }
    if mode == "check":
        if "search" in document:
            raise ValueError("check does not consume search options")
        result = check(model, parallelism=_object(Parallelism, document.get("parallelism", {})), **kwargs)
    elif mode == "search":
        if "parallelism" in document:
            raise ValueError("search uses search degrees; remove fixed parallelism")
        options = document.get("search", {})
        allowed_options = {"tp_values", "dp_values", "pp_values", "ep_values", "gpu_counts", "max_gpus", "limit", "optimize_pp"}
        if not isinstance(options, dict) or set(options) - allowed_options:
            raise ValueError("Unknown search options")
        result = search(model, **options, **kwargs)
    else:
        raise ValueError("mode must be check or search")
    return result.to_dict()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="inferfit", description="InferFit: per-GPU LLM memory preflight and finite-layout search")
    parser.add_argument("mode", choices=("check", "search", "pd-check"))
    parser.add_argument("scenario", type=Path, help="Scenario JSON (offline)")
    parser.add_argument("--output", type=Path, help="Write JSON result to a file")
    args = parser.parse_args(argv)
    try:
        document = json.loads(args.scenario.read_text())
        result = evaluate_document(document, mode=args.mode, base=args.scenario.resolve().parent)
        encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(encoded)
        else:
            print(encoded, end="")
        if args.mode in {"check", "pd-check"}:
            return 0 if result["memory_pass"] else 2
        return 0 if result["candidates"] else 2
    except UnsupportedModelError as error:
        print(json.dumps({"status":"unsupported","memory_pass":None,"reasons":[str(error)]}))
        return 2
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(json.dumps({"status": "input_error", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
