"""Hardware-independent capacity fallback plus independent execution checks."""
from collections import Counter
from dataclasses import replace
from itertools import product

from .analysis_types import (AnalysisResult, CapacityAssessment, CompatibilityAssessment,
                             MemoryAssumptions, PlanSearchResult)
from .backends import assess_uninspected, resolve_backend
from .engines import get_adapter
from .estimator import _check, check
from .partition import optimize_pipeline
from .search import _degrees
from .types import Hardware, Parallelism, RuntimeProfile, Workload


def _memory_runtime(runtime, assumptions):
    if assumptions.layout == "generic":
        if assumptions.reference_engine is not None or assumptions.reference_version is not None:
            raise ValueError("Reference engine/version require layout='runtime_reference'")
        engine, version = "generic", "1"
    elif assumptions.layout == "runtime_reference":
        engine, version = assumptions.reference_engine, assumptions.reference_version
        if not engine or not version:
            raise ValueError("runtime_reference requires reference_engine and reference_version")
    else:
        raise ValueError("Memory layout must be generic or runtime_reference")
    result = replace(runtime, engine=engine, version=version, name=f"capacity-reference-{engine}-{version}",
                     distribution="upstream", image=None, sdk_version=None, build_revision=None)
    get_adapter(result)  # Reject unknown reference versions; never follow latest.
    return result


def analyze(model, *, hardware=None, parallelism=None, workload=None, runtime=None,
            weight_quantization="fp8_block", kv_quantization="bf16",
            assumptions=None, optimize_pp=False):
    """Analyze capacity even without a device backend. Never proves a deployment.

    Default fallback assumes ordinary paged GQA and conventional TP/EP placement.
    Other families require MemoryAssumptions(layout='runtime_reference', ...).
    Numeric runtime reserves are always the caller's explicit/default assumptions.
    """
    h, p = hardware or Hardware(), parallelism or Parallelism()
    w, r = workload or Workload(), runtime or RuntimeProfile()
    a = assumptions or MemoryAssumptions()
    if not isinstance(a, MemoryAssumptions):
        raise TypeError("assumptions must be MemoryAssumptions")
    if type(optimize_pp) is not bool:
        raise ValueError("optimize_pp must be boolean")
    # Validate option spelling even if an inspected backend makes fallback unnecessary.
    memory_runtime = _memory_runtime(r, a)
    return _analyze(model, h, p, w, r, weight_quantization, kv_quantization, a,
                    memory_runtime, resolve_backend(h, r), optimize_pp)


def _analyze(model, h, p, w, r, weight, kv, assumptions, memory_runtime, backend, optimize_pp):
    strict = None
    compatibility = assess_uninspected(backend)
    if backend.kind == "inspected":
        evaluate = optimize_pipeline if optimize_pp else check
        strict = evaluate(model, hardware=h, parallelism=p, workload=w, runtime=r,
                          weight_quantization=weight, kv_quantization=kv)
        compatibility = CompatibilityAssessment(
            "supported" if strict.status in {"pass", "fail"} else "unsupported" if strict.status == "unsupported" else "unknown",
            backend.backend_id, strict.reasons if strict.status not in {"pass", "fail"} else [],
            {"source": "inspected_engine_contract", "deployment_verified": False})
    if strict is not None and strict.status in {"pass", "fail", "invalid"}:
        raw, basis = strict, "inspected_backend"
        notes = ["Runtime reserves and activation envelopes are assumed, not measured"]
    else:
        kwargs = dict(hardware=h, parallelism=p, workload=w, runtime=memory_runtime,
                      weight_quantization=weight, kv_quantization=kv)
        raw = (optimize_pipeline(model, **kwargs, _capacity_only=True) if optimize_pp
               else _check(model, **kwargs, capacity_only=True))
        basis = assumptions.layout
        notes = [
            "RuntimeProfile byte reserves/activation envelope are assumptions; cuda_context_bytes is used as the device-context reserve",
            "Weight and KV storage/placement are conditional on the selected memory layout, not device support",
            "Node indices describe arithmetic packing only; networking, purchase units and launch placement are not validated",
        ]
        if basis == "runtime_reference":
            notes.append(f"Explicitly assume {memory_runtime.engine} {memory_runtime.version} memory layout on the requested device; physical identity is unchanged")
        else:
            notes.extend([
                "Generic GQA uses one padded spare KV page; FP8 cache payload assumes no additional scale pool beyond the configured overhead ratio",
                "Generic expert placement shards whole experts by EP then tensor dimensions by TP/EP; attention TP remains TP",
                "FP8 block weight assumption is 128x128 with FP32 scales and unquantized exclusions from TensorSpec",
            ])
    status = raw.status if raw.status in {"pass", "fail", "invalid"} else "unknown"
    capacity = CapacityAssessment(status, basis, list(raw.reasons),
                                  list(raw.reasons) if status == "unknown" else [], notes,
                                  raw.ranks, dict(raw.summary))
    capacity.warnings = list(raw.warnings) if basis == "inspected_backend" else [
        "Capacity estimate uses explicit/default memory assumptions, not a measured target-device allocation",
        "Capacity pass does not imply support for requested quantization, KV kernels, parallel placement or launch",
    ]
    if basis != "inspected_backend":
        # A capacity-only result must not leak executable parameters or static-candidate labels.
        capacity.summary.update(engine_args={}, engine_env={}, deployment_verified=False,
                                placement_verified=False, memory_layout=basis)
        capacity.summary.pop("configured_kv_pool_tokens_per_dp", None)
        capacity.summary.pop("pool_size_policy", None)
    if status == "invalid":
        compatibility = CompatibilityAssessment("unknown", backend.backend_id,
                                               ["Invalid input; compatibility was not evaluated"])
    return AnalysisResult(model.model_id, raw.parallelism, raw.weight_quantization,
                          raw.kv_quantization, capacity, compatibility)


def search_plans(model, *, hardware=None, workload=None, runtime=None,
                 weight_quantization="fp8_block", kv_quantization="bf16", assumptions=None,
                 policy="static", tp_values=(1, 2, 4, 8), dp_values=(1, 2, 4),
                 pp_values=tuple(range(1, 17)), ep_values=None, gpu_counts=None,
                 max_gpus=64, limit=20, optimize_pp=True):
    """Search static-supported or capacity-only candidates with rejection diagnostics."""
    if policy not in {"static", "capacity"}:
        raise ValueError("policy must be static or capacity")
    if type(optimize_pp) is not bool:
        raise ValueError("optimize_pp must be boolean")
    if type(max_gpus) is not int or max_gpus <= 0 or type(limit) is not int or limit <= 0:
        raise ValueError("max_gpus and limit must be positive integers")
    tps, dps, pps = (_degrees(v, n) for v, n in
                     [(tp_values, "tp_values"), (dp_values, "dp_values"), (pp_values, "pp_values")])
    eps = _degrees(ep_values, "ep_values") if ep_values is not None else None
    gpu_counts = _degrees(gpu_counts, "gpu_counts") if gpu_counts is not None else None
    h, w, r = hardware or Hardware(), workload or Workload(), runtime or RuntimeProfile()
    a = assumptions or MemoryAssumptions()
    if not isinstance(a, MemoryAssumptions):
        raise TypeError("assumptions must be MemoryAssumptions")
    memory_runtime = _memory_runtime(r, a)
    backend = resolve_backend(h, r)
    # Resolve layout/identity once per search, outside candidate and PP loops.
    adapter = get_adapter(r if backend.kind == "inspected" else memory_runtime)
    candidates, cc, bc, rejected = [], Counter(), Counter(), Counter()
    for tp, dp, pp in product(tps, dps, pps):
        gpus = tp * dp * pp
        if gpus > max_gpus or (gpu_counts is not None and gpus not in gpu_counts):
            continue
        p = Parallelism(tp, dp, 1, pp, gpus)
        choices = eps or (adapter.expert_choices(p, r if backend.kind == "inspected" else memory_runtime)
                          if model.architecture != "gqa" else [1])
        for ep in choices:
            result = _analyze(model, h, replace(p, ep=ep), w, r, weight_quantization,
                              kv_quantization, a, memory_runtime, backend, optimize_pp)
            cc[result.capacity.status] += 1
            bc[result.compatibility.status] += 1
            if result.capacity.status == "pass" and (policy == "capacity" or result.static_candidate):
                candidates.append(result)
            else:
                reasons = list(result.capacity.reasons)
                if policy == "static" and result.compatibility.status != "supported":
                    reasons.extend(result.compatibility.reasons)
                rejected.update(set(reasons))
    candidates.sort(key=lambda x: (x.parallelism.num_gpus, x.parallelism.dp,
                                   -x.capacity.summary["minimum_headroom_bytes"],
                                   x.parallelism.pp, x.parallelism.tp, x.parallelism.ep))
    if not cc:
        rejected["No degree combination matches max_gpus and gpu_counts"] = 1
    space = dict(tp_values=list(tps), dp_values=list(dps), pp_values=list(pps),
                 ep_values=list(eps) if eps else "layout_mesh_degrees", max_gpus=max_gpus,
                 gpu_counts=list(gpu_counts) if gpu_counts else None, optimize_pp=optimize_pp,
                 passing_candidates_total=len(candidates), returned_limit=limit)
    return PlanSearchResult(candidates[:limit], sum(cc.values()), dict(cc), dict(bc),
                            candidates[0].parallelism.num_gpus if candidates else None,
                            policy, dict(rejected), space)
