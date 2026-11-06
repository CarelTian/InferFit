"""Finite search over explicitly allowed deployment degrees and GPU budgets."""
from __future__ import annotations

from collections import Counter
from itertools import product
from typing import Iterable

from .adapters import get_adapter
from .estimator import check
from .partition import optimize_pipeline
from .types import Hardware, ModelProfile, Parallelism, RuntimeProfile, SearchResult, Workload


def _degrees(values: Iterable[int], name: str) -> tuple[int, ...]:
    values = tuple(values)
    if not values or any(type(v) is not int or v <= 0 for v in values):
        raise ValueError(f"{name} must be a nonempty sequence of positive integers")
    return tuple(sorted(set(values)))


def search(model: ModelProfile, *, hardware: Hardware | None = None,
           workload: Workload | None = None, runtime: RuntimeProfile | None = None,
           weight_quantization: str = "fp8_block", kv_quantization: str = "bf16",
           tp_values: Iterable[int] = (1, 2, 4, 8),
           dp_values: Iterable[int] = (1, 2, 4),
           pp_values: Iterable[int] = tuple(range(1, 17)),
           ep_values: Iterable[int] | None = None,
           gpu_counts: Iterable[int] | None = None,
           max_gpus: int = 64, limit: int = 20, optimize_pp: bool = True) -> SearchResult:
    """Find memory-feasible static candidates; no throughput optimality claim.

    EP defaults to disabled and full TP*DP mesh. DP is an attention-replica count;
    workload.concurrency_per_dp remains per replica for every candidate.
    """
    if type(optimize_pp) is not bool:
        raise ValueError("optimize_pp must be boolean")
    if type(max_gpus) is not int or max_gpus <= 0 or type(limit) is not int or limit <= 0:
        raise ValueError("max_gpus and limit must be positive integers")
    tp_values = _degrees(tp_values, "tp_values")
    dp_values = _degrees(dp_values, "dp_values")
    pp_values = _degrees(pp_values, "pp_values")
    eps = _degrees(ep_values, "ep_values") if ep_values is not None else None
    counts = _degrees(gpu_counts, "gpu_counts") if gpu_counts is not None else None
    statuses: Counter[str] = Counter()
    candidates = []
    for tp, dp, pp in product(tp_values, dp_values, pp_values):
        gpus = tp * dp * pp
        if gpus > max_gpus or (counts is not None and gpus not in counts):
            continue
        plan = Parallelism(tp, dp, 1, pp, gpus)
        try:
            defaults = get_adapter(runtime or RuntimeProfile()).expert_choices(plan, runtime or RuntimeProfile())
        except ValueError:
            defaults = [1]
        choices = eps if eps is not None else (defaults if model.architecture in {"mla_dsa", "gqa_moe", "gdn_moe"} else [1])
        for ep in choices:
            evaluate = optimize_pipeline if optimize_pp else check
            result = evaluate(model, hardware=hardware, workload=workload, runtime=runtime,
                           parallelism=Parallelism(tp, dp, ep, pp, gpus),
                           weight_quantization=weight_quantization,
                           kv_quantization=kv_quantization)
            statuses[result.status] += 1
            if result.memory_pass:
                candidates.append(result)
    # For equal card counts, prefer fewer DP replicas, then greater headroom.
    candidates.sort(key=lambda x: (x.parallelism.num_gpus, x.parallelism.dp,
                                  -x.summary["minimum_headroom_bytes"],
                                  x.parallelism.pp, x.parallelism.tp, x.parallelism.ep))
    minimum = candidates[0].parallelism.num_gpus if candidates else None
    space = {"tp_values": list(tp_values), "dp_values": list(dp_values),
             "pp_values": list(pp_values), "ep_values": list(eps) if eps else "engine_legal_mesh_degrees",
             "gpu_counts": list(counts) if counts else None, "max_gpus": max_gpus,
             "optimize_pp": optimize_pp, "returned_limit": limit, "passing_candidates_total": len(candidates)}
    return SearchResult(candidates[:limit], sum(statuses.values()), dict(statuses), minimum, space)
