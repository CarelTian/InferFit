"""Exact minimax PP partition for the estimator's additive per-layer memory model."""
from __future__ import annotations

import math
from dataclasses import replace

from .tensor_rules import first_stage
from .engines import get_adapter
from .estimator import _tensor_local, check, _check, storage_bytes
from .types import CheckResult, Hardware, ModelProfile, Parallelism, RuntimeProfile, Workload


def optimize_pipeline(model: ModelProfile, *, parallelism: Parallelism,
                      hardware: Hardware | None = None, workload: Workload | None = None,
                      runtime: RuntimeProfile | None = None,
                      weight_quantization: str = "fp8_block",
                      kv_quantization: str = "bf16", _capacity_only: bool = False) -> CheckResult:
    """Find the best contiguous, nonempty layer partition for fixed degrees.

    The objective is max per-GPU peak bytes, considering load and serving phases.
    This is exact for this estimator's memory model, not measured kernel behavior.
    Supported layouts compile O(1) interval costs; DP takes O(PP * layers**2)
    time and O(layers**2 + PP * layers) space. No GPU is required.
    """
    p = parallelism
    h, w, r = hardware or Hardware(), workload or Workload(), runtime or RuntimeProfile()
    kwargs = dict(hardware=h, workload=w, runtime=r,
                  weight_quantization=weight_quantization, kv_quantization=kv_quantization)
    evaluate = check
    if _capacity_only:
        evaluate = _check
        kwargs["capacity_only"] = True
    initial = evaluate(model, parallelism=p, **kwargs)
    if initial.status not in {"pass", "fail"} or p.pp == 1:
        return initial
    adapter = get_adapter(r)
    n = model.config["num_hidden_layers"]
    weights = [0] * n
    first = last = 0
    for tensor in model.tensors:
        shape, copies = _tensor_local(tensor, model, p, initial.weight_quantization, r)
        size = storage_bytes(shape, adapter.tensor_dtype(tensor),
                             initial.weight_quantization == "fp8_block" and tensor.quantizable) * copies
        if tensor.layer is not None:
            weights[tensor.layer] += size
        elif first_stage(tensor):
            first += size
            if model.config.get("tie_word_embeddings", False):
                last += size
        else:
            last += size
    prefix = [0]
    for weight in weights:
        prefix.append(prefix[-1] + weight)
    exemplar = initial.ranks[0]
    count = exemplar.layer_end - exemplar.layer_start
    mla_per_layer = exemplar.runtime_breakdown["mla_bf16_projection_copies"] // count
    from .states import get_state_model
    state_model = get_state_model(model)
    allocation = state_model.allocate(model, p, w, r, initial.kv_quantization,
                                      exemplar.layer_start, count)
    fixed_runtime = exemplar.runtime_peak_bytes - count * mla_per_layer - sum(allocation.runtime_parts.values())
    fixed_load = exemplar.load_peak_bytes - exemplar.weight_bytes - count * mla_per_layer

    state_cost = state_model.prepare_interval_cost(model, p, w, r, initial.kv_quantization)
    # State allocation and rounding do not depend on the PP stage number.
    # Build each feasible interval once; endpoint weights are added per stage.
    interval_peak = [[]]
    max_stage_layers = n - p.pp + 1
    for end in range(1, n + 1):
        costs = [0] * end
        for start in range(max(0, end - max_stage_layers), end):
            costs[start] = (prefix[end] - prefix[start] + (end - start) * mla_per_layer
                            + max(fixed_load, fixed_runtime + state_cost(start, end)))
        interval_peak.append(costs)

    previous = [math.inf] * (n + 1)
    previous[0] = 0
    choices = [[-1] * (n + 1) for _ in range(p.pp)]
    for stage in range(p.pp):
        current = [math.inf] * (n + 1)
        parents = choices[stage]
        endpoint = first if stage == 0 else last if stage == p.pp - 1 else 0
        # Leave at least one layer for every remaining stage.
        for end in range(stage + 1, n - (p.pp - stage - 1) + 1):
            costs = interval_peak[end]
            if stage == 0:
                current[end], parents[end] = costs[0] + endpoint, 0
                continue
            best, parent = math.inf, -1
            # Every predecessor in this range is feasible from the prior stage.
            # Ascending starts and strict improvement preserve the original ties.
            for start in range(stage, end):
                before, local = previous[start], costs[start] + endpoint
                worst = before if before > local else local
                if worst < best:
                    best, parent = worst, start
            current[end], parents[end] = best, parent
        previous = current
    end = n
    partition = []
    for stage in range(p.pp, 0, -1):
        start = choices[stage - 1][end]
        partition.append(end - start)
        end = start
    partition.reverse()
    result = evaluate(model, parallelism=replace(p, stage_layers=tuple(partition)), **kwargs)
    actual_peak = max(rank.peak_bytes for rank in result.ranks)
    if actual_peak != previous[n]:
        raise RuntimeError("Partition cost model drifted from the memory estimator")
    result.summary["partition_optimization"] = {
        "method": "exact_minimax_contiguous_partition",
        "objective": "maximum_per_gpu_peak_bytes",
        "optimal_within": "fixed degrees and contiguous nonempty layer partitions in this memory model",
        "predicted_peak_bytes": actual_peak,
    }
    return result
