"""Exact minimax PP partition for the estimator's additive per-layer memory model."""
from __future__ import annotations

import math
from dataclasses import replace

from .tensor_rules import first_stage, tensor_role
from .adapters import get_adapter
from .estimator import (_tensor_local, check, ceildiv, kv_per_token, storage_bytes)
from .types import CheckResult, Hardware, ModelProfile, Parallelism, RuntimeProfile, Workload


def optimize_pipeline(model: ModelProfile, *, parallelism: Parallelism,
                      hardware: Hardware | None = None, workload: Workload | None = None,
                      runtime: RuntimeProfile | None = None,
                      weight_quantization: str = "fp8_block",
                      kv_quantization: str = "bf16") -> CheckResult:
    """Find the best contiguous, nonempty layer partition for fixed degrees.

    The objective is max per-GPU peak bytes, considering load and serving phases.
    This is exact for this estimator's current linear layer-state model, not for
    measured kernel behavior. Complexity O(PP * layers**2), no GPU required.
    """
    p = parallelism
    h, w, r = hardware or Hardware(), workload or Workload(), runtime or RuntimeProfile()
    kwargs = dict(hardware=h, workload=w, runtime=r,
                  weight_quantization=weight_quantization, kv_quantization=kv_quantization)
    initial = check(model, parallelism=p, **kwargs)
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
    fixed_runtime = exemplar.runtime_peak_bytes - count * mla_per_layer
    fixed_load = exemplar.load_peak_bytes - exemplar.weight_bytes - count * mla_per_layer
    k_per_layer = kv_per_token(model, tp=p.tp, layers=1,
                              kv_quantization=initial.kv_quantization, runtime=r)["total"]
    blocks = w.concurrency_per_dp * ceildiv(w.context_tokens, w.block_size) + 1
    kv_cost = [blocks * math.ceil(k_per_layer * count * w.block_size * r.kv_pool_overhead_ratio)
               for count in range(n + 1)]

    def stage_peak(stage: int, start: int, end: int) -> int:
        count = end - start
        weight = prefix[end] - prefix[start]
        if stage == 0:
            weight += first
        if stage == p.pp - 1:
            weight += last
        return weight + count * mla_per_layer + max(fixed_load, fixed_runtime + kv_cost[count])

    previous = [math.inf] * (n + 1)
    previous[0] = 0
    choices: dict[tuple[int, int], int] = {}
    for stage in range(p.pp):
        current = [math.inf] * (n + 1)
        # Leave at least one layer for every remaining stage.
        for end in range(stage + 1, n - (p.pp - stage - 1) + 1):
            for start in range(stage, end):
                if previous[start] == math.inf:
                    continue
                worst = max(previous[start], stage_peak(stage, start, end))
                if worst < current[end]:
                    current[end] = worst
                    choices[stage + 1, end] = start
        previous = current
    end = n
    partition = []
    for stage in range(p.pp, 0, -1):
        start = choices[stage, end]
        partition.append(end - start)
        end = start
    partition.reverse()
    result = check(model, parallelism=replace(p, stage_layers=tuple(partition)), **kwargs)
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
