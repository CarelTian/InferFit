"""Shared per-rank memory model with pinned, engine-specific contracts."""
from __future__ import annotations

import math
from dataclasses import fields
from typing import Any

from .tensor_rules import first_stage
from .engines import get_adapter
from .weights import storage_bytes, validate_weight_format
from .types import (CheckResult, Hardware, ModelProfile, Parallelism, RankMemory,
                    RuntimeProfile, TensorSpec, Workload)

WEIGHT_ALIASES = {"fp8": "fp8_block", "bfloat16": "bf16", "float16": "fp16"}


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b

def logical_weight_bytes(model: ModelProfile, weight_quantization: str = "bf16") -> int:
    """Unpartitioned inventory size, including scales; excludes runtime transforms."""
    fmt = WEIGHT_ALIASES.get(weight_quantization, weight_quantization)
    validate_weight_format(model, fmt)
    if fmt not in {"bf16", "fp16", "fp8_block", "native"}:
        raise ValueError(f"Unsupported weight format: {fmt}")
    return sum(storage_bytes(t.shape, t.dtype, fmt == "fp8_block" and t.quantizable)
               * t.multiplicity for t in model.tensors)


def _stage_counts(model: ModelProfile, plan: Parallelism, runtime: RuntimeProfile | None = None) -> tuple[int, ...]:
    count = model.config["num_hidden_layers"]
    if plan.stage_layers is not None:
        result = tuple(plan.stage_layers)
        if len(result) != plan.pp or any(type(x) is not int or x <= 0 for x in result):
            raise ValueError("stage_layers must contain pp positive integers")
        if sum(result) != count:
            raise ValueError("stage_layers must sum to num_hidden_layers")
        return result
    if runtime is not None and runtime.engine == "sglang":
        return tuple(count // plan.pp + (i >= plan.pp - count % plan.pp) for i in range(plan.pp))
    return tuple(count // plan.pp + (i < count % plan.pp) for i in range(plan.pp))


def _tensor_local(t: TensorSpec, model: ModelProfile, p: Parallelism,
                  fmt: str, runtime: RuntimeProfile | None = None) -> tuple[tuple[int, ...], int]:
    shape = list(t.shape)
    r = runtime or RuntimeProfile()
    kind, width, copies = get_adapter(r).tensor_plan(t, p, r)
    if kind == "vocab":
        alignment = math.lcm(64, width)
        shape[0] = ceildiv(shape[0], alignment) * alignment // width
    elif kind == "kv_column":
        kv_heads = model.config["num_key_value_heads"]
        shape[0] //= min(width, kv_heads)
    elif kind in {"column", "row"}:
        axis = 0 if kind == "column" else 1
        if shape[axis] % width:
            raise ValueError(f"{t.name}: sharded dimension is not divisible by tensor width {width}")
        shape[axis] //= width
        if ((fmt == "fp8_block" and t.quantizable) or t.dtype == "F8_E4M3") and shape[axis] % 128:
            raise ValueError(f"{t.name}: FP8 shard must align to 128-element weight blocks")
    elif kind != "replicated":
        raise ValueError(f"Unknown tensor placement: {kind}")
    return tuple(shape), copies


def kv_per_token(model: ModelProfile, *, tp: int = 1, layers: int | None = None,
                 kv_quantization: str = "bf16", runtime: RuntimeProfile | None = None,
                 layer_start: int = 0) -> dict[str, int]:
    """Per-rank token geometry; paged V4 rates are rounded-up amortized bytes.

    Exact demand (including pages and fixed states) is reported by check().
    """
    from .states import get_state_model
    return get_state_model(model).per_token(model, tp=tp, layers=layers,
        kv_quantization=kv_quantization, runtime=runtime, layer_start=layer_start)


def _validate(model: ModelProfile, h: Hardware, p: Parallelism, w: Workload,
              r: RuntimeProfile, fmt: str, kv: str) -> tuple[list[str], list[str]]:
    invalid: list[str] = []
    unsupported: list[str] = []
    for name in ("tp", "dp", "ep", "pp", "num_gpus"):
        v = getattr(p, name)
        if type(v) is not int or v <= 0:
            invalid.append(f"{name} must be a positive integer")
    for name in ("context_tokens", "concurrency_per_dp", "max_batch_tokens", "block_size"):
        v = getattr(w, name)
        if type(v) is not int or v <= 0:
            invalid.append(f"{name} must be a positive integer")
    if h.available_nodes is not None and (type(h.available_nodes) is not int or h.available_nodes <= 0):
        invalid.append("available_nodes must be a positive integer or null")
    for name in ("inter_node_network", "allow_cross_node_tp"):
        if type(getattr(h, name)) is not bool:
            invalid.append(f"hardware.{name} must be boolean")
    if type(r.sglang_dp_attention) is not bool:
        invalid.append("runtime.sglang_dp_attention must be boolean")
    if type(r.enforce_eager) is not bool:
        invalid.append("runtime.enforce_eager must be boolean")
    for name in ("memory_bytes", "gpus_per_node"):
        v = getattr(h, name)
        if type(v) is not int or v <= 0:
            invalid.append(f"hardware.{name} must be a positive integer")
    for name in ("external_used_bytes_per_gpu", "safety_margin_bytes"):
        v = getattr(h, name)
        if type(v) is not int or v < 0:
            invalid.append(f"hardware.{name} must be a non-negative integer")
    if type(h.utilization) not in (float, int) or not math.isfinite(h.utilization) or not 0 < h.utilization <= 1:
        invalid.append("utilization must be finite and in (0, 1]")
    for f in fields(r):
        if f.name.endswith("_bytes") or f.name in {"activation_copies", "max_batch_tokens_supported", "max_concurrency_per_dp_supported"}:
            value = getattr(r, f.name)
            if type(value) is not int or value < 0:
                invalid.append(f"runtime.{f.name} must be a non-negative integer")
    if type(r.kv_pool_overhead_ratio) not in (int, float) or not math.isfinite(r.kv_pool_overhead_ratio) or r.kv_pool_overhead_ratio < 1:
        invalid.append("kv_pool_overhead_ratio must be finite and >=1")
    if h.platform != "cuda":
        return invalid, ["No inspected PPU/non-CUDA runtime adapter; supply actual memory for inventory only"]
    if type(h.compute_capability) is not int or h.compute_capability <= 0:
        invalid.append("CUDA compute_capability must be a positive integer")
    if invalid:
        return invalid, unsupported
    if p.num_gpus != p.tp * p.dp * p.pp:
        invalid.append("num_gpus must equal TP * DP * PP; EP reuses TP*DP ranks")
    if model.architecture in {"gdn_moe", "gdn_qsa_moe", "dsv4"} or "shared" in model.config.get("indexer_types", []):
        if (r.engine, r.version) != ("sglang", "0.5.23"):
            unsupported.append("Hybrid GDN / shared-index / V4 models require the inspected SGLang 0.5.23 adapter")
    if p.pp > model.config["num_hidden_layers"]:
        invalid.append("PP cannot exceed the number of model layers")
    if model.config["num_attention_heads"] % p.tp:
        invalid.append("num_attention_heads must be divisible by TP")
    if w.context_tokens > model.config["max_position_embeddings"]:
        invalid.append("context_tokens exceeds model maximum context")
    if w.max_batch_tokens < w.concurrency_per_dp:
        invalid.append("max_batch_tokens must admit at least one decode token per active sequence")
    if h.external_used_bytes_per_gpu >= h.memory_bytes:
        invalid.append("External use must be smaller than physical memory")
    if kv in {"fp8", "fp8_e4m3", "fp8_ds_mla"} and h.compute_capability < 89:
        unsupported.append("This pinned adapter does not support FP8 KV on pre-FP8 GPUs")
    if model.architecture in {"gqa_moe", "gdn_moe", "gdn_qsa_moe"} and model.config["num_experts"] % p.ep:
        invalid.append("num_experts must be divisible by EP")
    if model.architecture == "mla_dsa":
        if (model.config["kv_lora_rank"],model.config["qk_rope_head_dim"],model.config["index_head_dim"]) != (512,64,128):
            unsupported.append("Pinned DSA backend requires latent/RoPE/index dimensions 512/64/128")
        if fmt == "fp16":
            unsupported.append("Pinned GLM FlashMLA sparse profile uses BF16 compute; use bf16 or fp8_block weights")
        if model.config["n_routed_experts"] % p.ep:
            invalid.append("n_routed_experts must be divisible by EP")
        if w.block_size != 64:
            unsupported.append("FLASHMLA_SPARSE requires block_size=64")
        if h.compute_capability // 10 != 9:
            unsupported.append("This runtime profile covers Hopper SM90 only")
        if model.provenance.get("mtp_included"):
            unsupported.append("MTP runtime/cache simulation is not implemented; use base model only")
    elif model.architecture in {"gqa", "gqa_moe", "gdn_moe", "gdn_qsa_moe"}:
        if model.architecture == "gqa" and p.ep != 1:
            invalid.append("Dense GQA models require EP=1")
    elif model.architecture == "dsv4":
        if model.config["n_routed_experts"] % p.ep:
            invalid.append("n_routed_experts must be divisible by EP")
    else:
        unsupported.append("Unknown model architecture")
    if fmt not in {"bf16", "fp16", "fp8_block", "native"}:
        unsupported.append("Supported weight formats: bf16, fp16, fp8_block, native (pinned native checkpoints only)")
    if fmt == "fp8_block" and h.compute_capability < 89:
        unsupported.append("FP8 runtime requires compatible GPU kernels")
    try:
        validate_weight_format(model, fmt)
        engine = get_adapter(r)
        if model.provenance.get("native_quantization") and not engine.supports_native_layout(model):
            unsupported.append("Native weight layout has no inspected contract for this engine/version/family")
        unsupported.extend(engine.validate(model, h, p, w, r, fmt, kv))
    except ValueError as error:
        unsupported.append(str(error))
    if r.provenance != "assumed":
        unsupported.append("Measured profiles require a calibration adapter; only assumed profiles are supported")
    if not r.enforce_eager:
        unsupported.append("Graph capture is not modeled; use enforce_eager=true")
    if r.graph_bytes:
        invalid.append("An eager profile must have graph_bytes=0")
    if w.max_batch_tokens > r.max_batch_tokens_supported or w.concurrency_per_dp > r.max_concurrency_per_dp_supported:
        unsupported.append("Workload exceeds the runtime profile envelope")
    if p.num_gpus > h.gpus_per_node and not h.inter_node_network:
        unsupported.append("Multi-node execution requires an inter-node network")
    if not invalid:
        try:
            _stage_counts(model, p, r)
        except (ValueError, TypeError) as e:
            invalid.append(str(e))
    if not invalid and not unsupported:
        try:
            kv_per_token(model, tp=p.tp, kv_quantization=kv, runtime=r)
        except (ValueError, TypeError) as e:
            unsupported.append(str(e))
    return invalid, unsupported


def check(model: ModelProfile, *, hardware: Hardware | None = None,
          parallelism: Parallelism | None = None, workload: Workload | None = None,
          runtime: RuntimeProfile | None = None, weight_quantization: str = "fp8_block",
          kv_quantization: str = "bf16") -> CheckResult:
    """A pass is conditional on the explicit memory model, never a GPU test."""
    h, p = hardware or Hardware(), parallelism or Parallelism()
    w, r = workload or Workload(), runtime or RuntimeProfile()
    fmt = WEIGHT_ALIASES.get(weight_quantization, weight_quantization)
    try:
        adapter = get_adapter(r)
        kv = adapter.normalize_kv(kv_quantization, model)
    except ValueError:
        kv = kv_quantization
    result = CheckResult("invalid", None, "not_evaluated", model.model_id, p, fmt, kv,
                         "analytical_weights_and_cache_assumed_runtime")
    invalid, unsupported = _validate(model, h, p, w, r, fmt, kv)
    if invalid or unsupported:
        result.status = "invalid" if invalid else "unsupported"
        result.deployability = "rejected"
        result.reasons = invalid + unsupported
        return result
    adapter = get_adapter(r)
    nodes, workers_per_node = adapter.topology(h, p, r)
    counts = _stage_counts(model, p, r)
    bounds = [0]
    for count in counts:
        bounds.append(bounds[-1] + count)
    local_weights = [0] * p.pp
    try:
        for t in model.tensors:
            if t.layer is None:
                stage = 0 if first_stage(t) else p.pp - 1
            else:
                stage = next(i for i in range(p.pp) if bounds[i] <= t.layer < bounds[i + 1])
            shape, copies = _tensor_local(t, model, p, fmt, r)
            size = storage_bytes(shape, adapter.tensor_dtype(t), fmt == "fp8_block" and t.quantizable)
            local_weights[stage] += size * copies
        # A tied output head needs its own copy on the last PP stage.
        if p.pp > 1 and model.config.get("tie_word_embeddings", False):
            t = next(t for t in model.tensors if first_stage(t))
            shape, copies = _tensor_local(t, model, p, fmt, r)
            local_weights[-1] += storage_bytes(shape, adapter.tensor_dtype(t), False) * copies
    except ValueError as e:
        result.status, result.deployability = "unsupported", "rejected"
        result.reasons = [str(e)]
        return result
    budget = min(int(h.memory_bytes * h.utilization),
                 h.memory_bytes - h.external_used_bytes_per_gpu) - h.safety_margin_bytes
    tokens = w.concurrency_per_dp * ceildiv(w.context_tokens, w.block_size) * w.block_size
    from .states import get_state_model
    state_model = get_state_model(model)
    tokens = state_model.token_cap(tokens)
    load_workspace = adapter.load_workspace_bytes(model, p, r)
    c = model.config
    for pp_rank, count in enumerate(counts):
        allocation = state_model.allocate(model, p, w, r, kv, bounds[pp_rank], count)
        parts, allocated_tokens = allocation.kv_parts, allocation.allocated_tokens
        k, kv_required = parts["total"], allocation.cache_bytes
        mla_copies = 0
        index_workspace = 0
        if model.architecture == "mla_dsa":
            index_workspace = adapter.workspace_bytes(w)
            # FP8 kv_b_proj retains BF16 decode projection views after dequantization.
            if adapter.projection_copies(fmt):
                mla_copies = (count * c["kv_lora_rank"] * (c["num_attention_heads"] // p.tp)
                              * (c["qk_nope_head_dim"] + c["v_head_dim"]) * 2)
        # Batch metadata and block tables have per-rank resident allocations.
        metadata = adapter.metadata_bytes(w)
        activation = (r.activation_fixed_bytes + w.max_batch_tokens * adapter.batch_multiplier(p, r)
                      * state_model.activation_width(model) * 2 * r.activation_copies)
        runtime_parts = {
            "cuda_context": r.cuda_context_bytes,
            "communication": r.communication_bytes,
            "allocator": r.allocator_bytes,
            "cuda_graph": r.graph_bytes,
            "activations_and_workspace_assumed": activation,
            "dsa_prefill_workspace_envelope": index_workspace,
            "mla_bf16_projection_copies": mla_copies,
            "block_table_metadata": metadata,
            "extra": r.extra_bytes,
        }
        runtime_parts.update(allocation.runtime_parts)
        resident_buffers = adapter.resident_buffers(model, p, w, r)
        runtime_parts.update(resident_buffers)
        overhead = sum(runtime_parts.values())
        weight = local_weights[pp_rank]
        load_peak = (weight + r.cuda_context_bytes + r.communication_bytes + r.allocator_bytes
                     + mla_copies + load_workspace + r.extra_bytes + sum(resident_buffers.values()))
        serving_peak = weight + overhead + kv_required
        peak = max(load_peak, serving_peak)
        fixed_runtime = overhead - sum(allocation.runtime_parts.values())
        capacity = state_model.capacity(model, p, w, r, kv, bounds[pp_rank], count, budget - weight - fixed_runtime)
        for dp_rank in range(p.dp):
            for tp_rank in range(p.tp):
                rank = adapter.rank(pp_rank, dp_rank, tp_rank, p, r)
                result.ranks.append(RankMemory(
                    rank, pp_rank, dp_rank, tp_rank, rank // workers_per_node,
                    bounds[pp_rank], bounds[pp_rank + 1], weight, k, parts,
                    allocated_tokens, kv_required, runtime_parts, overhead, load_peak,
                    serving_peak, budget, peak, budget - peak,
                    h.memory_bytes - h.external_used_bytes_per_gpu - peak,
                    capacity, peak <= budget, allocation.cache_parts,
                ))
    result.ranks.sort(key=lambda x: x.rank)
    passed = all(x.memory_pass for x in result.ranks)
    result.memory_pass = passed
    result.status = "pass" if passed else "fail"
    result.deployability = "static_candidate" if passed else "rejected"
    bottleneck = min(result.ranks, key=lambda x: x.headroom_bytes)
    logical = logical_weight_bytes(model, fmt)
    available_weight_only = max(0, budget)
    args, engine_env = adapter.engine_config(model, h, p, w, r, fmt, kv, budget, counts, tokens)
    result.summary = {
        "num_gpus": p.num_gpus,
        "nodes_required": nodes,
        "stage_layers": list(counts),
        "rank_order": adapter.rank_order(r),
        "workers_per_node": workers_per_node,
        "engine": r.engine, "engine_version": r.version,
        "native_tp_size": adapter.native_tp(p, r),
        "logical_weight_bytes": logical,
        "placed_weight_bytes_total": sum(x.weight_bytes for x in result.ranks),
        "weight_capacity_lower_bound_gpus": ceildiv(logical, h.memory_bytes),
        "budget_weight_only_lower_bound_gpus": ceildiv(logical, available_weight_only) if available_weight_only else None,
        "bottleneck_rank": bottleneck.rank,
        "minimum_headroom_bytes": bottleneck.headroom_bytes,
        "kv_bytes_per_token_logical": kv_per_token(model, kv_quantization=kv, runtime=r)["total"],
        "kv_bytes_per_token_per_rank_min": min(x.kv_bytes_per_token for x in result.ranks),
        "kv_bytes_per_token_per_rank_max": max(x.kv_bytes_per_token for x in result.ranks),
        # One logical token visits one DP replica; MLA/KV replicas across TP count here.
        "kv_bytes_per_token_per_dp_replica_physical": sum(x.kv_bytes_per_token for x in result.ranks if x.dp_rank == 0),
        "kv_capacity_tokens_per_dp": min(x.kv_capacity_tokens for x in result.ranks),
        "target_resident_tokens_per_dp": w.context_tokens * w.concurrency_per_dp,
        "block_rounded_target_tokens_per_dp": tokens,
        "deployment_verified": False,
        "engine_args": args,
        "engine_env": engine_env,
        "requested_utilization": h.utilization,
        "safety_margin_bytes": h.safety_margin_bytes,
        "runtime_profile": r.name,
        "runtime_provenance": r.provenance,
        "load_workspace_bytes_per_rank": load_workspace,
        "weight_checkpoint": ("zai-org/GLM-5.1-FP8" if fmt == "fp8_block" else model.model_id)
            if model.model_id == "zai-org/GLM-5.1" else model.model_id,
        "weight_checkpoint_revision": model.provenance.get("fp8_revision", model.revision)
            if fmt == "fp8_block" else model.revision,
    }
    if not passed:
        result.reasons.append(f"GPU rank {bottleneck.rank} exceeds budget by {-bottleneck.headroom_bytes} bytes")
    result.warnings = [
        "Runtime reserves/activation envelope are assumed, not measured; pass is conditional.",
        "No GPU boot test, driver/library validation, loading I/O or throughput/SLO proof was performed.",
        "No offload, prefix sharing, speculative/MTP decoding, redundant experts or CUDA Graphs are modeled.",
        "Reported slack is budget headroom, not device free memory after KV pool preallocation.",
        "DP workload is concurrency_per_dp per attention replica, not total concurrency divided silently.",
    ]
    if r.engine == "sglang":
        result.summary["pool_size_policy"] = "explicit_target_token_cap"
        result.summary["sglang_dp_attention"] = r.sglang_dp_attention
        result.summary["configured_kv_pool_tokens_per_dp"] = tokens
        result.warnings.append("SGLang mem_fraction_static is not a full-peak utilization control; max_total_tokens explicitly caps the modeled pool.")
    if model.provenance.get("native_quantization"):
        result.summary["native_weight_layout"] = model.provenance["native_quantization"]
        result.summary["weight_format_semantics"] = "Native checkpoint mixed dtypes and explicit scale tensors; no blanket requantization"
    if model.architecture == "dsv4":
        result.summary["kv_rate_semantics"] = "Rounded-up amortized paged bytes/token, excludes reserved pages and compressor states; use exact rank allocations for demand"
        result.summary["cache_allocation_policy"] = "CUDA V4 paged layout; SWA/full=1; FP32 offline C4/C128 states"
        result.warnings.append("V4 dense scale storage uses an upper bound; allocator/workspace envelopes are assumed. Packed expert weight bytes are native. Physical SWA pool uses explicit ratio=1, not a 128-token ring.")
    if model.architecture == "gdn_qsa_moe":
        result.summary["workload_scope"] = "text_only"
        result.summary["vision_weight_placement"] = "replicated_upper_bound"
        result.warnings.append("Vision encoder weights are conservatively replicated in the budget; image/video activation memory is not modeled.")
    if nodes > 1:
        result.warnings.append("Multi-node candidate requires an actual connected cluster and distributed bootstrap.")
    return result
