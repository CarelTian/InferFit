"""Pinned vLLM rules. No dependency on another serving engine."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from ..tensor_rules import tensor_role
from ..types import Hardware, ModelProfile, Parallelism, RuntimeProfile, TensorSpec, Workload
from .base import EngineAdapter

@dataclass(frozen=True)
class VllmAdapter(EngineAdapter):
    engine: str = "vllm"
    version: str = "0.19.0"

    def normalize_kv(self, value: str, model: ModelProfile) -> str:
        value = {"bfloat16": "bf16", "float16": "fp16"}.get(value, value)
        if model.architecture == "mla_dsa" and value == "fp8":
            return "fp8_ds_mla"
        return value

    def expert_choices(self, p: Parallelism, r: RuntimeProfile) -> list[int]:
        return sorted({1, p.tp * p.dp})

    def topology(self, h: Hardware, p: Parallelism, r: RuntimeProfile) -> tuple[int, int]:
        return (p.num_gpus + h.gpus_per_node - 1) // h.gpus_per_node, h.gpus_per_node

    def validate(self, model: ModelProfile, h: Hardware, p: Parallelism,
                 w: Workload, r: RuntimeProfile, fmt: str, kv: str) -> list[str]:
        errors = []
        if r.sglang_dp_attention:
            errors.append("sglang_dp_attention requires engine=sglang")
        if model.architecture in {"mla_dsa", "gqa_moe"} and p.ep not in {1, p.tp * p.dp}:
            errors.append("vLLM adapter supports EP=1 (off) or EP=TP*DP (enabled)")
        if model.architecture == "mla_dsa" and kv not in {"bf16", "fp8_ds_mla"}:
            errors.append("Hopper DSA backend supports bf16 or fp8_ds_mla KV")
        if h.available_nodes is not None and self.topology(h,p,r)[0] > h.available_nodes:
            errors.append("Placement exceeds available_nodes")
        crosses = p.tp > h.gpus_per_node or (
            p.num_gpus > h.gpus_per_node and h.gpus_per_node % p.tp != 0
        )
        if crosses and not h.allow_cross_node_tp:
            errors.append("Cross-node TP is disabled by the hardware placement policy")
        if model.architecture in {"gqa", "gqa_moe"} and kv not in {"bf16", "fp16", "fp8"}:
            errors.append("Unsupported GQA cache dtype")
        return errors

    def tensor_plan(self, t: TensorSpec, p: Parallelism, r: RuntimeProfile) -> tuple[str, int, int]:
        kind, width, copies = t.placement, p.tp, t.multiplicity
        if tensor_role(t) == "shared_expert" and p.ep > 1 and p.tp > 1 and p.dp > 1:
            kind = "replicated"
        if kind.startswith("expert_"):
            if p.ep > 1:
                copies //= p.ep
                kind = "replicated"
            else:
                kind = kind.removeprefix("expert_")
        return kind, width, copies

    def projection_copies(self, fmt: str) -> bool:
        return fmt == "fp8_block"

    def metadata_bytes(self, w: Workload) -> int:
        return w.concurrency_per_dp * ((w.context_tokens + w.block_size - 1) // w.block_size) * 4

    def workspace_bytes(self, w: Workload) -> int:
        return max(w.context_tokens * 40 * 132, w.context_tokens * 5 * 576 * 2)

    def engine_config(self, model: ModelProfile, h: Hardware, p: Parallelism,
                      w: Workload, r: RuntimeProfile, fmt: str, kv: str,
                      budget: int, counts: tuple[int, ...], tokens: int) -> tuple[dict[str, Any], dict[str, str]]:
        args: dict[str, Any] = {
            "tensor_parallel_size": p.tp, "data_parallel_size": p.dp,
            "pipeline_parallel_size": p.pp, "enable_expert_parallel": p.ep > 1,
            "dtype": "float16" if fmt == "fp16" else "bfloat16",
            "kv_cache_dtype": "bfloat16" if kv == "bf16" else "float16" if kv == "fp16" else kv,
            "all2all_backend": "allgather_reducescatter",
            "max_model_len": w.context_tokens, "max_num_seqs": w.concurrency_per_dp,
            "max_num_batched_tokens": w.max_batch_tokens, "enable_chunked_prefill": True,
            "enforce_eager": True, "block_size": w.block_size,
            "gpu_memory_utilization": max(0, budget) / h.memory_bytes,
        }
        if fmt == "fp8_block":
            args["quantization"] = "fp8"
        return args, {"VLLM_PP_LAYER_PARTITION": ",".join(map(str, counts))}
