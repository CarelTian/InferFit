"""Engine contracts inspected at pinned upstream versions; no GPU imports."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .tensor_rules import tensor_role
from .types import Hardware, ModelProfile, Parallelism, RuntimeProfile, TensorSpec, Workload

SGLANG_VERSION = "0.5.10"
SGLANG_COMMIT = "1519acf37c23f2189adb93f57ca9cd2db1bebf18"


@dataclass(frozen=True)
class VllmAdapter:
    engine: str = "vllm"
    version: str = "0.19.0"

    def native_tp(self, p: Parallelism, r: RuntimeProfile) -> int:
        return p.tp

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

    def rank(self, pp: int, dp: int, tp: int, p: Parallelism, r: RuntimeProfile) -> int:
        return (dp * p.pp + pp) * p.tp + tp

    def rank_order(self, r: RuntimeProfile) -> str:
        return "dp,pp,tp (tp fastest)"

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

    def tensor_dtype(self, t: TensorSpec) -> str:
        return t.dtype

    def projection_copies(self, fmt: str) -> bool:
        return fmt == "fp8_block"

    def metadata_bytes(self, w: Workload) -> int:
        return w.concurrency_per_dp * ((w.context_tokens + w.block_size - 1) // w.block_size) * 4

    def workspace_bytes(self, w: Workload) -> int:
        return max(w.context_tokens * 40 * 132, w.context_tokens * 5 * 576 * 2)

    def batch_multiplier(self, p: Parallelism, r: RuntimeProfile) -> int:
        return 1

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


@dataclass(frozen=True)
class SglangAdapter(VllmAdapter):
    engine: str = "sglang"
    version: str = SGLANG_VERSION

    def native_tp(self, p: Parallelism, r: RuntimeProfile) -> int:
        return p.tp * p.dp if r.sglang_dp_attention else p.tp

    def normalize_kv(self, value: str, model: ModelProfile) -> str:
        return {"bfloat16": "bf16", "float16": "fp16", "fp8": "fp8_e4m3"}.get(value, value)

    def expert_choices(self, p: Parallelism, r: RuntimeProfile) -> list[int]:
        width = self.native_tp(p, r)
        return [degree for degree in range(1, width + 1) if width % degree == 0]

    def topology(self, h: Hardware, p: Parallelism, r: RuntimeProfile) -> tuple[int, int]:
        # SGLang's native TP*PP world is evenly split across nnodes.
        minimum = (p.num_gpus + h.gpus_per_node - 1) // h.gpus_per_node
        world = self.native_tp(p, r) * p.pp
        for nodes in range(minimum, min(world,h.available_nodes or world) + 1):
            # Native launcher assigns whole PP stages per node, or equally splits
            # one stage across an integer number of nodes. World divisibility
            # alone does not guarantee that every rank is launched.
            if p.pp >= nodes:
                if p.pp % nodes: continue
            else:
                if nodes % p.pp or self.native_tp(p,r) % (nodes//p.pp): continue
            if world % nodes:
                continue
            local = p.num_gpus // nodes
            if local > h.gpus_per_node:
                continue
            if not h.allow_cross_node_tp and local % p.tp:
                continue
            return nodes, local
        raise ValueError("No native SGLang TP/PP placement fits available_nodes and TP policy")

    def validate(self, model: ModelProfile, h: Hardware, p: Parallelism,
                 w: Workload, r: RuntimeProfile, fmt: str, kv: str) -> list[str]:
        errors = []
        native = self.native_tp(p, r)
        if w.max_batch_tokens % w.block_size:
            errors.append("SGLang chunked prefill size must be divisible by page_size")
        if native % p.ep:
            errors.append("SGLang EP must divide native TP (TP*DP with DP attention, otherwise TP)")
        if r.sglang_dp_attention and model.architecture != "mla_dsa":
            errors.append("This SGLang DP-attention adapter is validated only for GLM MLA/DSA")
        if r.sglang_dp_attention and w.max_batch_tokens * p.dp > r.max_batch_tokens_supported:
            errors.append("Aggregate DP-attention batch exceeds the SGLang runtime envelope")
        if p.dp > 1 and not r.sglang_dp_attention and p.num_gpus > h.gpus_per_node:
            errors.append("SGLang multi-node ordinary DP is unsupported; use DP attention or separate launches")
        if model.architecture == "mla_dsa" and kv not in {"bf16", "fp8_e4m3"}:
            errors.append("SGLang DSA supports bf16 or fp8_e4m3 KV (fp8 alias)")
        if model.architecture in {"gqa", "gqa_moe"} and kv not in {"bf16", "fp16", "fp8_e4m3"}:
            errors.append("Unsupported SGLang GQA cache dtype")
        if native > model.config.get("n_routed_experts", model.config.get("num_experts", native)):
            errors.append("SGLang native TP cannot exceed the number of routed experts")
        try:
            nodes, _ = self.topology(h, p, r)
            if nodes > 1 and not h.inter_node_network:
                errors.append("SGLang equal-worker placement requires inter-node networking")
        except ValueError as error:
            errors.append(str(error))
        return errors

    def rank(self, pp: int, dp: int, tp: int, p: Parallelism, r: RuntimeProfile) -> int:
        if r.sglang_dp_attention:
            return (pp * p.dp + dp) * p.tp + tp
        return super().rank(pp, dp, tp, p, r)

    def rank_order(self, r: RuntimeProfile) -> str:
        return "pp,dp,tp (tp fastest)" if r.sglang_dp_attention else super().rank_order(r)

    def tensor_plan(self, t: TensorSpec, p: Parallelism, r: RuntimeProfile) -> tuple[str, int, int]:
        kind, copies = t.placement, t.multiplicity
        native = self.native_tp(p, r)
        width = p.tp
        if kind.startswith("expert_"):
            copies //= p.ep
            width = native // p.ep
            kind = kind.removeprefix("expert_")
        elif tensor_role(t) in {"dense_ffn", "shared_expert", "output"}:
            width = native
        # Embedding uses attention TP. LM head is explicitly DP-sharded below.
        return kind, width, copies

    def tensor_dtype(self, t: TensorSpec) -> str:
        # SGLang Indexer creates LayerNorm parameters in FP32.
        return "F32" if tensor_role(t) == "index_norm" or t.name.endswith(".linear_attn.A_log") else t.dtype

    def projection_copies(self, fmt: str) -> bool:
        # SGLang materializes contiguous w_kc/w_vc even for BF16 weights.
        return True

    def metadata_bytes(self, w: Workload) -> int:
        # ReqToTokenPool is int32 [max_requests, context_len + 4].
        return w.concurrency_per_dp * (w.context_tokens + 4) * 4

    def workspace_bytes(self, w: Workload) -> int:
        # Explicit analytical envelope for FP32 prefill index logits, not profiling.
        return w.max_batch_tokens * w.context_tokens * 4

    def batch_multiplier(self, p: Parallelism, r: RuntimeProfile) -> int:
        return p.dp if r.sglang_dp_attention else 1

    def engine_config(self, model: ModelProfile, h: Hardware, p: Parallelism,
                      w: Workload, r: RuntimeProfile, fmt: str, kv: str,
                      budget: int, counts: tuple[int, ...], tokens: int) -> tuple[dict[str, Any], dict[str, str]]:
        nodes, _ = self.topology(h, p, r)
        multiplier = self.batch_multiplier(p, r)
        args: dict[str, Any] = {
            "tp_size": self.native_tp(p, r), "dp_size": p.dp,
            "pp_size": p.pp, "ep_size": p.ep,
            "enable_dp_attention": r.sglang_dp_attention,
            # Make vocab-head sharding match the analytical attention mesh.
            "enable_dp_lm_head": False,
            "dtype": "float16" if fmt == "fp16" else "bfloat16",
            "kv_cache_dtype": "bfloat16" if kv == "bf16" else "float16" if kv == "fp16" else kv,
            "page_size": w.block_size,
            "context_length": w.context_tokens,
            "max_running_requests": w.concurrency_per_dp * multiplier,
            "chunked_prefill_size": w.max_batch_tokens * multiplier,
            "max_prefill_tokens": w.max_batch_tokens * multiplier,
            # Native auto-profiling differs from vLLM; cap the actual pool demand.
            "max_total_tokens": tokens,
            "mem_fraction_static": max(0, budget) / (h.memory_bytes - h.external_used_bytes_per_gpu),
            "disable_cuda_graph": True, "disable_piecewise_cuda_graph": True,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True, "disable_shared_experts_fusion": True,
            "moe_a2a_backend": "none", "moe_runner_backend": "triton",
            "nnodes": nodes,
        }
        if model.architecture == "mla_dsa":
            args.update(nsa_prefill_backend="flashmla_auto" if kv == "fp8_e4m3" else "flashmla_sparse",
                        nsa_decode_backend="flashmla_kv" if kv == "fp8_e4m3" else "fa3")
        if fmt == "fp8_block":
            args["quantization"] = "fp8"
        return args, {"SGLANG_PP_LAYER_PARTITION": ",".join(map(str, counts)),
                      "SGL_USE_DEEPGEMM_BMM": "false"}


def get_adapter(runtime: RuntimeProfile) -> VllmAdapter:
    if runtime.engine == "vllm" and runtime.version == "0.19.0":
        return VllmAdapter()
    if runtime.engine == "sglang" and runtime.version == SGLANG_VERSION:
        return SglangAdapter()
    if runtime.engine == "sglang" and runtime.version == "0.5.23":
        return SglangFlagshipAdapter()
    raise ValueError("Supported pinned adapters: vllm 0.19.0, sglang " + SGLANG_VERSION + ", sglang 0.5.23 (flagship scope)")


@dataclass(frozen=True)
class SglangFlagshipAdapter(SglangAdapter):
    """Conservative 0.5.23 eager contract; no implicit DP or state conversion."""
    version: str = "0.5.23"

    def validate(self, model, h, p, w, r, fmt, kv):
        errors = super().validate(model, h, p, w, r, fmt, kv)
        if model.architecture not in {"gdn_moe", "gdn_qsa_moe"} and "shared" not in model.config.get("indexer_types", []):
            errors.append("SGLang 0.5.23 adapter is scoped to hybrid GDN and shared-index GLM")
        if r.sglang_dp_attention or p.dp != 1:
            errors.append("Flagship adapter currently requires DP=1; new DP layer-boundary mappings are not validated")
        if kv != "bf16":
            errors.append("Flagship cache contract currently covers BF16 KV only")
        if model.architecture == "gdn_qsa_moe":
            if p.pp != 1:
                errors.append("Flash-Next currently requires PP=1; vision and PLE pipeline placement is not validated")
            if w.block_size <= 1 or w.block_size % model.config["indexer_compress_ratio"]:
                errors.append("QSA page size must be >1 and divisible by indexer_compress_ratio")
            if h.compute_capability != 90:
                errors.append("This Flash-Next contract currently covers Hopper SM90 only")
        if model.architecture in {"gdn_moe", "gdn_qsa_moe"}:
            from .flagships import recurrent_state_bytes
            try:
                recurrent_state_bytes(model, tp=p.tp)
            except ValueError as error:
                errors.append(str(error))
            if fmt != "bf16":
                errors.append("Hybrid GDN weight contract currently covers BF16 only; FP8 checkpoint exclusions need validation")
        elif p.pp != 1:
            errors.append("Shared-index GLM currently requires PP=1; cross-stage index ownership is not validated")
        return errors

    def engine_config(self, model, h, p, w, r, fmt, kv, budget, counts, tokens):
        args, env = super().engine_config(model, h, p, w, r, fmt, kv, budget, counts, tokens)
        if model.architecture in {"gdn_moe", "gdn_qsa_moe"}:
            args["max_mamba_cache_size"] = w.concurrency_per_dp
            env.update(SGLANG_MAMBA_SSM_DTYPE="float32", SGLANG_MAMBA_CONV_DTYPE="bfloat16")
        if model.architecture == "gdn_qsa_moe":
            args["qsa_indexer_dtype"] = "bfloat16"
            args["disable_radix_cache"] = True
        return args, env
