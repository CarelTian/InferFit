"""Pinned SGLang rules; independent of the vLLM implementation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from ..tensor_rules import tensor_role
from ..types import Hardware, ModelProfile, Parallelism, RuntimeProfile, TensorSpec, Workload
from ..engine_versions import SGLANG_VERSION
from .base import EngineAdapter

@dataclass(frozen=True)
class SglangAdapter(EngineAdapter):
    engine: str = "sglang"
    version: str = SGLANG_VERSION

    def supports_native_layout(self, model):
        return (model.architecture, model.provenance.get("native_quantization")) in {
            ("gdn_qsa_moe", "fp8_experts_and_ngram_with_bf16_exclusions"),
            ("dsv4", "fp8_128x128_ue8m0_dense_mxfp4_experts"),
        }

    def native_tp(self, p: Parallelism, r: RuntimeProfile) -> int:
        # if enable DP Attention
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
        advanced = (model.architecture in {"gdn_moe", "gdn_qsa_moe", "dsv4"}
                    or "shared" in model.config.get("indexer_types", []))
        if not advanced:
            return errors
        if r.sglang_dp_attention or p.dp != 1:
            errors.append("Hybrid/shared/compressed layouts currently require DP=1; new DP layer-boundary mappings are not validated")
        if kv != "bf16" and model.architecture != "dsv4":
            errors.append("Hybrid/shared cache contract currently covers BF16 KV only")
        if model.architecture == "gdn_qsa_moe":
            if p.pp != 1:
                errors.append("Flash-Next currently requires PP=1; vision and PLE pipeline placement is not validated")
            if w.block_size <= 1 or w.block_size % model.config["indexer_compress_ratio"]:
                errors.append("QSA page size must be >1 and divisible by indexer_compress_ratio")
            if h.compute_capability != 90:
                errors.append("This Flash-Next contract currently covers Hopper SM90 only")
        if model.architecture in {"gdn_moe", "gdn_qsa_moe"}:
            from ..states.gdn import recurrent_state_bytes
            try:
                recurrent_state_bytes(model, tp=p.tp)
            except ValueError as error:
                errors.append(str(error))
            if fmt != "bf16" and not model.provenance.get("native_quantization"):
                errors.append("Hybrid GDN weight contract currently covers BF16 only; FP8 checkpoint exclusions need validation")
            if model.provenance.get("native_quantization") and p.ep != p.tp:
                errors.append("Native Flash-Next FP8 requires EP=TP: 640-wide experts cannot be TP-split into aligned 128-element blocks")
        elif model.architecture == "dsv4":
            if h.compute_capability != 90:
                errors.append("Native V4 FlashInfer CUTLASS MXFP4 / FlashMLA contract covers Hopper SM90 only")
            if p.ep != p.tp:
                errors.append("Native V4 MXFP4 contract currently requires EP=TP (whole expert ownership)")
            if model.config["o_groups"] % p.tp:
                errors.append("V4 attention output groups must be divisible by TP")
            if kv != "fp8_e4m3" or w.block_size != 256:
                errors.append("V4 CUDA paged layout requires fp8_e4m3 KV and block_size=256")
        elif "shared" in model.config.get("indexer_types", []) and p.pp != 1:
            errors.append("Shared-index GLM currently requires PP=1; cross-stage index ownership is not validated")
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
        if tensor_role(t) == "index_norm" or t.name.endswith(".linear_attn.A_log"):
            return "F32"
        if t.dtype == "F8_E8M0" and tensor_role(t) != "routed_expert":
            return "F32"  # upper bound, including ordinary FP8 linear scale casts
        if t.dtype == "BF16" and t.name.endswith(".weight_scale_inv"):
            return "F32"
        return t.dtype

    def projection_copies(self, fmt: str) -> bool:
        # SGLang materializes contiguous w_kc/w_vc even for BF16 weights.
        return True

    def metadata_bytes(self, w: Workload) -> int:
        # ReqToTokenPool includes a sentinel request row in 0.5.23.
        return (w.concurrency_per_dp + 1) * (w.context_tokens + 4) * 4

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
            "tp_size": self.native_tp(p, r),
            "dp_size": 1 if r.sglang_dp_attention else p.dp,
            "attn_dp_size": p.dp if r.sglang_dp_attention else 1,
            "pp_size": p.pp, "ep_size": p.ep,
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
            "disable_prefill_cuda_graph": True, "disable_decode_cuda_graph": True,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True, "disable_shared_experts_fusion": True,
            "moe_a2a_backend": "none", "moe_runner_backend": "triton",
            "nnodes": nodes,
        }
        if model.architecture == "mla_dsa":
            args.update(dsa_prefill_backend="flashmla_auto" if kv == "fp8_e4m3" else "flashmla_sparse",
                        dsa_decode_backend="flashmla_kv" if kv == "fp8_e4m3" else "fa3",
                        enable_dsa_fused_indexer=False)
        if fmt in {"fp8_block", "native"}:
            args["quantization"] = "fp8"
        env = {"SGLANG_PP_LAYER_PARTITION": ",".join(map(str, counts)),
               "SGLANG_USE_DEEPGEMM_BMM": "false"}
        if model.architecture in {"gdn_moe", "gdn_qsa_moe"}:
            args["max_mamba_cache_size"] = w.concurrency_per_dp
            env.update(SGLANG_MAMBA_SSM_DTYPE="float32", SGLANG_MAMBA_CONV_DTYPE="bfloat16")
        if model.architecture == "gdn_qsa_moe":
            args["qsa_indexer_dtype"] = "bfloat16"
            args["disable_radix_cache"] = True
        if model.architecture == "dsv4":
            args.update(attention_backend="dsv4", dsv4_attn_backend="flashmla", moe_runner_backend="flashinfer_mxfp4",
                        flashinfer_mxfp4_moe_precision="bf16", swa_full_tokens_ratio=1.0,
                        swa_prefix_tails=0, enable_encoder_swa_bounded_replay=False)
            env.update(SGLANG_DSV4_KV_LAYOUT="v4", SGLANG_OPT_USE_ONLINE_COMPRESS="false",
                       SGLANG_DSV4_COMPRESS_STATE_DTYPE="float32", SGLANG_DSV4_FP4_DEQUANT="false",
                       SGLANG_OPT_FP8_WO_A_GEMM="true", SGLANG_OPT_FUSE_WQA_WKV="false")
        return args, env

    def resident_buffers(self, model, plan, workload, runtime):
        if model.architecture != "dsv4":
            return {}
        c = model.config
        # lru_cache(2) shares frequency tables across equal RoPE parameter sets.
        kinds = len({bool(ratio) for ratio in c["compress_ratios"]})
        return {"v4_rope_frequency_tables": kinds * c["max_position_embeddings"] * (c["qk_rope_head_dim"] // 2) * 8,
                "v4_auxiliary_buffer_envelope": c["num_hidden_layers"] * 65536}

    def load_workspace_bytes(self, model, plan, runtime):
        if model.architecture != "dsv4":
            return super().load_workspace_bytes(model, plan, runtime)
        from ..weights import storage_bytes
        per_layer = {}
        for t in model.tensors:
            if tensor_role(t) == "routed_expert":
                per_layer[t.layer] = per_layer.get(t.layer, 0) + storage_bytes(t.shape, t.dtype) * t.multiplicity // plan.ep
        # SM90 creates interleaved copies and releases the source per layer.
        return max(runtime.load_workspace_bytes, max(per_layer.values()))
