"""Standard GQA, MLA/DSA and hybrid GDN/QSA token geometry."""
from __future__ import annotations
from ..engines import get_adapter
from ..types import ModelProfile, RuntimeProfile

def kv_per_token(model: ModelProfile, *, tp: int = 1, layers: int | None = None,
                 kv_quantization: str = "bf16",
                 runtime: RuntimeProfile | None = None, layer_start: int = 0) -> dict[str, int]:
    """Physical payload bytes/token on one GPU, before block/pool rounding."""
    if type(tp) is not int or tp <= 0:
        raise ValueError("tp must be positive")
    c = model.config
    layers = c["num_hidden_layers"] - layer_start if layers is None else layers
    if type(layers) is not int or layers <= 0:
        raise ValueError("layers must be positive")
    if type(layer_start) is not int or layer_start < 0 or layer_start + layers > c["num_hidden_layers"]:
        raise ValueError("Invalid layer interval")
    adapter = get_adapter(runtime or RuntimeProfile())
    if model.architecture in {"gdn_moe", "gdn_qsa_moe"} or "shared" in c.get("indexer_types", []):
        if (adapter.engine, adapter.version) != ("sglang", "0.5.23"):
            raise ValueError("Hybrid GDN / shared-index KV requires SGLang 0.5.23")
    dtype = adapter.normalize_kv(kv_quantization, model)
    if model.architecture == "mla_dsa":
        if dtype == "bf16":
            latent, rope, scales = c["kv_lora_rank"] * 2, c["qk_rope_head_dim"] * 2, 0
        elif dtype == ("fp8_e4m3" if adapter.engine == "sglang" else "fp8_ds_mla"):
            if c["kv_lora_rank"] != 512 or c["qk_rope_head_dim"] != 64:
                raise ValueError("fp8_ds_mla backend requires latent=512 and RoPE=64")
            latent, rope, scales = 512, 128, 16
        else:
            raise ValueError(f"{adapter.engine} MLA DSA supports bf16 or its engine-specific FP8 format (fp8 alias)")
        if c["index_head_dim"] != 128:
            raise ValueError("Pinned DSA backend requires index_head_dim=128")
        # Shared index K, not index_n_heads independent cache vectors.
        index_layers = c.get("indexer_types", ["full"] * c["num_hidden_layers"])[layer_start:layer_start + layers].count("full")
        parts = {"latent": latent * layers, "rope": rope * layers,
                 "mla_scales": scales * layers,
                 "index_keys": 128 * index_layers, "index_scales": 4 * index_layers}
    elif model.architecture in {"gqa", "gqa_moe", "gdn_moe", "gdn_qsa_moe"}:
        if dtype not in {"bf16", "fp16", "fp8" if adapter.engine == "vllm" else "fp8_e4m3"}:
            raise ValueError("GQA supports bf16, fp16 or fp8 KV")
        kv_heads = c["num_key_value_heads"]
        if (kv_heads >= tp and kv_heads % tp) or (kv_heads < tp and tp % kv_heads):
            raise ValueError("KV heads cannot be evenly sharded/replicated by this TP")
        heads = max(1, kv_heads // tp)
        cached_layers = c["layer_types"][layer_start:layer_start + layers].count("full_attention") if model.architecture in {"gdn_moe", "gdn_qsa_moe"} else layers
        size = cached_layers * heads * c["head_dim"] * (1 if dtype in {"fp8", "fp8_e4m3"} else 2)
        parts = {"keys": size, "values": size}
        if model.architecture == "gdn_qsa_moe":
            if dtype != "bf16":
                raise ValueError("Flash-Next cache contract covers BF16 only")
            parts["qsa_compressed_keys"] = (cached_layers * c["indexer_kv_heads"] *
                c["indexer_head_dim"] * 2 // c["indexer_compress_ratio"])
    else:
        raise ValueError(f"Unsupported state architecture: {model.architecture}")
    return {**parts, "total": sum(parts.values())}

