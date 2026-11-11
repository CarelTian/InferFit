"""Architecture-derived tensor inventory."""
from __future__ import annotations
import copy
from typing import Any
from ..types import ModelProfile, TensorSpec
from .common import _positive

def gqa_profile(config: dict[str, Any], *, model_id: str = "custom-gqa",
                revision: str = "user-config") -> ModelProfile:
    """Bias-free dense Llama/Qwen2/Qwen3 adapter; rejects unsupported variants."""
    c = copy.deepcopy(config)
    if c.get("model_type") not in {"llama", "qwen2", "qwen3"}:
        raise ValueError("Supported dense model types: llama, qwen2, qwen3")
    if c.get("num_local_experts") or c.get("num_experts") or c.get("kv_lora_rank"):
        raise ValueError("Use an explicit adapter for MoE or MLA")
    if any(kind != "full_attention" for kind in c.get("layer_types", [])):
        raise ValueError("Hybrid layer_types require a dedicated adapter")
    if c.get("mlp_bias", False):
        raise ValueError("MLP biases are not supported by this dense adapter")
    if c.get("sliding_window") or c.get("use_sliding_window"):
        raise ValueError("Sliding-window attention is not supported by the dense adapter")
    h, layers, vocab, intermediate, heads = (_positive(c, key) for key in (
        "hidden_size", "num_hidden_layers", "vocab_size", "intermediate_size",
        "num_attention_heads"))
    kv_heads = c.get("num_key_value_heads", heads)
    if type(kv_heads) is not int or kv_heads <= 0 or heads % kv_heads:
        raise ValueError("Invalid num_key_value_heads")
    if "head_dim" not in c and h % heads:
        raise ValueError("head_dim is required when hidden_size is not divisible by heads")
    dim = c.get("head_dim", h // heads)
    if type(dim) is not int or dim <= 0:
        raise ValueError("Invalid head_dim")
    c["head_dim"] = dim
    c["num_key_value_heads"] = kv_heads
    c["max_position_embeddings"] = _positive(c, "max_position_embeddings")
    out = [TensorSpec("model.embed_tokens.weight", (vocab, h), "vocab"),
           TensorSpec("model.norm.weight", (h,), "replicated")]
    if not c.get("tie_word_embeddings", False):
        out.append(TensorSpec("lm_head.weight", (vocab, h), "vocab"))
    # Qwen2 defaults to QKV biases; Qwen3/Llama follow attention_bias.
    bias = c.get("attention_bias", c["model_type"] == "qwen2")
    for layer in range(layers):
        p = f"model.layers.{layer}"
        for name in ("input_layernorm", "post_attention_layernorm"):
            out.append(TensorSpec(f"{p}.{name}.weight", (h,), "replicated", layer))
        for name, shape, placement in (
            ("self_attn.q_proj", (heads * dim, h), "column"),
            ("self_attn.k_proj", (kv_heads * dim, h), "kv_column"),
            ("self_attn.v_proj", (kv_heads * dim, h), "kv_column"),
            ("self_attn.o_proj", (h, heads * dim), "row"),
            ("mlp.gate_proj", (intermediate, h), "column"),
            ("mlp.up_proj", (intermediate, h), "column"),
            ("mlp.down_proj", (h, intermediate), "row"),
        ):
            out.append(TensorSpec(f"{p}.{name}.weight", shape, placement, layer,
                                  quantizable=True))
            if bias and name.startswith("self_attn.") and name != "self_attn.o_proj":
                out.append(TensorSpec(f"{p}.{name}.bias", (shape[0],), placement, layer))
        if bias and c["model_type"] != "qwen2":
            out.append(TensorSpec(f"{p}.self_attn.o_proj.bias", (h,), "replicated", layer))
        if c["model_type"] == "qwen3":
            for name in ("q_norm", "k_norm"):
                out.append(TensorSpec(f"{p}.self_attn.{name}.weight", (dim,), "replicated", layer))
    return ModelProfile(model_id, revision, c, tuple(out), "gqa", {
        "tensor_inventory": "architecture_derived", "mtp_included": False,
    })
