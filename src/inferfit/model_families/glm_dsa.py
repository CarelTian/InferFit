"""Architecture-derived tensor inventory."""
from __future__ import annotations
import copy
from typing import Any
from ..errors import UnsupportedModelError
from ..types import ModelProfile, TensorSpec
from .common import _positive
from dataclasses import replace

def glm_profile(config: dict[str, Any], *, model_id: str = "custom-glm-dsa",
                revision: str = "user-config", include_mtp: bool = False) -> ModelProfile:
    """Build the GLM-5/5.1 DSA tensor inventory; MTP is excluded by default."""
    c = copy.deepcopy(config)
    if c.get("model_type") != "glm_moe_dsa":
        raise ValueError("glm_profile requires model_type=glm_moe_dsa")
    if c.get("attention_bias", False) or c.get("tie_word_embeddings", False):
        raise ValueError("This GLM adapter requires bias-free attention and untied embeddings")
    if c.get("moe_layer_freq", 1) != 1:
        raise ValueError("Only moe_layer_freq=1 is supported")
    h, layers, vocab = (_positive(c, k) for k in
                        ("hidden_size", "num_hidden_layers", "vocab_size"))
    heads, q, k, rope, nope, value, ih, index_dim = (_positive(c, key) for key in (
        "num_attention_heads", "q_lora_rank", "kv_lora_rank", "qk_rope_head_dim",
        "qk_nope_head_dim", "v_head_dim", "index_n_heads", "index_head_dim"))
    dense_i, expert_i, experts = (_positive(c, key) for key in (
        "intermediate_size", "moe_intermediate_size", "n_routed_experts"))
    _positive(c, "max_position_embeddings")
    _positive(c, "index_topk")
    activated = _positive(c, "num_experts_per_tok")
    if activated > experts:
        raise ValueError("num_experts_per_tok exceeds n_routed_experts")
    if c.get("qk_head_dim", nope + rope) != nope + rope:
        raise ValueError("qk_head_dim must equal qk_nope_head_dim + qk_rope_head_dim")
    first_dense = c.get("first_k_dense_replace", 0)
    shared = c.get("n_shared_experts", 0)
    if type(first_dense) is not int or not 0 <= first_dense <= layers:
        raise ValueError("Invalid first_k_dense_replace")
    if type(shared) is not int or shared < 0:
        raise ValueError("Invalid n_shared_experts")
    indexers = c.get("indexer_types", ["full"] * layers)
    if len(indexers) != layers or indexers[0] != "full" or any(x not in {"full", "shared"} for x in indexers):
        raise UnsupportedModelError("indexer_types must describe every base layer and start with full")
    out: list[TensorSpec] = []

    def add(name: str, shape: tuple[int, ...], placement: str = "replicated",
            layer: int | None = None, multiplicity: int = 1,
            quantizable: bool = False, dtype: str = "BF16") -> None:
        out.append(TensorSpec(name, shape, placement, layer, multiplicity, quantizable, dtype))

    add("model.embed_tokens.weight", (vocab, h), "vocab")
    add("lm_head.weight", (vocab, h), "vocab")
    add("model.norm.weight", (h,))
    total_layers = layers + (c.get("num_nextn_predict_layers", 0) if include_mtp else 0)
    for layer in range(total_layers):
        p = f"model.layers.{layer}"
        add(p + ".input_layernorm.weight", (h,), layer=layer)
        add(p + ".post_attention_layernorm.weight", (h,), layer=layer)
        attn = p + ".self_attn"
        for name, shape, placement in (
            ("q_a_proj", (q, h), "replicated"),
            ("q_b_proj", (heads * (nope + rope), q), "column"),
            ("kv_a_proj_with_mqa", (k + rope, h), "replicated"),
            ("kv_b_proj", (heads * (nope + value), k), "column"),
            ("o_proj", (h, heads * value), "row"),
            ("indexer.wq_b", (ih * index_dim, q), "replicated"),
            ("indexer.wk", (index_dim, h), "replicated"),
        ):
            add(f"{attn}.{name}.weight", shape, placement, layer, quantizable=True)
        for name, dim in (("q_a_layernorm", q), ("kv_a_layernorm", k),
                          ("indexer.k_norm", index_dim)):
            add(f"{attn}.{name}.weight", (dim,), layer=layer)
        add(attn + ".indexer.k_norm.bias", (index_dim,), layer=layer)
        add(attn + ".indexer.weights_proj.weight", (ih, h), layer=layer)
        if layer < first_dense:
            for name, shape, placement in (
                ("gate_proj", (dense_i, h), "column"),
                ("up_proj", (dense_i, h), "column"),
                ("down_proj", (h, dense_i), "row"),
            ):
                add(f"{p}.mlp.{name}.weight", shape, placement, layer, quantizable=True)
        else:
            add(p + ".mlp.gate.weight", (experts, h), layer=layer)
            add(p + ".mlp.gate.e_score_correction_bias", (experts,),
                layer=layer, dtype="F32")
            for name, shape, placement in (
                ("gate_proj", (expert_i, h), "expert_column"),
                ("up_proj", (expert_i, h), "expert_column"),
                ("down_proj", (h, expert_i), "expert_row"),
            ):
                add(f"{p}.mlp.experts.*.{name}.weight", shape, placement,
                    layer, experts, True)
            if shared:
                for name, shape, placement in (
                    ("gate_proj", (expert_i * shared, h), "column"),
                    ("up_proj", (expert_i * shared, h), "column"),
                    ("down_proj", (h, expert_i * shared), "row"),
                ):
                    add(f"{p}.mlp.shared_experts.{name}.weight", shape,
                        placement, layer, quantizable=True)
        if layer >= layers:
            add(p + ".eh_proj.weight", (h, 2 * h), layer=layer)
            for name in ("enorm", "hnorm", "shared_head.norm"):
                add(f"{p}.{name}.weight", (h,), layer=layer)
    out = [t for t in out if t.layer is None or t.layer >= layers or indexers[t.layer] == "full" or ".indexer." not in t.name]
    if c.get("moe_router_dtype") == "float32":
        out = [replace(t, dtype="F32") if t.name.endswith(".mlp.gate.weight") else t for t in out]
    return ModelProfile(model_id, revision, c, tuple(out), "mla_dsa", {
        "tensor_inventory": "architecture_derived",
        "mtp_included": include_mtp,
    })
