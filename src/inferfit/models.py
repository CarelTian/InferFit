"""Architecture adapters: tensor inventory, not activated-parameter estimates."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from .types import ModelProfile, TensorSpec


def _positive(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"config.{key} must be a positive integer")
    return value


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


_MODEL_ADAPTERS = {}

def register_model_adapter(model_type: str, builder) -> None:
    """Register trusted Python config conversion; JSON never imports plugins."""
    if not isinstance(model_type,str) or not model_type or not callable(builder):
        raise ValueError("A model_type and callable builder are required")
    if model_type in _MODEL_ADAPTERS or model_type in {"glm_moe_dsa","qwen3_moe","llama","qwen2","qwen3","qwen3_5_moe_text"}:
        raise ValueError("Adapter already registered; built-ins cannot be replaced")
    _MODEL_ADAPTERS[model_type] = builder

def from_config(config: dict[str, Any] | str | Path, **kwargs: Any) -> ModelProfile:
    if isinstance(config, (str, Path)):
        config = json.loads(Path(config).read_text())
    if config.get("model_type") in _MODEL_ADAPTERS:
        from .profiles import validate_model_profile
        return validate_model_profile(_MODEL_ADAPTERS[config["model_type"]](copy.deepcopy(config), **kwargs))
    if config.get("model_type") == "qwen3_5_moe_text":
        from .flagships import qwen35_moe_profile
        return qwen35_moe_profile(config, **kwargs)
    if str(config.get("model_type", "")).startswith("deepseek_v4"):
        raise UnsupportedModelError("DeepSeek V4/V4.1 requires dedicated compressed/shared cache, Engram and mixed quantization adapters; no capacity pass is inferred")
    if config.get("model_type") == "qwen3_moe":
        return qwen3_moe_profile(config, **kwargs)
    if config.get("model_type") == "glm_moe_dsa":
        return glm_profile(config, **kwargs)
    return gqa_profile(config, **kwargs)


def load_glm51() -> ModelProfile:
    data = files("inferfit").joinpath("data")
    c = json.loads(data.joinpath("glm51-config.json").read_text())
    source = json.loads(data.joinpath("sources.json").read_text())
    p = glm_profile(c, model_id="zai-org/GLM-5.1", revision=source["bf16_revision"])
    return ModelProfile(p.model_id, p.revision, p.config, p.tensors, p.architecture,
                        {**p.provenance, **source})


class UnsupportedModelError(ValueError):
    """A recognized input cannot be modeled faithfully by this release."""


def qwen3_moe_profile(config: dict[str, Any], *, model_id: str = "custom-qwen3-moe",
                      revision: str = "user-config") -> ModelProfile:
    c = copy.deepcopy(config)
    if c.get("decoder_sparse_step", 1) != 1 or c.get("mlp_only_layers", []):
        raise UnsupportedModelError("Qwen3 MoE adapter currently requires every layer to be sparse")
    experts = _positive(c, "num_experts")
    width = _positive(c, "moe_intermediate_size")
    if _positive(c, "num_experts_per_tok") > experts:
        raise ValueError("num_experts_per_tok exceeds num_experts")
    dense = {**c, "model_type": "qwen3", "intermediate_size": width}
    dense.pop("num_experts", None)
    base = gqa_profile(dense, model_id=model_id, revision=revision)
    tensors = [t for t in base.tensors if ".mlp." not in t.name]
    h = c["hidden_size"]
    for layer in range(c["num_hidden_layers"]):
        p = f"model.layers.{layer}.mlp"
        tensors.append(TensorSpec(p+".gate.weight", (experts, h), "replicated", layer))
        for name, shape, kind in (("gate_proj", (width,h), "expert_column"),
                                  ("up_proj", (width,h), "expert_column"),
                                  ("down_proj", (h,width), "expert_row")):
            tensors.append(TensorSpec(p+f".experts.*.{name}.weight", shape, kind,
                                      layer, experts, True))
    c["head_dim"] = base.config["head_dim"]
    return ModelProfile(model_id, revision, c, tuple(tensors), "gqa_moe",
                        {"tensor_inventory":"architecture_derived", "mtp_included":False})


def load_qwen3_235b() -> ModelProfile:
    data = files("inferfit").joinpath("data")
    c = json.loads(data.joinpath("qwen3-235b-config.json").read_text())
    source = json.loads(data.joinpath("qwen3-235b-source.json").read_text())
    p = qwen3_moe_profile(c, model_id=source["model_id"], revision=source["revision"])
    return ModelProfile(p.model_id,p.revision,p.config,p.tensors,p.architecture,
                        {**p.provenance, "source":source})
