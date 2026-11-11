"""GQA/GDN hybrid MoE inventory."""
from __future__ import annotations
import copy
from ..errors import UnsupportedModelError
from ..types import ModelProfile, TensorSpec
from .common import _positive
from dataclasses import replace

def qwen35_moe_profile(config: dict, *, model_id="custom-qwen35-moe",
                       revision="user-config") -> ModelProfile:
    from .qwen3_moe import qwen3_moe_profile
    c = copy.deepcopy(config)
    if c.get("model_type") != "qwen3_5_moe_text":
        raise ValueError("Expected model_type=qwen3_5_moe_text")
    kinds = c.get("layer_types", [])
    n = _positive(c, "num_hidden_layers")
    if len(kinds) != n or set(kinds) != {"linear_attention", "full_attention"}:
        raise UnsupportedModelError("Expected explicit full/linear attention for every hybrid layer")
    if c.get("attention_bias", False) or c.get("tie_word_embeddings", False):
        raise UnsupportedModelError("Hybrid adapter requires bias-free attention and untied embeddings")
    if c.get("mamba_ssm_dtype", "float32") != "float32":
        raise UnsupportedModelError("Only FP32 recurrent state has an inspected contract")
    for key in ("linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
                "linear_value_head_dim", "linear_conv_kernel_dim", "shared_expert_intermediate_size"):
        _positive(c, key)
    if c["linear_num_value_heads"] % c["linear_num_key_heads"]:
        raise ValueError("Linear value heads must be divisible by linear key heads")
    dense = {**c, "model_type": "qwen3_moe"}
    dense.pop("layer_types")
    base = qwen3_moe_profile(dense, model_id=model_id, revision=revision)
    tensors = []
    for t in base.tensors:
        if t.layer is not None and ".self_attn." in t.name:
            if kinds[t.layer] == "linear_attention":
                continue
            if c.get("attn_output_gate", False) and t.name.endswith("q_proj.weight"):
                t = replace(t, shape=(2 * t.shape[0], t.shape[1]))
        tensors.append(t)
    h = c["hidden_size"]
    key = c["linear_num_key_heads"] * c["linear_key_head_dim"]
    value = c["linear_num_value_heads"] * c["linear_value_head_dim"]
    heads = c["linear_num_value_heads"]
    shared = c["shared_expert_intermediate_size"]
    for layer, kind in enumerate(kinds):
        prefix = f"model.layers.{layer}"
        if kind == "linear_attention":
            p = prefix + ".linear_attn"
            for name, shape, placement in (
                ("in_proj_qkvz", (2 * key + 2 * value, h), "column"),
                ("in_proj_ba", (2 * heads, h), "column"),
                ("out_proj", (h, value), "row"),
            ):
                tensors.append(TensorSpec(p + "." + name + ".weight", shape, placement, layer, quantizable=True, role="attention"))
            tensors.extend([
                TensorSpec(p + ".conv1d.weight", (2 * key + value, c["linear_conv_kernel_dim"]), "column", layer, role="attention"),
                TensorSpec(p + ".dt_bias", (heads,), "column", layer, role="attention"),
                TensorSpec(p + ".A_log", (heads,), "column", layer, dtype="F32", role="attention"),
                TensorSpec(p + ".norm.weight", (c["linear_value_head_dim"],), "replicated", layer, role="norm"),
            ])
        for name, shape, placement in (
            ("gate_proj", (shared, h), "column"),
            ("up_proj", (shared, h), "column"),
            ("down_proj", (h, shared), "row"),
        ):
            tensors.append(TensorSpec(prefix + ".mlp.shared_expert." + name + ".weight", shape, placement, layer, quantizable=True, role="shared_expert"))
        tensors.append(TensorSpec(prefix + ".mlp.shared_expert_gate.weight", (1, h), "replicated", layer, role="generic"))
    return ModelProfile(model_id, revision, c, tuple(tensors), "gdn_moe",
                        {"tensor_inventory": "architecture_derived", "mtp_included": False})
