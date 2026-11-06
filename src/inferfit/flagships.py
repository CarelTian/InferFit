"""Offline flagship snapshots and hybrid state accounting.

Model availability and analytical support are deliberately separate fields.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from importlib.resources import files

from .types import ModelProfile, TensorSpec


def flagship_catalog() -> list[dict]:
    return json.loads(files("inferfit").joinpath("data/flagships.json").read_text())


def load_model(name: str) -> ModelProfile:
    """Load a named, frozen snapshot; never query the network at runtime."""
    from .models import from_config, UnsupportedModelError
    for entry in flagship_catalog():
        if name in {entry["id"], entry["alias"]}:
            if entry["support"] != "analytical":
                raise UnsupportedModelError("; ".join(entry["blockers"]))
            payload = files("inferfit").joinpath("data", entry["config_file"]).read_bytes()
            if hashlib.sha256(payload).hexdigest() != entry["config_sha256"]:
                raise ValueError("Bundled model snapshot digest mismatch")
            model = from_config(json.loads(payload), model_id=entry["id"], revision=entry["revision"])
            return replace(model, provenance={**model.provenance, "source": entry})
    raise ValueError(f"Unknown catalog model: {name}")


def qwen35_moe_profile(config: dict, *, model_id="custom-qwen35-moe",
                       revision="user-config") -> ModelProfile:
    from .models import qwen3_moe_profile, _positive, UnsupportedModelError
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


def recurrent_state_bytes(model: ModelProfile, *, tp=1, layer_start=0, layers=None) -> int:
    """Fixed BF16 conv + FP32 recurrent state per request on one TP rank."""
    if model.architecture != "gdn_moe":
        return 0
    c = model.config
    layers = c["num_hidden_layers"] - layer_start if layers is None else layers
    if type(tp) is not int or tp <= 0 or c["linear_num_key_heads"] % tp or c["linear_num_value_heads"] % tp:
        raise ValueError("GDN TP must divide linear key and value heads")
    if type(layer_start) is not int or type(layers) is not int or layer_start < 0 or layers < 0 or layer_start + layers > c["num_hidden_layers"]:
        raise ValueError("Invalid recurrent layer interval")
    count = c["layer_types"][layer_start:layer_start + layers].count("linear_attention")
    conv = (2 * c["linear_num_key_heads"] * c["linear_key_head_dim"] +
            c["linear_num_value_heads"] * c["linear_value_head_dim"])
    conv = conv // tp * (c["linear_conv_kernel_dim"] - 1) * 2
    temporal = c["linear_num_value_heads"] // tp * c["linear_key_head_dim"] * c["linear_value_head_dim"] * 4
    return count * (conv + temporal)
