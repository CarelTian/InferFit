"""Architecture-derived tensor inventory."""
from __future__ import annotations
import copy
from typing import Any
from ..errors import UnsupportedModelError
from ..types import ModelProfile, TensorSpec
from .common import _positive
from .gqa import gqa_profile

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
