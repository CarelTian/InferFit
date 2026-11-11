"""Original DeepSeek-V4 Flash/Pro native FP8 + MXFP4 inventories.

V4.1 is a separate contract: sharing, Engram and new quantization blocks may
not be inferred from these original V4 snapshots.
"""
from __future__ import annotations

import copy
import re

from ..snapshots import inventory_for_config
from ..types import ModelProfile, TensorSpec


def deepseek_v4_profile(config: dict, *, model_id="custom-deepseek-v4",
                        revision="user-config", include_mtp=False) -> ModelProfile:
    from ..errors import UnsupportedModelError
    quant = config.get("quantization_config", {})
    if (config.get("model_type") != "deepseek_v4" or config.get("expert_dtype") != "fp4" or
            quant.get("quant_method") != "fp8" or quant.get("weight_block_size") != [128, 128] or
            quant.get("scale_fmt") != "ue8m0"):
        raise UnsupportedModelError("Original V4 native contract requires 128x128 UE8M0 FP8 and FP4 experts")
    if include_mtp:
        raise UnsupportedModelError("Original V4 requires a complete pinned Flash/Pro config and base-only loading; custom variants and MTP need a new inspected contract")
    metadata = inventory_for_config(config)
    tensors = []
    for row in metadata["tensors"]:
        name = row["name"]
        if name.startswith("mtp."):
            continue
        match = re.match(r"layers\.(\d+)\.", name)
        layer = int(match[1]) if match else None
        placement, role, owner = "replicated", "generic", None
        if name == "embed.weight":
            placement, role, owner = "vocab", "embedding", "first"
        elif name == "head.weight":
            placement, role, owner = "vocab", "output", "last"
        elif ".ffn.experts." in name:
            placement = "expert_row" if ".w2." in name else "expert_column"
            role = "routed_expert"
        elif ".ffn.shared_experts." in name:
            placement = "row" if ".w2." in name else "column"
            role = "shared_expert"
        elif ".attn." in name:
            role = "attention"
            # Indexer, compressors, wq_a and shared wkv are replicated.
            if ".indexer." not in name and ".compressor." not in name:
                if ".wq_b." in name or ".wo_a." in name:
                    placement = "column"
                elif ".wo_b." in name:
                    placement = "row"
        if layer is None and owner is None:
            owner = "last"
        tensors.append(TensorSpec(name, tuple(row["shape"]), placement, layer,
            row.get("multiplicity", 1), dtype=row["dtype"], role=role, pp_owner=owner))
    c = copy.deepcopy(config)
    # Keep only base layers; the final source ratio belongs to the MTP layer.
    c["compress_ratios"] = c["compress_ratios"][:c["num_hidden_layers"]]
    c["qk_nope_head_dim"] = c["head_dim"] - c["qk_rope_head_dim"]
    return ModelProfile(model_id, revision, c, tuple(tensors), "dsv4", {
        "tensor_inventory": "pinned_checkpoint_headers",
        "native_quantization": "fp8_128x128_ue8m0_dense_mxfp4_experts",
        "checkpoint_bytes": metadata["checkpoint_bytes"],
        "checkpoint_base_bytes": metadata["checkpoint_bytes"] - metadata["mtp_bytes"],
        "mtp_bytes_excluded": metadata["mtp_bytes"], "mtp_included": False,
        "runtime_scale_budget": "FP32 upper bound for dense scales; native E8M0 expert scales",
    })
