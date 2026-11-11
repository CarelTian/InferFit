"""Pinned Flash-Next inventory and QSA/PLE state contracts for SGLang 0.5.23."""
from __future__ import annotations

import copy
import re

from ..snapshots import inventory_for_config
from ..types import ModelProfile, TensorSpec


def flash_next_profile(config: dict, *, model_id="Qwen/Qwen3.8-Flash-Next",
                       revision="user-config", include_mtp=False) -> ModelProfile:
    from ..errors import UnsupportedModelError
    if config.get("model_type") not in {"qwen4_exp", "qwen4_exp_text"}:
        raise UnsupportedModelError("Flash-Next requires a Qwen4-Exp config")
    native = "quantization_config" in config
    if native and (config["quantization_config"].get("quant_method") != "fp8" or
                   config["quantization_config"].get("weight_block_size") != [128, 128]):
        raise UnsupportedModelError("Flash-Next native contract requires FP8 128x128 blocks")
    if include_mtp:
        raise UnsupportedModelError("Flash-Next MTP state/loading is not modeled")
    metadata = inventory_for_config(config)
    c = copy.deepcopy(config["text_config"])
    c["vision_config"] = copy.deepcopy(config["vision_config"])
    tensors = []
    ngram_rows = 0
    ngram_shape = None
    ngram_layer = None
    for row in metadata["tensors"]:
        name, shape, dtype = row["name"], tuple(row["shape"]), row["dtype"]
        if name.startswith("mtp."):
            continue
        match = re.search(r"\.layers\.(\d+)\.", name)
        layer = int(match[1]) if match else None
        placement, role, multiplicity, owner = "replicated", "generic", row.get("multiplicity", 1), None
        if ".ngram_embedding.shard_" in name:
            if ngram_shape is not None and (shape[1:] != ngram_shape or layer != ngram_layer):
                raise ValueError("Unexpected N-gram checkpoint shards")
            ngram_rows += shape[0]
            ngram_shape, ngram_layer = shape[1:], layer
            continue
        if name == "lm_head.weight":
            placement, role, owner = "vocab", "output", "last"
        elif name.endswith("embed_tokens.weight"):
            placement, role, owner = "vocab", "embedding", "first"
        elif name.startswith("model.visual."):
            # Conservative upper budget: do not assume vision TP/DP sharding.
            owner = "first"
        elif ".mlp.experts." in name:
            if not native:
                multiplicity, shape = shape[0], shape[1:]
            placement = "expert_row" if "down_proj" in name else "expert_column"
            role = "routed_expert"
        elif ".mlp.shared_expert." in name:
            placement = "row" if name.endswith("down_proj.weight") else "column"
            role = "shared_expert"
        elif ".linear_attn." in name:
            role = "attention"
            if name.endswith("out_proj.weight"):
                placement = "row"
            elif not name.endswith("norm.weight"):
                placement = "column"
        elif ".self_attn." in name:
            role = "attention"
            if ".indexer." not in name:
                if name.endswith("q_proj.weight"):
                    placement = "column"
                elif name.endswith(("k_proj.weight", "v_proj.weight")):
                    placement = "kv_column"
                elif name.endswith("o_proj.weight"):
                    placement = "row"
        if layer is None and owner is None:
            owner = "last"
        tensors.append(TensorSpec(name, shape, placement, layer, multiplicity,
                                  dtype=dtype, role=role, pp_owner=owner))
    name = f"model.language_model.layers.{ngram_layer}.ple.ple_embedding.ngram_embedding"
    tensors.append(TensorSpec(name + ".weight", (ngram_rows,) + ngram_shape,
                              "vocab", ngram_layer, dtype="F8_E4M3" if native else "BF16", role="generic"))
    # SGLang regenerates this persistent scalar, absent from checkpoint headers.
    if not native:
        tensors.append(TensorSpec(name + ".weight_scale", (1,), "replicated", ngram_layer))
    return ModelProfile(model_id, revision, c, tuple(tensors), "gdn_qsa_moe", {
        "tensor_inventory": "pinned_checkpoint_headers",
        "mtp_included": False,
        "checkpoint_bytes": metadata["checkpoint_bytes"],
        "checkpoint_base_bytes": metadata["checkpoint_bytes"] - metadata["mtp_bytes"],
        "mtp_bytes_excluded": metadata["mtp_bytes"],
        "vision_weight_placement": "replicated_upper_bound",
        "scope": "text-only workload; GPU-resident PLE table; BF16 KV",
        **({"native_quantization": "fp8_experts_and_ngram_with_bf16_exclusions"} if native else {}),
    })

