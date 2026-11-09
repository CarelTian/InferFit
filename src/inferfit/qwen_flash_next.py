"""Pinned Flash-Next inventory and QSA/PLE state contracts for SGLang 0.5.23."""
from __future__ import annotations

import copy
import json
import re
from importlib.resources import files

from .types import ModelProfile, TensorSpec


def flash_next_profile(config: dict, *, model_id="Qwen/Qwen3.8-Flash-Next",
                       revision="user-config", include_mtp=False) -> ModelProfile:
    from .models import UnsupportedModelError
    data = files("inferfit").joinpath("data")
    reference = json.loads(data.joinpath("qwen38-flash-next-config.json").read_text())
    if config != reference:
        raise UnsupportedModelError("Flash-Next requires the complete pinned text/vision config; custom variants, overrides and offload need a new inspected inventory")
    if include_mtp:
        raise UnsupportedModelError("Flash-Next MTP state/loading is not modeled")
    metadata = json.loads(data.joinpath("qwen38-flash-next-metadata.json").read_text())
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
        placement, role, multiplicity, owner = "replicated", "generic", 1, None
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
            multiplicity, shape = shape[0], shape[1:]
            placement = "expert_row" if name.endswith("down_proj") else "expert_column"
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
                              "vocab", ngram_layer, role="generic"))
    # SGLang regenerates this persistent scalar, absent from checkpoint headers.
    tensors.append(TensorSpec(name + ".weight_scale", (1,), "replicated", ngram_layer))
    return ModelProfile(model_id, revision, c, tuple(tensors), "gdn_qsa_moe", {
        "tensor_inventory": "pinned_checkpoint_headers",
        "mtp_included": False,
        "checkpoint_bytes": metadata["checkpoint_bytes"],
        "checkpoint_base_bytes": metadata["checkpoint_bytes"] - metadata["mtp_bytes"],
        "mtp_bytes_excluded": metadata["mtp_bytes"],
        "vision_weight_placement": "replicated_upper_bound",
        "scope": "text-only workload; GPU-resident PLE table; BF16 weights/KV",
    })


def flash_next_fixed_state(model: ModelProfile, *, layer_start=0, layers=None) -> dict[str, int]:
    """Per-request fixed PLE + QSA side state; QSA rings are TP-replicated."""
    if model.architecture != "gdn_qsa_moe":
        return {}
    c = model.config
    layers = c["num_hidden_layers"] - layer_start if layers is None else layers
    end = layer_start + layers
    owners = [layer - 1 for layer in c["ple_layer_ids"] if layer_start <= layer - 1 < end]
    qsa = c["layer_types"][layer_start:end].count("full_attention")
    ratio = c["indexer_compress_ratio"]
    return {
        "ple_short_conv_state_pool": len(owners) * c["hidden_size"] * c["hc_count"] * (c["ple_conv_kernel_size"] - 1) * c["ngram_size"] * 2,
        "ple_ngram_context_pool": (c["ngram_size"] - 1) * 8 if owners else 0,
        "qsa_pending_key_pool": qsa * ratio * c["indexer_kv_heads"] * c["indexer_head_dim"] * 2,
        "qsa_pending_position_pool": ratio * 3 * 8 if qsa else 0,
    }
