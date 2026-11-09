"""QSA and PLE request-scoped side states."""
from __future__ import annotations
from ..types import ModelProfile

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
