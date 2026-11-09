"""GDN request-scoped convolution and recurrent state geometry."""
from __future__ import annotations
from ..types import ModelProfile

def recurrent_state_bytes(model: ModelProfile, *, tp=1, layer_start=0, layers=None) -> int:
    """Fixed BF16 conv + FP32 recurrent state per request on one TP rank."""
    if model.architecture not in {"gdn_moe", "gdn_qsa_moe"}:
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
