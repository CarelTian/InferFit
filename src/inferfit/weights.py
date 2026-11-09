"""Storage arithmetic, independent of model names and execution engines.

Native inventories already contain packed weights AND their scale tensors.
Never quantize them again or infer their size from the advertised precision.
"""
from __future__ import annotations

from functools import lru_cache
import math

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8, "I32": 4,
               "I8": 1, "U8": 1, "F8_E4M3": 1, "F8_E8M0": 1}


@lru_cache(maxsize=32768)
def storage_bytes(shape: tuple[int, ...], dtype: str, quantized: bool = False) -> int:
    if quantized:
        if len(shape) != 2:
            raise ValueError("Block quantization requires a matrix")
        return math.prod(shape) + 4 * ((shape[0] + 127) // 128) * ((shape[1] + 127) // 128)
    try:
        return math.prod(shape) * DTYPE_BYTES[dtype]
    except KeyError:
        raise ValueError(f"Unsupported tensor storage dtype: {dtype}") from None


def validate_weight_format(model, fmt: str) -> None:
    if model.provenance.get("native_quantization"):
        if fmt not in {"fp8_block", "native"}:
            raise ValueError("Native quantized checkpoint requires weight_quantization='native' or 'fp8'; BF16 conversion is not modeled")
    elif fmt == "native":
        raise ValueError("This model does not have a pinned native quantized inventory")
    elif model.architecture == "gdn_qsa_moe" and fmt != "bf16":
        raise ValueError("BF16 Flash-Next inventory covers BF16 weights only; load qwen3.8-flash-next-fp8 for the native FP8 checkpoint")

