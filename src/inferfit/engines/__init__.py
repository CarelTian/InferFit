"""Explicit engine/version dispatch. Unknown versions fail closed."""
from .base import EngineAdapter
from .vllm import VllmAdapter
from .sglang import SglangAdapter
from ..engine_versions import SGLANG_VERSION
from ..types import RuntimeProfile

__all__ = ["EngineAdapter", "VllmAdapter", "SglangAdapter", "get_adapter"]

def get_adapter(runtime: RuntimeProfile) -> EngineAdapter:
    if runtime.engine == "generic" and runtime.version == "1":
        from .generic import GenericMemoryAdapter
        return GenericMemoryAdapter()
    if runtime.engine == "vllm" and runtime.version == "0.19.0":
        return VllmAdapter()
    if runtime.engine == "sglang" and runtime.version == SGLANG_VERSION:
        return SglangAdapter()
    raise ValueError(f"Supported pinned adapters: vllm 0.19.0, sglang {SGLANG_VERSION}")
