"""Explicit engine/version dispatch. Unknown versions fail closed."""
from .base import EngineAdapter
from .vllm import VllmAdapter
from .sglang import SglangAdapter, SglangFlagshipAdapter
from ..engine_versions import SGLANG_CURRENT_VERSION, SGLANG_LEGACY_VERSION, SGLANG_LEGACY_COMMIT
from ..types import RuntimeProfile

# Deprecated compatibility aliases; internal dispatch uses explicit contract names.
SGLANG_VERSION = SGLANG_LEGACY_VERSION
SGLANG_COMMIT = SGLANG_LEGACY_COMMIT

__all__ = ["EngineAdapter", "VllmAdapter", "SglangAdapter", "SglangFlagshipAdapter",
           "SGLANG_VERSION", "SGLANG_COMMIT", "SGLANG_CURRENT_VERSION", "SGLANG_LEGACY_VERSION", "get_adapter"]

def get_adapter(runtime: RuntimeProfile) -> EngineAdapter:
    if runtime.engine == "vllm" and runtime.version == "0.19.0":
        return VllmAdapter()
    if runtime.engine == "sglang" and runtime.version == SGLANG_LEGACY_VERSION:
        return SglangAdapter()
    if runtime.engine == "sglang" and runtime.version == SGLANG_CURRENT_VERSION:
        return SglangFlagshipAdapter()
    raise ValueError(f"Supported pinned adapters: vllm 0.19.0, sglang {SGLANG_LEGACY_VERSION} (legacy), sglang {SGLANG_CURRENT_VERSION} (current model scope)")
