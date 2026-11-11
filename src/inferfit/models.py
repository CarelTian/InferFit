"""Public model loaders and explicit config-converter registration."""
from __future__ import annotations
import copy
import json
from importlib.resources import files
from pathlib import Path
from typing import Any
from .errors import UnsupportedModelError
from .types import ModelProfile
from .model_families.glm_dsa import glm_profile
from .model_families.gqa import gqa_profile
from .model_families.qwen3_moe import qwen3_moe_profile
from .model_families.qwen35 import qwen35_moe_profile
from .model_families.qwen_flash_next import flash_next_profile
from .model_families.deepseek_v4 import deepseek_v4_profile

_BUILTIN_ADAPTERS = {
    "glm_moe_dsa": glm_profile, "qwen3_moe": qwen3_moe_profile,
    "llama": gqa_profile, "qwen2": gqa_profile, "qwen3": gqa_profile,
    "qwen3_5_moe_text": qwen35_moe_profile,
    "qwen4_exp": flash_next_profile, "qwen4_exp_text": flash_next_profile,
    "deepseek_v4": deepseek_v4_profile,
}
_MODEL_ADAPTERS = {}


def register_model_adapter(model_type: str, builder) -> None:
    """Register trusted Python conversion; JSON never imports executable code."""
    if not isinstance(model_type, str) or not model_type or not callable(builder):
        raise ValueError("A model_type and callable builder are required")
    if model_type in _MODEL_ADAPTERS or model_type in _BUILTIN_ADAPTERS:
        raise ValueError("Adapter already registered; built-ins cannot be replaced")
    _MODEL_ADAPTERS[model_type] = builder


def from_config(config: dict[str, Any] | str | Path, **kwargs: Any) -> ModelProfile:
    if isinstance(config, (str, Path)):
        config = json.loads(Path(config).read_text())
    if not isinstance(config, dict):
        raise ValueError("Model config must be a JSON object")
    model_type = config.get("model_type")
    if model_type in _MODEL_ADAPTERS:
        from .profiles import validate_model_profile
        return validate_model_profile(_MODEL_ADAPTERS[model_type](copy.deepcopy(config), **kwargs))
    if model_type in _BUILTIN_ADAPTERS:
        return _BUILTIN_ADAPTERS[model_type](config, **kwargs)
    if str(model_type).startswith("deepseek_v4"):
        raise UnsupportedModelError("DeepSeek V4.1 requires its own compressed/shared cache, Engram and quantization contract; original V4 Flash/Pro support does not imply V4.1 support")
    return gqa_profile(config, **kwargs)


def load_glm51() -> ModelProfile:
    data = files("inferfit").joinpath("data")
    c = json.loads(data.joinpath("glm51-config.json").read_text())
    source = json.loads(data.joinpath("sources.json").read_text())
    p = glm_profile(c, model_id="zai-org/GLM-5.1", revision=source["bf16_revision"])
    return ModelProfile(p.model_id, p.revision, p.config, p.tensors, p.architecture,
                        {**p.provenance, **source})


def load_qwen3_235b() -> ModelProfile:
    data = files("inferfit").joinpath("data")
    c = json.loads(data.joinpath("qwen3-235b-config.json").read_text())
    source = json.loads(data.joinpath("qwen3-235b-source.json").read_text())
    p = qwen3_moe_profile(c, model_id=source["model_id"], revision=source["revision"])
    return ModelProfile(p.model_id,p.revision,p.config,p.tensors,p.architecture,
                        {**p.provenance, "source":source})
