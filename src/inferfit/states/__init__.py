"""Memory-family dispatch. Model loaders describe geometry; engines choose layouts."""
from .base import StandardStateModel
from .deepseek_v4 import DeepSeekV4StateModel

_STANDARD = StandardStateModel()
_STATE_MODELS = {name: _STANDARD for name in ("gqa", "gqa_moe", "mla_dsa", "gdn_moe", "gdn_qsa_moe")}
_STATE_MODELS["dsv4"] = DeepSeekV4StateModel()


def get_state_model(model):
    try:
        return _STATE_MODELS[model.architecture]
    except KeyError:
        raise ValueError(f"No state model for memory family: {model.architecture}") from None
