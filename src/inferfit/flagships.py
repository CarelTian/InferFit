"""Offline flagship snapshots and hybrid state accounting.

Model availability and analytical support are deliberately separate fields.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from importlib.resources import files

from .types import ModelProfile


def flagship_catalog() -> list[dict]:
    return json.loads(files("inferfit").joinpath("data/flagships.json").read_text())


def load_model(name: str) -> ModelProfile:
    """Load a named, frozen snapshot; never query the network at runtime."""
    from .models import from_config, UnsupportedModelError
    for entry in flagship_catalog():
        if name in {entry["id"], entry["alias"]}:
            if entry["support"] != "analytical":
                raise UnsupportedModelError("; ".join(entry["blockers"]))
            payload = files("inferfit").joinpath("data", entry["config_file"]).read_bytes()
            if hashlib.sha256(payload).hexdigest() != entry["config_sha256"]:
                raise ValueError("Bundled model snapshot digest mismatch")
            if entry.get("metadata_file"):
                metadata = files("inferfit").joinpath("data", entry["metadata_file"]).read_bytes()
                if hashlib.sha256(metadata).hexdigest() != entry["metadata_sha256"]:
                    raise ValueError("Bundled tensor inventory digest mismatch")
            model = from_config(json.loads(payload), model_id=entry["id"], revision=entry["revision"])
            from .profiles import validate_model_profile
            validate_model_profile(model)
            return replace(model, provenance={**model.provenance, "source": entry})
    raise ValueError(f"Unknown catalog model: {name}")
