"""Resolve exact offline inventories from the versioned data catalog.

Builders contain architecture rules; new same-family snapshots belong in data.
"""
from __future__ import annotations

import hashlib
import json
from importlib.resources import files

from .errors import UnsupportedModelError


def inventory_for_config(config: dict) -> dict:
    data = files("inferfit").joinpath("data")
    for entry in json.loads(data.joinpath("flagships.json").read_text()):
        if entry["support"] != "analytical" or not entry.get("metadata_file"):
            continue
        payload = data.joinpath(entry["config_file"]).read_bytes()
        reference = json.loads(payload)
        if config != reference:
            continue
        if hashlib.sha256(payload).hexdigest() != entry["config_sha256"]:
            raise ValueError("Bundled model snapshot digest mismatch")
        payload = data.joinpath(entry["metadata_file"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry["metadata_sha256"]:
            raise ValueError("Bundled tensor inventory digest mismatch")
        metadata = json.loads(payload)
        if (metadata["model_id"], metadata["revision"]) != (entry["id"], entry["revision"]):
            raise ValueError("Tensor inventory identity disagrees with catalog")
        return metadata
    raise UnsupportedModelError("Model requires a complete pinned config and an inspected tensor inventory; custom variants, overrides and offload need a new snapshot")
