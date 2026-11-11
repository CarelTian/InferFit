"""Named nominal-capacity fixtures. Runtime-visible bytes override specifications."""
from dataclasses import replace
from importlib.resources import files
import json

from .types import Hardware

CATALOG = {
    "h800-80gb": (80_000_000_000, 90, "cuda"),
    "h20-96gb": (96_000_000_000, 90, "cuda"),
    "h100-sxm-80gb": (80_000_000_000, 90, "cuda"),
    "a100-80gb": (80_000_000_000, 80, "cuda"),
    "a100-40gb": (40_000_000_000, 80, "cuda"),
}


def hardware_catalog() -> dict:
    """Return named NVIDIA fixtures and provenance without network access."""
    sources = json.loads(files("inferfit").joinpath("data/hardware-sources.json").read_text())
    return {
        "schema_version": 1,
        "capacity_policy": sources["capacity_policy"],
        "sources": sources["sources"],
        "variants": {name: dict(memory_bytes=memory, compute_capability=cc,
                                platform=platform, vendor="nvidia")
                     for name, (memory, cc, platform) in CATALOG.items()},
    }


def load_hardware(name: str, *, memory_bytes: int | None = None, **overrides) -> Hardware:
    if name not in CATALOG:
        raise ValueError(f"Unknown hardware variant: {name}")
    nominal, cc, platform = CATALOG[name]
    values = dict(vendor="nvidia", memory_provenance="caller_override" if memory_bytes is not None else "published_nominal")
    values.update(overrides)
    return replace(Hardware(name=name, memory_bytes=nominal if memory_bytes is None else memory_bytes,
                            compute_capability=cc, platform=platform), **values)
