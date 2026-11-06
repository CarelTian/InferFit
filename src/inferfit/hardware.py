"""Named nominal-capacity fixtures. Runtime-visible bytes always override specs."""
from dataclasses import replace
from .types import Hardware

CATALOG = {
 "h800-80gb": (80_000_000_000,90,"cuda"),
 "h20-96gb": (96_000_000_000,90,"cuda"),
 "h100-sxm-80gb": (80_000_000_000,90,"cuda"),
 "a100-80gb": (80_000_000_000,80,"cuda"),
 "a100-40gb": (40_000_000_000,80,"cuda"),
 "ppu-zw810": (None,None,"ppu"),
}

def load_hardware(name: str, *, memory_bytes: int | None = None, **overrides) -> Hardware:
    if name not in CATALOG:
        raise ValueError(f"Unknown hardware variant: {name}")
    capacity, cc, platform = CATALOG[name]
    capacity = memory_bytes if memory_bytes is not None else capacity
    if capacity is None:
        raise ValueError("ZW810 requires actual ppu-smi memory_bytes; no SKU capacity is assumed")
    return replace(Hardware(name=name,memory_bytes=capacity,compute_capability=cc,platform=platform),**overrides)
