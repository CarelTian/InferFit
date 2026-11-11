"""Resolve execution evidence without substituting another device architecture."""
from dataclasses import dataclass

from ..engines import get_adapter
from ..analysis_types import CompatibilityAssessment
from ..types import RuntimeProfile


@dataclass(frozen=True)
class BackendResolution:
    kind: str  # inspected / unknown
    backend_id: str | None = None


def backend_catalog() -> dict:
    """List maintained upstream CUDA contracts, subject to per-request checks."""
    entries = []
    for runtime in (RuntimeProfile.sglang(), RuntimeProfile()):
        adapter = get_adapter(runtime)
        entries.append(dict(id=f"nvidia-{adapter.engine}-{adapter.version}",
                            engine=adapter.engine, version=adapter.version,
                            distribution="upstream", platform="cuda", vendor="nvidia",
                            evidence="inspected_engine_contract",
                            scope="Model, precision, and topology constraints are checked per request"))
    return {"schema_version": 1, "backends": entries}


def resolve_backend(hardware, runtime) -> BackendResolution:
    # CUDA profiles may omit vendor identity but must declare an architecture.
    if (hardware.platform == "cuda" and hardware.vendor in {None, "nvidia"}
            and type(hardware.compute_capability) is int and hardware.compute_capability > 0
            and runtime.distribution == "upstream" and not runtime.image
            and not runtime.build_revision and not runtime.sdk_version):
        try:
            adapter = get_adapter(runtime)
            if adapter.engine != "generic":
                return BackendResolution("inspected", f"nvidia-{adapter.engine}-{adapter.version}")
        except ValueError:
            pass
    return BackendResolution("unknown")


def assess_uninspected(resolution: BackendResolution) -> CompatibilityAssessment:
    return CompatibilityAssessment("unknown", backend_id=resolution.backend_id, reasons=["No inspected device/runtime build combination matched"])
