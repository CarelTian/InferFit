"""Capacity and execution compatibility are independent result axes."""
from dataclasses import asdict, dataclass, field
from typing import Any

from .types import Parallelism, RankMemory


@dataclass(frozen=True)
class MemoryAssumptions:
    # Generic is ordinary GQA only. Reference layouts require explicit opt-in.
    layout: str = "generic"
    reference_engine: str | None = None
    reference_version: str | None = None


@dataclass
class CapacityAssessment:
    status: str  # pass / fail / unknown / invalid
    basis: str
    reasons: list[str] = field(default_factory=list)
    missing_data: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    ranks: list[RankMemory] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class CompatibilityAssessment:
    status: str  # supported / unsupported / unknown
    backend_id: str | None = None
    reasons: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    model_id: str
    parallelism: Parallelism
    weight_quantization: str
    kv_quantization: str
    capacity: CapacityAssessment
    compatibility: CompatibilityAssessment
    verification: str = "not_run"

    @property
    def static_candidate(self) -> bool:
        return self.capacity.status == "pass" and self.compatibility.status == "supported"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "static_candidate": self.static_candidate, "deployment_verified": False}


@dataclass
class PlanSearchResult:
    candidates: list[AnalysisResult]
    evaluated: int
    capacity_counts: dict[str, int]
    compatibility_counts: dict[str, int]
    minimum_num_gpus: int | None
    policy: str
    rejection_reasons: dict[str, int]
    search_space: dict[str, Any]
    note: str = "Minimum within enumerated degrees and explicit memory assumptions; no execution verification."

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["candidates"] = [x.to_dict() for x in self.candidates]
        return result
