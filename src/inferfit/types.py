"""Public, JSON-friendly input and output types. All memory is integer bytes."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

GiB = 1024**3


@dataclass(frozen=True)
class Hardware:
    name: str = "NVIDIA H800 80GB"
    # Nominal decimal vendor capacity, not an assertion about driver-visible HBM.
    memory_bytes: int = 80_000_000_000
    compute_capability: int | None = 90
    gpus_per_node: int = 8
    external_used_bytes_per_gpu: int = 0
    utilization: float = 0.9
    safety_margin_bytes: int = GiB
    inter_node_network: bool = True
    allow_cross_node_tp: bool = False
    platform: str = "cuda"
    available_nodes: int | None = None


@dataclass(frozen=True)
class Parallelism:
    tp: int = 8
    dp: int = 1
    ep: int = 1
    pp: int = 1
    num_gpus: int = 8
    # Optional contiguous, explicit PP partition; otherwise balanced by layer count.
    stage_layers: tuple[int, ...] | None = None


@dataclass(frozen=True)
class Workload:
    # Total resident context per request: prompt + generated/reserved output.
    context_tokens: int = 8192
    concurrency_per_dp: int = 1
    max_batch_tokens: int = 2048
    block_size: int = 64


@dataclass(frozen=True)
class RuntimeProfile:
    name: str = "vllm-0.19.0-hopper-eager-assumed"
    engine: str = "vllm"
    version: str = "0.19.0"
    provenance: str = "assumed"
    enforce_eager: bool = True
    cuda_context_bytes: int = GiB
    communication_bytes: int = 2 * GiB
    allocator_bytes: int = GiB
    graph_bytes: int = 0
    activation_fixed_bytes: int = GiB
    # Analytical envelope, not a calibrated kernel allocation formula.
    activation_copies: int = 16
    load_workspace_bytes: int = 2 * GiB
    extra_bytes: int = 0
    max_batch_tokens_supported: int = 8192
    max_concurrency_per_dp_supported: int = 128
    # Payload -> physical cache pool overhead; 1.0 assumes compact mixed pages.
    kv_pool_overhead_ratio: float = 1.0
    # Public TP is attention TP per DP replica; native SGLang TP then is TP*DP.
    sglang_dp_attention: bool = False

    @classmethod
    def sglang(cls, **overrides: Any) -> RuntimeProfile:
        """Pinned eager SGLang profile; reserves remain explicit assumptions."""
        values = dict(name="sglang-0.5.10-hopper-eager-assumed", engine="sglang",
                      version="0.5.10", activation_fixed_bytes=2 * GiB)
        values.update(overrides)
        return cls(**values)


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    # replicated / column / row / vocab / expert_column / expert_row
    placement: str
    layer: int | None = None
    multiplicity: int = 1
    # Some GLM indexer/routing/normalization modules stay BF16 / FP32.
    quantizable: bool = False
    dtype: str = "BF16"
    role: str = "auto"
    pp_owner: str | None = None


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    revision: str
    config: dict[str, Any]
    tensors: tuple[TensorSpec, ...]
    architecture: str
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class RankMemory:
    rank: int
    pp_rank: int
    dp_rank: int
    tp_rank: int
    node: int
    layer_start: int
    layer_end: int
    weight_bytes: int
    kv_bytes_per_token: int
    kv_breakdown_per_token: dict[str, int]
    kv_allocated_tokens: int
    kv_required_bytes: int
    runtime_breakdown: dict[str, int]
    runtime_peak_bytes: int
    load_peak_bytes: int
    serving_peak_bytes: int
    budget_bytes: int
    peak_bytes: int
    headroom_bytes: int
    # Arithmetic physical slack at peak, not nvidia-smi free after pool preallocation.
    physical_slack_at_peak_bytes: int
    kv_capacity_tokens: int
    memory_pass: bool


@dataclass
class CheckResult:
    status: str
    memory_pass: bool | None
    deployability: str
    model_id: str
    parallelism: Parallelism
    weight_quantization: str
    kv_quantization: str
    confidence: str
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ranks: list[RankMemory] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResult:
    candidates: list[CheckResult]
    evaluated: int
    counts: dict[str, int]
    minimum_num_gpus: int | None
    search_space: dict[str, Any]
    note: str = (
        "Minimum is within the enumerated search space. Candidates passed memory "
        "and static capability checks; no GPU execution or throughput validation."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
