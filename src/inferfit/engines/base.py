"""Engine-neutral adapter contract and conventional rank coordinates."""
from __future__ import annotations
from abc import ABC, abstractmethod
from ..types import Hardware, ModelProfile, Parallelism, RuntimeProfile, TensorSpec, Workload

class EngineAdapter(ABC):
    engine: str
    version: str

    def native_tp(self, p: Parallelism, r: RuntimeProfile) -> int:
        return p.tp

    def rank(self, pp: int, dp: int, tp: int, p: Parallelism, r: RuntimeProfile) -> int:
        return (dp * p.pp + pp) * p.tp + tp

    def rank_order(self, r: RuntimeProfile) -> str:
        return "dp,pp,tp (tp fastest)"

    def tensor_dtype(self, t: TensorSpec) -> str:
        return t.dtype

    def batch_multiplier(self, p: Parallelism, r: RuntimeProfile) -> int:
        return 1

    def resident_buffers(self, model, plan, workload, runtime) -> dict[str, int]:
        return {}

    def load_workspace_bytes(self, model, plan, runtime) -> int:
        return runtime.load_workspace_bytes

    def supports_native_layout(self, model: ModelProfile) -> bool:
        return False

    @abstractmethod
    def normalize_kv(self, value: str, model: ModelProfile) -> str:
        raise NotImplementedError

    @abstractmethod
    def expert_choices(self, p: Parallelism, r: RuntimeProfile) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def topology(self, h: Hardware, p: Parallelism, r: RuntimeProfile) -> tuple[int, int]:
        raise NotImplementedError

    @abstractmethod
    def tensor_plan(self, t: TensorSpec, p: Parallelism, r: RuntimeProfile) -> tuple[str, int, int]:
        raise NotImplementedError

    @abstractmethod
    def projection_copies(self, fmt: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def metadata_bytes(self, w: Workload) -> int:
        raise NotImplementedError

    @abstractmethod
    def workspace_bytes(self, w: Workload) -> int:
        raise NotImplementedError

    @abstractmethod
    def validate(self, model, hardware, plan, workload, runtime, weight_format, kv_format):
        """Return unsupported execution-contract reasons."""
        raise NotImplementedError

    @abstractmethod
    def engine_config(self, model, hardware, plan, workload, runtime, weight_format,
                      kv_format, budget, counts, tokens):
        """Return engine-native sizing arguments and environment settings."""
        raise NotImplementedError
