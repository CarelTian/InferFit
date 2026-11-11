"""Explicit memory layout assumptions, never an executable engine contract."""
from dataclasses import dataclass

from .base import EngineAdapter


@dataclass(frozen=True)
class GenericMemoryAdapter(EngineAdapter):
    engine: str = "generic"
    version: str = "1"

    def normalize_kv(self, value, model):
        return {"bfloat16": "bf16", "float16": "fp16", "fp8_e4m3": "fp8"}.get(value, value)

    def expert_choices(self, p, r):
        return [x for x in range(1, p.tp + 1) if p.tp % x == 0]

    def topology(self, h, p, r):
        return (p.num_gpus + h.gpus_per_node - 1) // h.gpus_per_node, h.gpus_per_node

    def tensor_plan(self, t, p, r):
        kind, width, copies = t.placement, p.tp, t.multiplicity
        if kind.startswith("expert_"):
            copies //= p.ep
            width //= p.ep
            kind = kind.removeprefix("expert_")
        return kind, width, copies

    def projection_copies(self, fmt):
        return False

    def metadata_bytes(self, w):
        return w.concurrency_per_dp * ((w.context_tokens + w.block_size - 1) // w.block_size) * 4

    def workspace_bytes(self, w):
        return 0  # Only ordinary GQA is accepted; fixed workspace is a RuntimeProfile input.

    def validate(self, *args):
        return ["Generic memory layout is capacity-only, not an execution backend"]

    def engine_config(self, *args):
        return {}, {}
