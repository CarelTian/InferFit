"""Cache/state accounting contract; exact allocations may be nonlinear in tokens."""
from __future__ import annotations

from dataclasses import dataclass
import math


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass(frozen=True)
class StateAllocation:
    kv_parts: dict[str, int]
    allocated_tokens: int
    cache_parts: dict[str, int]
    runtime_parts: dict[str, int]

    @property
    def cache_bytes(self) -> int:
        return sum(self.cache_parts.values())

    @property
    def total_bytes(self) -> int:
        return self.cache_bytes + sum(self.runtime_parts.values())


class StandardStateModel:
    def per_token(self, *args, **kwargs):
        from .standard import kv_per_token
        return kv_per_token(*args, **kwargs)

    def token_cap(self, tokens: int) -> int:
        return tokens

    def activation_width(self, model) -> int:
        return model.config["hidden_size"] * (model.config["hc_count"] if model.architecture == "gdn_qsa_moe" else 1)

    def allocate(self, model, p, w, r, kv, start, count, *, tokens=None):
        parts = self.per_token(model, tp=p.tp, layers=count, layer_start=start,
                              kv_quantization=kv, runtime=r)
        tokens = w.concurrency_per_dp * ceildiv(w.context_tokens, w.block_size) * w.block_size if tokens is None else tokens
        allocated = tokens + w.block_size
        block_bytes = math.ceil(parts["total"] * w.block_size * r.kv_pool_overhead_ratio)
        runtime = {}
        if model.architecture in {"gdn_moe", "gdn_qsa_moe"}:
            from .gdn import recurrent_state_bytes
            runtime["recurrent_state_pool"] = recurrent_state_bytes(model, tp=p.tp,
                layer_start=start, layers=count) * (w.concurrency_per_dp + 1)
        if model.architecture == "gdn_qsa_moe":
            from .qwen_flash_next import flash_next_fixed_state
            runtime.update({key: value * (w.concurrency_per_dp + 1) for key, value in
                flash_next_fixed_state(model, layer_start=start, layers=count).items()})
            c = model.config
            runtime["qsa_prefill_workspace_envelope"] = (128 * 1024**2 +
                w.max_batch_tokens * (c["indexer_budget"] + c["indexer_compress_ratio"] - 1) * 4)
        return StateAllocation(parts, allocated, {"paged_kv": allocated // w.block_size * block_bytes}, runtime)

    def capacity(self, model, p, w, r, kv, start, count, available):
        """Preserve the SDK's linear-family hardware-capacity arithmetic."""
        allocation = self.allocate(model, p, w, r, kv, start, count, tokens=0)
        block_bytes = allocation.cache_bytes
        available -= sum(allocation.runtime_parts.values())
        if block_bytes:
            return max(0, available // block_bytes - 1) * w.block_size
        return ceildiv(model.config["max_position_embeddings"] * w.concurrency_per_dp, w.block_size) * w.block_size

    def capacity_from_allocations(self, model, p, w, r, kv, start, count, available):
        """Aggregate token capacity at fixed request slots and runtime envelope."""
        # Binary search allocations, never divide a nonlinear state pool by a rate.
        upper = ceildiv(model.config["max_position_embeddings"] * w.concurrency_per_dp, w.block_size)
        lo, hi = 0, upper + 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            allocation = self.allocate(model, p, w, r, kv, start, count, tokens=mid * w.block_size)
            if allocation.total_bytes <= available:
                lo = mid
            else:
                hi = mid
        return lo * w.block_size
