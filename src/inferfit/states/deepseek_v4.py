"""SGLang 0.5.23 CUDA V4-layout paged cache, explicit SWA/full ratio=1.

Source contracts: DSV4PoolConfigurator, DeepSeekV4TokenToKVPool,
DeepSeekV4SingleKVPool, DeepSeekV4IndexerPool and CompressStatePool.
No unified HIP ring, online compressor, HiSparse, MTP or V4.1 sharing.
"""
from __future__ import annotations

from fractions import Fraction
from dataclasses import replace
import math

from .base import StandardStateModel, StateAllocation, ceildiv
from ..engine_versions import SGLANG_VERSION


def page_bytes(page_size: int) -> int:
    # 448 FP8 nope + 64 BF16 rope + 8 scale/pad bytes, padded to 576-byte rows.
    return ceildiv(page_size * 584, 576) * 576


class DeepSeekV4StateModel(StandardStateModel):
    def prepare_interval_cost(self, model, p, w, r, kv):
        """Price pool ownership counts using the allocator's actual raw bytes.

        Page padding is inside each coefficient. Pool-level overhead is rounded
        only after multiplying by the interval's owner count; allocator metadata
        is charged once per stage, including stages without C4/C128 owners.
        """
        n = model.config["num_hidden_layers"]
        c4_prefix, c128_prefix = [0], [0]
        for ratio in model.config["compress_ratios"]:
            c4_prefix.append(c4_prefix[-1] + (ratio == 4))
            c128_prefix.append(c128_prefix[-1] + (ratio == 128))
        layers_prefix = range(n + 1)
        cache_owners = {"swa_paged_kv": layers_prefix,
                        "c4_compressed_kv": c4_prefix,
                        "c4_index_keys_and_scales": c4_prefix,
                        "c128_compressed_kv": c128_prefix}
        runtime_owners = {"c4_compressor_state": c4_prefix,
                          "c4_indexer_compressor_state": c4_prefix,
                          "c128_compressor_state": c128_prefix}
        allocation = self.allocate(model, p, w, replace(r, kv_pool_overhead_ratio=1.0), kv, 0, n)
        # A future layout/component must not silently inherit these assumptions.
        if (allocation.cache_parts.keys() != cache_owners.keys() or
                allocation.runtime_parts.keys() != runtime_owners.keys() | {"v4_allocator_metadata_envelope"}):
            return self._cached_interval_cost(model, p, w, r, kv)
        cache_units, runtime_units = [], []
        for parts, owners_by_name, units in ((allocation.cache_parts, cache_owners, cache_units),
                                             (allocation.runtime_parts, runtime_owners, runtime_units)):
            for key, owners in owners_by_name.items():
                value, count = parts[key], owners[-1]
                if (count and value % count) or (not count and value):
                    return self._cached_interval_cost(model, p, w, r, kv)
                if count:
                    units.append((value // count, owners))
        fixed = allocation.runtime_parts["v4_allocator_metadata_envelope"]
        overhead = r.kv_pool_overhead_ratio

        def cost(start: int, end: int) -> int:
            cache = sum(math.ceil(unit * (owners[end] - owners[start]) * overhead)
                        for unit, owners in cache_units)
            state = sum(unit * (owners[end] - owners[start]) for unit, owners in runtime_units)
            return cache + state + fixed
        return cost

    def capacity(self, *args, **kwargs):
        return self.capacity_from_allocations(*args, **kwargs)

    def per_token(self, model, *, tp=1, layers=None, kv_quantization="fp8",
                  runtime=None, layer_start=0):
        if type(tp) is not int or tp <= 0:
            raise ValueError("tp must be positive")
        if runtime is None or (runtime.engine, runtime.version) != ("sglang", SGLANG_VERSION):
            raise ValueError("Original V4 cache requires SGLang 0.5.23")
        if kv_quantization not in {"fp8", "fp8_e4m3"}:
            raise ValueError("Original V4 CUDA paged cache requires fp8_e4m3; BF16 is not this layout")
        c = model.config
        layers = c["num_hidden_layers"] - layer_start if layers is None else layers
        if (type(layers) is not int or layers <= 0 or type(layer_start) is not int or
                layer_start < 0 or layer_start + layers > c["num_hidden_layers"]):
            raise ValueError("Invalid V4 layer interval")
        ratios = c["compress_ratios"][layer_start:layer_start + layers]
        # A rate is an amortized descriptor, not the byte count of a short request.
        # SGLang stores shared KV and index keys replicated on every TP rank.
        parts = {"swa_paged_kv": layers * page_bytes(256) // 256,
                 "c4_compressed_kv": math.ceil(ratios.count(4) * Fraction(page_bytes(64), 256)),
                 "c4_index_keys_and_scales": ratios.count(4) * 33,
                 "c128_compressed_kv": math.ceil(ratios.count(128) * Fraction(page_bytes(2), 256))}
        return {**parts, "total": sum(parts.values())}

    def token_cap(self, tokens):
        # Native ratio sizing rejects pools <= sliding_window + page_size (384).
        return max(512, tokens)

    def activation_width(self, model):
        return model.config["hidden_size"] * model.config["hc_mult"]

    def allocate(self, model, p, w, r, kv, start, count, *, tokens=None):
        parts = self.per_token(model, tp=p.tp, layers=count, layer_start=start,
                              kv_quantization=kv, runtime=r)
        tokens = w.concurrency_per_dp * ceildiv(w.context_tokens, 256) * 256 if tokens is None else tokens
        full = self.token_cap(tokens)
        ratios = model.config["compress_ratios"][start:start + count]
        c4, c128 = ratios.count(4), ratios.count(128)
        # global_page_size=256 reserves 256 slots even in a compressed pool;
        # the physical pages of those pools are 64 and 2 slots respectively.
        swa_pages = ceildiv(full + 256, 256)
        c4_pages = ceildiv(full // 4 + 256, 64)
        c128_pages = ceildiv(full // 128 + 256, 2)
        cache = {"swa_paged_kv": swa_pages * page_bytes(256) * count,
                 "c4_compressed_kv": c4_pages * page_bytes(64) * c4,
                 "c4_index_keys_and_scales": c4_pages * 64 * 132 * c4,
                 "c128_compressed_kv": c128_pages * page_bytes(2) * c128}
        cache = {key: math.ceil(value * r.kv_pool_overhead_ratio) for key, value in cache.items()}
        # Offline FP32 (kv, score) states, C4 overlap=2, plus null/write rings.
        c4_rows = ceildiv(full // 256 * 8 + 8 + 1, 4) * 4
        c128_rows = ceildiv((w.concurrency_per_dp + 1) * 128 + 128 + 1, 128) * 128
        runtime = {"c4_compressor_state": c4_rows * 4 * 512 * 4 * c4,
                   "c4_indexer_compressor_state": c4_rows * 4 * 128 * 4 * c4,
                   "c128_compressor_state": c128_rows * 2 * 512 * 4 * c128,
                   # Analytical reserve for allocator maps/free-slot tables.
                   # This is an assumption, not a measured/proven upper bound.
                   "v4_allocator_metadata_envelope": (full + 256) * 32}
        return StateAllocation(parts, full, cache, runtime)
