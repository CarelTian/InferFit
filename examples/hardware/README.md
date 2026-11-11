# Hardware examples

These scenarios use the pinned upstream SGLang memory contract. Run commands from
the repository root after installing InferFit:

```bash
inferfit hardware
inferfit check examples/hardware/h200-141gb-check.json
inferfit check examples/hardware/b200-180gb-check.json
inferfit check examples/hardware/gb200-186gb-check.json
inferfit check examples/hardware/b300-270gb-check.json
inferfit check examples/hardware/b300-288gb-check.json
inferfit check examples/hardware/gb300-279gb-check.json
inferfit check examples/hardware/gb300-288gb-check.json
inferfit check examples/hardware/rtx5090-32gb-check.json
inferfit search examples/hardware/b200-180gb-search.json
inferfit pd-check examples/hardware/h200-b200-pd.json
```

The data-center check examples place Qwen3-235B-A22B BF16 on four GPUs with
TP=EP=4, DP=PP=1. The RTX 5090 example uses an illustrative dense GQA 8B
configuration with BF16 weights and FP8 KV on one GPU. This config is a shape
example, not a collected checkpoint or a hardware benchmark.

The B200 search varies TP and PP within eight GPUs. The P/D example sizes two
independent eight-GPU pools, H200 for prefill and B200 for decode, with the same
TP/EP layout and BF16 weight/cache format. Connector execution is unverified.

All capacities are per GPU in decimal GB. GB200/GB300 do not include Grace CPU
memory, combine both GPUs in a Superchip, or treat an NVL72 rack as one worker.
The example `gpus_per_node` values are explicit placement assumptions; adapt them
to your actual installation. No NVLink, GPU P2P or communication timing is inferred
from a catalog key.

NVIDIA sources publish different B300/GB300 capacity specifications. The 270/279 GB
entries follow the current Blackwell Ultra datasheet; the 288 GB entries follow
the DGX B300 guide / GB300 launch article. Inspect `inferfit hardware` for source
URLs and notes. Override `memory_bytes` with actual driver-visible bytes before
making a deployment decision.

Model-family restrictions still apply. H200 shares the maintained SM90 contracts;
adding Blackwell fixtures does not expand the SM90-only Flash-Next, MLA/DSA or
native V4 kernel contracts. A capacity result is an analytical estimate, not a GPU
execution test. Runtime reserves are assumed, and no latency is predicted.
