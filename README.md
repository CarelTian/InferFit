# InferFit

**Offline GPU memory preflight and parallelism planning for LLM inference.**

InferFit estimates per-rank weight residency, KV cache and recurrent state, loading
and serving peaks, and memory headroom. It checks supported runtime constraints
and searches tensor, expert, data, and pipeline parallel configurations without
loading model weights or installing an inference framework.

**Version:** 0.1.0 · **Python:** 3.11+ · **License:** MIT

## Features

- Per-rank memory breakdowns, bottleneck ranks, token-cache geometry, and failure reasons.
- Separate capacity and backend compatibility assessments.
- Finite configuration search with explicit resource limits and rejection diagnostics.
- Contiguous pipeline partition optimization that minimizes the largest per-rank memory peak.
- Prefill/decode pool capacity checks, including handoff overlap and transfer-buffer reserves.
- Offline model snapshots, external JSON profiles, a Python SDK, and a JSON CLI.
- No third-party runtime dependencies and no runtime network access.

A passing result is conditional on the selected memory layout and runtime reserves.
It is not a successful deployment test, a throughput prediction, or an SLA guarantee.
Results retain `deployment_verified=false`.

## Installation

From a checkout of this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -c 'import inferfit; print(inferfit.__version__)'
inferfit --help
```

For a regular installation, use `python -m pip install .` or install a locally built
wheel. Editable installation is recommended for development so the CLI loads the
current checkout rather than an older copy in `site-packages`.

## Quick start

Check GLM-5.1 with a fixed 16-GPU configuration:

```python
from inferfit import (
    Parallelism, RuntimeProfile, Workload, check, load_glm51, load_hardware,
)

model = load_glm51()
options = dict(
    hardware=load_hardware("h800-80gb", available_nodes=2),
    runtime=RuntimeProfile.sglang(),
    workload=Workload(context_tokens=8192, concurrency_per_dp=1),
    weight_quantization="fp8",
    kv_quantization="bf16",
)

result = check(
    model,
    parallelism=Parallelism(tp=8, ep=8, pp=2, num_gpus=16),
    **options,
)
print(result.status)
print(result.summary.get("minimum_headroom_bytes"))
```

Search explicitly allowed configurations:

```python
from inferfit import search_plans

plans = search_plans(
    model,
    tp_values=[1, 2, 4, 8],
    dp_values=[1],
    pp_values=range(1, 9),
    max_gpus=32,
    policy="static",
    **options,
)
print(plans.minimum_num_gpus)
print(plans.rejection_reasons)
```

These examples assume a connected cluster with sufficient available nodes. They
perform analytical sizing and static checks; they do not launch distributed workers.

## API overview

| API | Purpose |
| --- | --- |
| `check()` | Strict preflight for one configuration under a maintained backend contract. |
| `analyze()` | Separate capacity and compatibility results, with conditional capacity fallback. |
| `search()` | Search configurations that pass strict checks. |
| `search_plans()` | Search with `static` or `capacity` filtering and rejection diagnostics. |
| `optimize_pipeline()` | Optimize contiguous PP layer partitions for fixed parallel degrees. |
| `check_pd()` | Check specified prefill and decode pools and their handoff memory requirements. |
| `logical_weight_bytes()` | Count unpartitioned weight inventory bytes, including quantization scales. |
| `kv_per_token()` | Report token-cache geometry; exact allocation may include pages and fixed state. |
| `load_model()` / `from_config()` | Load a bundled model snapshot or a supported config. |
| `hardware_catalog()` / `backend_catalog()` | Inspect named hardware fixtures and maintained backend contracts. |

### Capacity versus compatibility

`analyze()` returns independent assessments:

- `capacity.status`: `pass`, `fail`, `unknown`, or `invalid`.
- `compatibility.status`: `supported`, `unsupported`, or `unknown`.
- `static_candidate`: true only when capacity passes and compatibility is supported.

An unknown hardware or runtime build can still receive a conditional capacity
estimate. Generic fallback covers ordinary GQA/GQA-MoE. Complex cache and recurrent
layouts require an explicit pinned reference layout through `MemoryAssumptions`.
Missing layout information produces `unknown` and `missing_data`, not a zero-byte
allocation. Known unsupported execution features remain unsupported after fallback.

`search_plans(policy="static")`, the default, requires both capacity and compatibility
to pass. `policy="capacity"` keeps capacity-passing candidates while retaining their
compatibility status. This distinction does not change deployment verification.

### Parallelism and ranking

- GPU count is `TP * DP * PP`; EP reuses the appropriate expert mesh.
- `concurrency_per_dp` is per attention replica, not total concurrency.
- `context_tokens` includes prompt tokens and generated or reserved output tokens.
- `stage_layers` specifies contiguous PP layer counts; `optimize_pp=True` searches them.
- Candidates are sorted by GPU count, DP, descending minimum headroom, PP, TP, then EP.
- Reported minimum GPU count applies only to the enumerated search space.
- Ranking optimizes capacity/resource use, not inference latency or throughput.

## Supported backends and hardware

Strict contracts are pinned to **SGLang 0.5.23** and **vLLM 0.19.0**. Family-specific
precision, hardware, topology, and state-layout restrictions still apply. Unknown
versions or uninspected builds cannot become supported by changing a profile name.

Use `RuntimeProfile.sglang()` for the maintained upstream SGLang contract.
`RuntimeProfile()` selects the pinned vLLM contract. Backend versions are independent
of the InferFit package version.

Named hardware fixtures cover NVIDIA H800 80GB, H20 96GB, H100 SXM 80GB, and A100
40GB/80GB. Published capacities use decimal GB and do not represent measured free
memory. Override `memory_bytes` with actual driver-visible capacity and separately
specify external usage and safety reserves. Arbitrary devices can be described with
`Hardware` for conditional capacity analysis; this does not establish execution support.

### Model coverage

| Model or family | Scope |
| --- | --- |
| Supported Llama/Qwen dense GQA and Qwen3-235B-A22B | Resident weight inventory, TP KV-head sharding or replication. |
| GLM-5.1 | MLA latent/RoPE/index state, TP/EP/PP and constrained attention DP. |
| GLM-5.3 | Shared-index ownership; currently DP=1, PP=1 and BF16 KV. |
| Qwen3.8-2.4T-A95B | GDN recurrent state and full-attention KV; BF16 contract. |
| Qwen3.8-Flash-Next BF16/FP8 | GDN/QSA/PLE/N-gram; text workload, DP=1, PP=1 and BF16 KV. |
| Original DeepSeek V4 Flash/Pro | Native mixed FP8/MXFP4, paged SWA/C4/C128 state; page=256, EP=TP and DP=1. |
| DeepSeek V4.1 | Recognized but rejected until its separate state and quantization contract is modeled. |

Native quantized snapshots use `weight_quantization="native"`; mixed dtypes, packed
shapes, exclusions, and scale tensors are counted explicitly rather than uniformly
requantized. Flash-Next currently assumes GPU-resident N-gram weights and a conservative
replicated vision-weight budget. Offloaded tables and image/video activations are not modeled.

## CLI

```bash
inferfit models
inferfit hardware
inferfit backends
inferfit check examples/h800-glm51-sglang-check.json
inferfit search examples/h800-glm51-sglang-search.json --output /tmp/search-result.json
inferfit check examples/flagships/qwen38-flash-next-fp8-check.json
inferfit pd-check examples/h100-h20-qwen-pd.json
```

`analyze` and `search-plans` accept the same model/hardware/runtime references, with
additional memory assumptions and search policy options. For example:

```bash
inferfit analyze examples/h800-glm51-sglang-check.json
inferfit search-plans examples/h800-glm51-sglang-search.json
```

`check`, `analyze`, and `pd-check` return exit code 0 when their memory/capacity check
passes; search commands return 0 when candidates exist. Otherwise they return 2.
Input errors return 1. For `analyze`, exit code 0 alone does not imply backend support;
inspect the compatibility assessment separately.

## Limits and interpretation

- Memory budget is `min(total_bytes * utilization, total_bytes - external_used) - safety_margin`.
- Headroom is budget minus the greater of loading and serving peaks. It is not device
  free memory after a runtime preallocates its cache pool.
- Runtime, activation, allocator, and communication reserves are assumed unless
  otherwise explicitly recorded; no live-device calibration is included.
- A token cap is a configured upper bound, not a measured physical pool capacity.
- PD checks currently require matching engine versions, precision, and parallel layouts.
  Cross-layout reshards, complex-state handoff, and automatic P:D ratio search are not implemented.
- No performance Roofline, queueing/SLA simulation, speculative decoding, or offload
  execution model is provided.

## Development and validation

```bash
python -m pip install -e '.[test]'
python -m pytest
python -m flake8 src tests examples tools --select=E9,F63,F7,F82
python -m flake8 src --select=F401,F811,F841
python examples/run_extended_matrix.py
python tools/benchmark_partition.py --iterations 5
```

The repository keeps scenario inputs and independent tests. Generate result JSON
with `--output`; regression scenarios can also be saved with
`python examples/run_extended_matrix.py --output /tmp/matrix-results.json`.

Build distributions with:

```bash
python -m pip install build
python -m build
```

The package version is defined once in `src/inferfit/_version.py`. Setuptools reads
that value for wheel/sdist metadata, and the SDK exports it as `inferfit.__version__`.
The fixed repository version is **0.1.0**.

## Documentation and contributing

- [Architecture](ARCHITECTURE.md) — detailed implementation notes (Chinese).
- [Profile contracts](PROFILE_CONTRACT.md) — input schemas and extension rules (Chinese).
- [Examples](examples/) — SDK usage and offline CLI scenarios.
- [Data and implementation references](THIRD_PARTY.md) — provenance and third-party notices.

Contributions should state the concrete problem, scope, and validation performed.
For model or backend changes, provide pinned sources, explicit layout assumptions,
independent byte/shape checks, and tests for unsupported configurations. Keep runtime
operation offline and do not commit model weights, credentials, or generated results.
Report bugs and propose changes through [GitHub issues](https://github.com/CarelTian/InferFit/issues)
and [pull requests](https://github.com/CarelTian/InferFit/pulls).

## License

InferFit is distributed under the [MIT License](LICENSE). Model metadata and upstream
implementation references are documented in [THIRD_PARTY.md](THIRD_PARTY.md).
