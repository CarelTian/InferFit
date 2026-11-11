# Data and implementation references

The package does not import or vendor vLLM/SGLang/PyTorch implementation code.

Bundled GLM-5.1 configuration and tensor metadata originate from Z.ai's MIT-licensed
model repositories. The exact revisions, URLs, header hashes, shapes, dtypes and
checkpoint tensor counts are retained in `inferfit/data/*.json`.
No tensor payloads / model weights are distributed.

The analytical formulas and static capability rules were checked against upstream
vLLM v0.19.0 and SGLang v0.5.23, both Apache-2.0. Source hashes and the
pinned SGLang revision are recorded in `data/sources.json` and `data/sglang-contract.json`.
The independent implementation here is a capacity estimator, not either inference engine itself.

References:

- https://huggingface.co/zai-org/GLM-5.1
- https://huggingface.co/zai-org/GLM-5.1-FP8
- https://github.com/vllm-project/vllm/tree/v0.19.0
- https://github.com/sgl-project/sglang/tree/v0.5.23
- https://docs.nvidia.com/ai-enterprise/release-7/latest/infra-software/vgpu/reference/hopper.html

Additional pinned configuration metadata: Qwen/Qwen3-235B-A22B and
deepseek-ai/DeepSeek-V4.1-Flash. Repository URLs, revisions, retrieval dates and
parameter metadata are recorded in data/*-source.json. This V4.1 fixture is an unsupported
architecture, not a supported estimator. NVIDIA hardware reference URLs and
the nominal-capacity policy are recorded in data/hardware-sources.json.

## Flagship snapshots

Frozen configuration snapshots for Qwen3.8-2.4T-A95B, GLM-5.3 and DeepSeek-V4.1-Flash-1017 are identified in `inferfit/data/flagships.json` by source URL, content SHA-256 and retrieval date. Model-specific license terms remain applicable; GLM-5.3 is not covered by the earlier GLM-5.1 MIT statement. See each original model repository for its license.

SGLang v0.5.23 implementation references are Apache-2.0 upstream sources. Links are retained in README.md; InferFit does not distribute or import upstream implementation snapshots.

## Qwen3.8-Flash-Next

Configuration and tensor header metadata originate from `Qwen/Qwen3.8-Flash-Next`, revision `812d96e10a0dd82b69d749e68693698cf6c36daa`, under the model repository's Qwen Community License 1.0. See https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/812d96e10a0dd82b69d749e68693698cf6c36daa/LICENSE for the original terms. No tensor payloads are distributed. SGLang 0.5.23 implementation references remain Apache-2.0; their URLs and source digests are retained in `data/qwen38-flash-next-metadata.json`. Upstream implementation code is not distributed or imported.

## Native quantized models

New configuration and tensor-header metadata are pinned in `data/flagships.json`:
Qwen/Qwen3.8-Flash-Next-FP8, deepseek-ai/DeepSeek-V4-Flash and
DeepSeek-V4-Pro. Original V4 Flash/Pro are supported analytical profiles;
V4.1 remains a separate unsupported fixture. No tensor payloads are distributed.

The original model license terms are available in the pinned repositories:

- https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/tree/7fb48d708ca6cd8eeed8f4db0433be18af96984f
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/60d8d70770c6776ff598c94bb586a859a38244f1
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/b5968e9190ef611bbf34a7229255be88a0e937c1

SGLang v0.5.23 formulas were independently modeled from its Apache-2.0
implementation. Source URLs and SHA-256 digests, checkpoint header digests,
encoded dtypes, and exact checkpoint byte totals are retained in the new
`*-metadata.json` files. Upstream source code is not vendored.
