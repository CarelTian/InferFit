# Data and implementation references

The package does not import or vendor vLLM/SGLang/PyTorch implementation code.

Bundled GLM-5.1 configuration and tensor metadata originate from Z.ai's MIT-licensed
model repositories. The exact revisions, URLs, header hashes, shapes, dtypes and
checkpoint tensor counts are retained in `inferfit/data/*.json`.
No tensor payloads / model weights are distributed.

The analytical formulas and static capability rules were checked against upstream
vLLM v0.19.0 and SGLang v0.5.10, both Apache-2.0. Source hashes and the
pinned SGLang commit are recorded in `data/sources.json` and `data/sglang-contract.json`.
The independent implementation here is a capacity estimator, not either inference engine itself.

References:

- https://huggingface.co/zai-org/GLM-5.1
- https://huggingface.co/zai-org/GLM-5.1-FP8
- https://github.com/vllm-project/vllm/tree/v0.19.0
- https://github.com/sgl-project/sglang/tree/1519acf37c23f2189adb93f57ca9cd2db1bebf18
- https://docs.nvidia.com/ai-enterprise/release-7/latest/infra-software/vgpu/reference/hopper.html

Additional pinned configuration metadata: Qwen/Qwen3-235B-A22B and
deepseek-ai/DeepSeek-V4.1-Flash. Repository URLs, revisions, retrieval dates and
parameter metadata are recorded in data/*-source.json. DeepSeek is an unsupported
architecture fixture, not a supported estimator. Hardware reference URLs and the
ZW810 illustrative-capacity policy are recorded in data/hardware-sources.json.

## 0.6.0 flagship snapshots

Frozen configuration snapshots for Qwen3.8-2.4T-A95B, GLM-5.3 and DeepSeek-V4.1-Flash-1017 are identified in `inferfit/data/flagships.json` by source URL, content SHA-256 and retrieval date. Model-specific license terms remain applicable; GLM-5.3 is not covered by the earlier GLM-5.1 MIT statement. See each original model repository for its license.

SGLang v0.5.23 implementation references are Apache-2.0 upstream sources. Links are retained in README.md; InferFit does not distribute or import upstream implementation snapshots.

## Qwen3.8-Flash-Next (0.7.0)

Configuration and tensor header metadata originate from `Qwen/Qwen3.8-Flash-Next`, revision `812d96e10a0dd82b69d749e68693698cf6c36daa`, under the model repository's Qwen Community License 1.0. See https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/812d96e10a0dd82b69d749e68693698cf6c36daa/LICENSE for the original terms. No tensor payloads are distributed. SGLang 0.5.23 implementation references remain Apache-2.0; their URLs and source digests are retained in `data/qwen38-flash-next-metadata.json`. Upstream implementation code is not distributed or imported.
