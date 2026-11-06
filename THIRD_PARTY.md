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
