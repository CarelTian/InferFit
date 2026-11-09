# InferFit

**Engine-aware memory preflight and parallelism planning for LLM inference.**

面向大模型推理的显存预检与并行部署规划工具：估算每卡显存、KV Cache 和剩余容量，校验引擎约束，并搜索满足条件的部署组合。

独立、离线、无运行时第三方依赖的 Python 组件。0.8.0 保留 **vLLM 0.19.0 / SGLang 0.5.10** 适配，并新增面向旗舰结构的 **SGLang 0.5.23** 有限支持。其他版本返回 unsupported，避免套用未经核实的规则。

1. 检查指定模型、硬件、TP/DP/EP/PP 和负载的逐卡显存预算。
2. 输出权重、KV、运行开销、峰值、余量、单 token KV 和缓存容量。
3. 枚举可行组合，并用动态规划优化 PP 分层，输出引擎参数及分层环境变量。

**`pass` 表示在显式假设下通过显存和静态能力检查。运行时开销尚未在 H800 实测，输出始终包含 `deployment_verified=false`。候选组合需要真实环境启动验证，不能当作吞吐/SLO 或部署成功保证。**

## 0.8.0：原生 FP8 / FP4 与独立引擎适配器

截至 2026-11-09，核实 [上游正式版](https://github.com/sgl-project/sglang/releases/tag/v0.5.23)
与 [PyPI](https://pypi.org/project/sglang/0.5.23/) 均为 **0.5.23**。
新模型入口使用 `RuntimeProfile.sglang_current()`；现有 `sglang_flagship()` 仍指向
同一版本。`RuntimeProfile.sglang_legacy()` 明确选择旧的 0.5.10；`sglang()` 作为
旧 SDK 兼容入口保留原行为。版本常量统一在 `engine_versions.py` 中区分
`SGLANG_CURRENT_VERSION` / `SGLANG_LEGACY_VERSION`。旧 `SGLANG_VERSION` 仅为
兼容别名，内部不再使用；各版本的模型适用范围仍分别检查，不能直接换号套用。

新增以下三个冻结快照，均可离线 `check/search`。原生量化使用
`weight_quantization="native"`，也接受 `fp8` 别名；实际按 checkpoint 中的
混合 dtype 与独立 scale 计费，不能理解为全部权重都是 FP8。

| 模型 alias | 基础权重字节数（不含 MTP） | 新支持范围 |
|---|---:|---|
| `qwen3.8-flash-next-fp8` | 182,804,206,074 | FP8 routed experts / N-gram + BF16 exclusions；BF16 KV；TP/EP 搜索，PP=1 |
| `deepseek-v4-flash` | 156,015,698,140 | FP8 主干 + 原生 MXFP4 专家；FP8 V4 分页缓存；TP/EP/PP 搜索 |
| `deepseek-v4-pro` | 850,748,205,004 | 同上，使用独立 Pro 配置和 tensor headers |

这三个新合同固定 **SGLang 0.5.23、Hopper SM90、DP=1、EP=TP**，关闭 radix、
CUDA Graph 和 MTP，权重完整驻留 GPU。Flash-Next 仍仅覆盖文本负载，视觉权重
采用复制上界。V4 固定 FlashMLA attention、FlashInfer CUTLASS MXFP4 W4A16、
page=256、SWA/full=1 和离线 FP32 compressor state；这里显式覆盖 SGLang
默认 SWA 比率。其他 GPU、其他 kernel/压缩布局、FP4→FP8 展开、offload 和新模型
的 PD 交接均需新的合同。**V4.1 仍单独标记 unsupported。**

V4 缓存分别报告 SWA、C4 KV、C4 index、C128 KV，另在 runtime 中列出 C4/C128
压缩器状态和 RoPE 表。单 token 数值是分页摊销速率（向上取整），不能直接乘
context 代替实际池分配。预检和容量反解使用完整池、保留页及 ring 尺寸。
FP4 专家维度已经按每字节两个数打包；expert scales 保留 E8M0。
普通 dense scale 按保守上界计费，辅助分配和工作区采用显式分析预留，运行开销仍未实测。

H800 每卡 80 decimal GB、利用率 0.9、安全余量 1 GiB、context=8192、并发=1、
batch=2048，在示例枚举范围内：

| 模型 | 搜索最少卡数 | 一个通过组合（TP/DP/EP/PP） | 最小预算余量 |
|---|---:|---|---:|
| Flash-Next FP8 | 4 | 4/1/4/1 | 14.871 GiB |
| V4 Flash | 3 | 1/1/1/3，层数 14,15,14 | 8.354 GiB |
| V4 Pro | 16 | 8/1/8/2，层数 31,30 | 6.009 GiB |

这是有限搜索的显存候选，不是实测最小部署或吞吐建议。V4 Flash 还可选 4 卡
TP4/EP4/PP1，余量约 20.979 GiB；PP3 虽省卡，也有流水线通信与性能取舍。

```python
from inferfit import load_model, search, RuntimeProfile, Workload

result = search(
    load_model("deepseek-v4-pro"),
    runtime=RuntimeProfile.sglang_current(),
    workload=Workload(context_tokens=8192, block_size=256),
    weight_quantization="native", kv_quantization="fp8",
    tp_values=[1, 2, 4, 8], dp_values=[1], pp_values=range(1, 17),
    max_gpus=32,
)
print(result.minimum_num_gpus)
print(result.candidates[0].summary["engine_args"])
print(result.candidates[0].summary["engine_env"])
```

```bash
inferfit check examples/flagships/qwen38-flash-next-fp8-check.json
inferfit check examples/flagships/deepseek-v4-flash-check.json
inferfit search examples/flagships/deepseek-v4-pro-search.json
python examples/native_models.py
```

原有 SGLang→vLLM 继承已移除：两个引擎分别实现中立 `EngineAdapter`。
模型转换、存储和状态规则分别放在 `model_families/`、`weights.py`、`states/`；
旧 SDK 导入路径保持兼容。新增模型所需材料与修改位置见 [ARCHITECTURE.md](ARCHITECTURE.md)。

本次核对 131/46/64 个分片、152089/69187/145116 个 tensor 的全部 headers，
其总字节数与各自官方索引一致，没有下载 tensor payload。
来源：[Qwen FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)、
[V4 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)、
[V4 Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)、
[SGLang 池配置](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/model_executor/pool_configurator.py)、
[原生 MXFP4 后端](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py)。

## 0.7.0：Qwen3.8-Flash-Next（BF16 快照）

新增 `load_model("qwen3.8-flash-next")`，也可使用完整模型 ID `Qwen/Qwen3.8-Flash-Next`。该模型实际为 **Qwen4-Exp** 架构，不能直接套用 Qwen3.8-2.4T 的 GQA+GDN 清单。

- 冻结官方 revision `812d96e10a0dd82b69d749e68693698cf6c36daa` 的完整嵌套 text/vision 配置；通过 HTTP Range 收集 **131 个分片、1658 个张量**的文件头，没有下载权重正文。全 checkpoint 为 **359,999,963,128 bytes**，与官方索引一致；剔除未启用的 MTP 后为 **354,785,661,432 bytes**。SGLang 另生成一个 BF16 标量，运行时还将 GDN A_log 升为 FP32，分别计入预算。
- 权重清单包含 Gated Residual、约 **102.4 GB BF16 N-gram 表**、视觉编码器及 I64 缓冲。128 个磁盘 N-gram shards 合并后再按运行时 TP 切分，不能将磁盘分片数当成 GPU 数。
- 48 层包含 **36 层 GDN、12 层 QSA**。QSA 保留全部历史主 KV，并增加每 4 token 一份的压缩 index key；top-k=2048 不代表只保存 2048 token。BF16 逻辑状态为 **25,344 bytes/token**；TP=2/4/8 时每卡为 **13,056 bytes/token**，其中 768 bytes 的压缩索引在 TP 间复制。
- 固定状态单独计入：GDN recurrent/conv、PLE 的 9 步 short-conv、N-gram token history、QSA pending-key ring 和共享位置。它们按请求槽位计费，不乘上下文长度；QSA 工作区与四路 residual 的激活使用显式分析预留。

本版支持 **SGLang 0.5.23、Hopper SM90、BF16 权重/KV、DP=1、PP=1、文本输入、N-gram 完全驻留 GPU、关闭 radix/CUDA Graph/MTP**。可检查 TP/EP 并搜索组合。视觉编码器权重采用每卡完整复制的保守预算；**未模拟图像/视频激活、NVFP4、CPU 卸载、CP 或 PD 交接**；原生 FP8 请使用上面的独立快照。未知变体和不支持的组合返回 unsupported，不静默沿用普通 Qwen 公式。

```python
from inferfit import load_model, RuntimeProfile, Hardware, search

result = search(
    load_model("qwen3.8-flash-next"),
    runtime=RuntimeProfile.sglang_flagship(),
    hardware=Hardware(memory_bytes=80_000_000_000),
    weight_quantization="bf16", kv_quantization="bf16",
    tp_values=[1, 2, 4, 8], dp_values=[1], pp_values=[1], max_gpus=8,
)
print(result.minimum_num_gpus)
```

```bash
inferfit check examples/flagships/qwen38-flash-next-check.json
inferfit search examples/flagships/qwen38-flash-next-search.json
```

示例在每卡 80 decimal GB、90% 利用率、额外 1 GiB 安全余量、8192 token、并发 1、batch 2048 的假设下，搜索到 8 卡候选。该结论仅对示例搜索空间成立，运行时预留仍未校准，**没有 GPU 启动验证**。

配置、tensor shapes/dtypes、文件头摘要和 SGLang 来源链接保存在 `src/inferfit/data/qwen38-flash-next-*.json`；不保留上游源码副本或本地报告目录。参考：[官方模型](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)、[SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next)、[QSA 缓存池](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/mem_cache/qsa_kv_pool.py)。

## 0.6.0：最新旗舰快照与 SGLang 适配

核实日期：2026-11-06。这里的“最新”是版本化快照，不会在运行时联网更新。

| 模型 | 本版容量支持 | SGLang 合同 |
|---|---|---|
| Qwen/Qwen3.8-2.4T-A95B | GQA + Gated DeltaNet + MoE；BF16 权重、BF16 KV；TP/EP/PP 搜索 | 0.5.23，DP=1，TP 整除线性 K/V heads，关闭 radix/图捕获/MTP |
| zai-org/GLM-5.3 | 共享索引 DSA；BF16/FP8-block 权重、BF16 KV；TP/EP 搜索 | 0.5.23，DP=1、PP=1 |
| deepseek-ai/DeepSeek-V4.1-Flash-1017 | **只识别与诊断，完整容量适配尚未完成**；check/search 返回 unsupported，不能产生部署候选 | 原生 32×32 UE8M0 FP8、FP4 experts、Engram 和共享压缩缓存尚未纳入完整预算 |

GLM-5.3 的 78 层只拥有 21 份 indexer，不能按 78 份索引缓存/权重计算。Qwen 的 92 层包含 23 层 full attention、69 层 GDN：完整逻辑 BF16 KV 为 **94,208 bytes/token**；GDN 的 BF16 convolution + FP32 recurrent state 为 **587,292,672 bytes/request**（TP=1）。后者按请求槽位计入 `runtime_breakdown.recurrent_state_pool`，不随 context 长度重复增长。当前池预算为 `(concurrency_per_dp + 1)` 个槽位，输出 `max_mamba_cache_size` 和状态 dtype 环境变量。

```python
from inferfit import load_model, RuntimeProfile, Hardware, Parallelism, check

model = load_model("glm-5.3")
result = check(
    model,
    runtime=RuntimeProfile.sglang_flagship(),
    hardware=Hardware(memory_bytes=80_000_000_000, allow_cross_node_tp=True),
    parallelism=Parallelism(tp=16, dp=1, ep=16, pp=1, num_gpus=16),
    weight_quantization="fp8", kv_quantization="bf16",
)
print(result.status, result.summary.get("minimum_headroom_bytes"))
```

```bash
inferfit models
inferfit check examples/flagships/glm53-check.json
inferfit search examples/flagships/qwen38-search.json
# 此例预期 unsupported，退出码为 2：
inferfit check examples/flagships/deepseek41-check.json
```

`flagship_catalog()` 返回每个模型的支持状态、数据摘要及未完成项。`load_model()` 加载冻结配置；旧模型仍通过原有 loader 加载。Qwen 的 revision 来自 HF commit；GLM/DeepSeek 因本次直接元数据访问受限，明确使用按日期和内容摘要标识的配置快照，不冒充 Hugging Face commit。BF16 表示目标驻留精度；FP8 checkpoint 不能仅修改 dtype 就变成 BF16 checkpoint，实际启动需匹配权重文件/转换流程。新结构的 PD 状态交接尚未验证，明确拒绝。

上述权重为结构推导，未逐 tensor 对照新 checkpoint headers；运行预留仍是分析假设，**没有新模型的 GPU 启动验证**。不能将这一支持矩阵解读为三家最新模型均已完成适配。

官方配置：[Qwen3.8](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/main/config.json)、[GLM-5.3](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json)、[DeepSeek V4.1](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/config.json)。SGLang 依据：[Qwen 实现](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/models/qwen3_5.py)、[线性状态形状](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/configs/mamba_utils.py)、[DSA indexer 构建](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/models/deepseek_v2.py)、[DeepSeek 共享结构](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/models/deepseek_v4.py)。

## 0.4.0 数据接口

新增版本化模型/硬件/运行时 profile、语义张量角色和可注册的 config 转换器。新模型或同平台硬件可由数据文件接入，预检、PP 优化、搜索和 PD 共用原有核心。详见 [输入合同与扩展指南](PROFILE_CONTRACT.md)，可执行示例在 `examples/profiles/scenario.json`。

```bash
inferfit check examples/profiles/scenario.json
```

## 0.3.0 扩展模型、硬件与 PD 用例

新增 **Qwen/Qwen3-235B-A22B**（原始模型，不混用 Instruct-2507）的 GQA+MoE 适配器，固定 revision `8efa61729e24bd65b1d152b5ab5409052aa80e65`。结构推导的 BF16 权重为 **470,187,269,120 bytes**，与官方 235,093,634,560 个 BF16 参数完全一致。完整模型 KV 为 **192,512 bytes/token**；TP4/TP8 每卡都是 **48,128 bytes/token**，TP8 会复制 KV heads。FP8 为目标存储布局估算，未宣称已与该模型的 FP8 checkpoint 文件逐项对齐。

```python
from inferfit import *
model = load_qwen3_235b()
hardware = load_hardware("a100-80gb")
result = check(model, hardware=hardware, weight_quantization="bf16",
               parallelism=Parallelism(tp=8, dp=1, ep=8, pp=1, num_gpus=8))
```

JSON 可使用 `"model":"qwen3-235b-a22b"`。硬件 catalog 区分 `h20-96gb / h100-sxm-80gb / a100-80gb / a100-40gb / h800-80gb`，均为名义 decimal GB；实际容量应覆盖 `memory_bytes`。所有结果仍使用未实测的运行开销预留。A100 的 BF16 Qwen 用例可通过，FP8 权重/KV 在当前固定适配器下返回 unsupported。

**边界覆盖不等于已经适配：**

- `ppu-zw810` 要求调用方提供实际容量，例如 `load_hardware("ppu-zw810", memory_bytes=...)`，不假设默认 SKU 显存，也不伪造 CUDA compute capability。当前两个 CUDA runtime 对它返回 unsupported。负例中的 96 GiB 是示意输入，不是产品规格。
- DeepSeek V4.1 Flash 的官方配置固定到 `314113f8a7dadb388215396ba5fbee2c610a04d6` 并随包保存；因专有压缩/共享状态、Engram、视觉及混合量化尚无估算器，API 抛出 `UnsupportedModelError`，CLI 输出 `status=unsupported`、退出码 2。它不会套用 GLM 的单 token 公式。

可离线重跑 **54 个场景**，涵盖两种引擎、各硬件容量边界、量化拒绝、PD 通过/失败和布局拒绝：

```bash
python3 examples/run_extended_matrix.py
inferfit pd-check examples/h100-h20-qwen-pd.json
```

场景输入与期望状态在 `examples/extended/scenarios.json`，结果在 `examples/extended/results.json`。独立单元测试使用官方参数量、手算 KV、容量不足及状态保留等断言；场景快照用于复现，而非实机验证。

### PD 分离容量检查

```python
from inferfit import *
plan = Parallelism(tp=8, dp=1, ep=8, pp=1, num_gpus=8)
pool_p = PDPool(tuple(f"p-node/gpu{i}" for i in range(8)),
                 hardware=load_hardware("h100-sxm-80gb"), parallelism=plan)
pool_d = PDPool(tuple(f"d-node/gpu{i}" for i in range(8)),
                 hardware=load_hardware("h20-96gb"), parallelism=plan)
result = check_pd(load_qwen3_235b(), prefill=pool_p, decode=pool_d,
                  demand=PDDemand(prompt_tokens=8192, output_tokens=1024,
                                  inflight_handoffs_per_dp=2))
```

P/D 两池分别保存完整模型副本、分别检查逐卡预算，不能把两池的总显存凑起来装一份权重。物理卡 ID 必须完整、唯一且两池不重叠；ID 是调用方提供的规范身份，不替代实际机器库存查询或启动器的拓扑验证。

状态生命周期口径：

- P 在 ACK 前保留源状态，按 `(prefill_concurrency + inflight_handoffs) * prompt_tokens` 预留。
- D 按 `(decode_concurrency + inflight_handoffs) * (prompt_tokens + output_tokens)` 保守预留，包含目的侧已接收但尚未交接完成的请求。
- 每请求分别按页取整，再加引擎 dummy page；可通过 `transfer_buffer_bytes_per_gpu` 指定每卡 GPU staging/connector 缓冲，不把主机缓冲混进显存。
- `handoff_logical_payload_bytes_per_request` 是逻辑状态量，未乘 TP 复制份数，也不是 RDMA 实际线上字节数。

本版 PD 是**分析式容量预检**。只接受相同引擎/版本、相同精度及相同并行和分层布局；重分片、KV 转码、跨引擎交接明确拒绝。池的 `engine_args` 仅作为单池容量提示，不能直接当作 PD 启动命令。输出固定 `connector_verified=false`，未模拟带宽、延迟、连接器实现和 PD 吞吐；目前没有 PD 自动组合搜索 API。

## 安装与使用

Python >=3.11；本次测试环境为 Python 3.15.2。wheel 安装不需要网络或 PyTorch。

```bash
python3 -m pip install dist/inferfit-0.8.0-py3-none-any.whl
inferfit check examples/h800-glm51-check.json
inferfit search examples/h800-glm51-search.json --output search-result.json
```

检查命令退出码：`0` 为 pass，`2` 为显存失败/无可用候选/不支持/组合非法，`1` 为输入文件或字段错误。JSON 的 `status` 提供更细的区分。

Python API：

```python
from inferfit import (
    Hardware, Parallelism, Workload, check, search, load_glm51,
    from_config, kv_per_token, optimize_pipeline,
)

model = load_glm51()  # 固定 revision 的官方 config，包内自带
# 也可使用用户的本地配置：
# model = from_config("/path/to/config.json", model_id="my-model")

hardware = Hardware(memory_bytes=80_000_000_000)
workload = Workload(context_tokens=8192, concurrency_per_dp=1,
                    max_batch_tokens=2048)
result = check(
    model, hardware=hardware, workload=workload,
    parallelism=Parallelism(tp=8, dp=1, ep=8, pp=2, num_gpus=16),
    weight_quantization="fp8_block", kv_quantization="bf16",
)
print(result.status, result.summary["minimum_headroom_bytes"])
print(result.ranks[0].kv_bytes_per_token)

# 单独计算单 token 状态；默认是整个逻辑模型、TP=1。
print(kv_per_token(model, kv_quantization="fp8"))

# 对固定并行度，寻找使最紧张 GPU 峰值最小的连续 PP 分层。
optimized = optimize_pipeline(
    model, hardware=hardware, workload=workload,
    parallelism=Parallelism(tp=4, dp=1, ep=4, pp=3, num_gpus=12),
)
print(optimized.summary["stage_layers"])  # [28, 25, 25]

found = search(
    model, hardware=hardware, workload=workload,
    weight_quantization="fp8_block", kv_quantization="bf16",
    tp_values=[1, 2, 4, 8], dp_values=[1, 2, 4],
    pp_values=range(1, 17), gpu_counts=[8, 12, 16, 24, 32],
    max_gpus=32, optimize_pp=True,
)
print(found.minimum_num_gpus)  # 12，只针对声明的搜索空间和负载
for candidate in found.candidates:
    print(candidate.parallelism, candidate.summary["minimum_headroom_bytes"])
```

`check` 严格使用传入的 PP 分层；没有 `stage_layers` 时按层数均分，不会静默改配置。`optimize_pipeline` 返回优化后的 `Parallelism.stage_layers`。`search` 默认启用分层优化。

## SGLang 使用方法与差异

SGLang adapter 固定到官方 `v0.5.10`、提交 `1519acf37c23f2189adb93f57ca9cd2db1bebf18`。这是 GLM-5.1 官方文档列出的支持版本起点，不表示覆盖后续所有 SGLang 版本。

```python
from inferfit import RuntimeProfile, Parallelism, check, search, load_glm51

runtime = RuntimeProfile.sglang()
result = check(
    load_glm51(), runtime=runtime,
    parallelism=Parallelism(tp=8, dp=1, ep=8, pp=2, num_gpus=16),
    weight_quantization="fp8_block", kv_quantization="fp8",
)
found = search(load_glm51(), runtime=runtime,
               gpu_counts=[8, 12, 16, 24, 32], max_gpus=32)

# DP attention：公共 TP=每个 attention 副本的宽度，GPU 计数口径保持一致。
dp_runtime = RuntimeProfile.sglang(sglang_dp_attention=True)
dp_result = check(
    load_glm51(), runtime=dp_runtime,
    parallelism=Parallelism(tp=8, dp=2, ep=16, pp=1, num_gpus=16),
)
# 对应 SGLang native tp_size=16, dp_size=2, ep_size=16。
```

JSON 中增加 `"runtime": {"engine": "sglang"}` 即可选择工厂默认值；也可显式写 `"version": "0.5.10"`。Python API 使用 `RuntimeProfile.sglang()`，避免直接沿用 vLLM dataclass 的默认版本和预留。

```bash
inferfit check examples/h800-glm51-sglang-check.json
inferfit search examples/h800-glm51-sglang-search.json
```

关键区别：

| 项目 | vLLM adapter | SGLang adapter |
|---|---|---|
| 普通 DP | 独立 attention 副本，EP 可跨 TP*DP | DP attention 关闭时为独立 worker；单次启动的多节点普通 DP 拒绝 |
| DP attention | 当前公共口径即按 attention 副本表达 | 用 `sglang_dp_attention=true`；native TP=公共 TP*DP，卡数仍为 TP*DP*PP |
| EP | 1 或 TP*DP | EP 整除 native TP，支持部分 EP；剩余 expert TP=native TP/EP |
| Dense/shared MLP | 按 vLLM 后端的分片/复制规则 | 固定 `moe_a2a_backend=none`、关闭 shared fusion，按 native TP 分片 |
| FP8 KV 参数 | `fp8_ds_mla` | `fp8_e4m3`；公共 `fp8` 自动映射，拒绝误用另一引擎专有名字 |
| MLA 投影 | FP8 产生额外 BF16 投影副本 | BF16/FP8 都计入连续 `w_kc/w_vc` 副本；索引 LayerNorm 按 FP32 |
| request 映射 | 分块 metadata 估计 | int32 `[并发, context+4]` |
| PP 默认分层 | 余数在前面分配 | 按源码余数分配到最后几层；优化模式输出显式分层 |
| PP 执行 | eager | 关闭整图、piecewise graph 和 overlap schedule |
| 节点放置 | 可用部分节点卡数 | native TP*PP world 要等额分配到节点 |

SGLang 固定 Hopper NSA 后端：BF16 使用 `flashmla_sparse` prefill 和 `fa3` decode；FP8 使用 `flashmla_auto` prefill 和 `flashmla_kv` decode。对这个固定后端组合，MLA/索引 payload 与上文单 token 数值一致；一致性来自分别核对源码，不是把任意引擎都视为相同布局。

`hardware.utilization` 始终是本组件的整体峰值预算。SGLang 的 `mem_fraction_static` 是内存池规划参数，不能单独替代整体峰值限制，因此输出中还会设置 **`max_total_tokens` 为目标请求向上按页取整后的驻留 token 数**。`configured_kv_pool_tokens_per_dp` 表示实际配置池大小；原有 `kv_capacity_tokens_per_dp` 表示预算容许的潜在容量，扩池需重新预检。额外 dummy page 已计入。

SGLang 的并发参数在 DP attention 下会除以 DP，所以输出的 `max_running_requests` 为输入并发乘 DP；`chunked_prefill_size` 也做相应反向转换。rank 在 DP attention 下为 `(pp,dp,tp)`；其他情况沿用 `(dp,pp,tp)`。

默认 SGLang runtime 仍为假设：固定激活预留 2 GiB，其余基础预留与 vLLM 的默认值一样；激活 envelope 在 DP attention 下按聚合 batch 放大，DSA prefill logits envelope 为 `max_batch_tokens * context_tokens * 4 bytes`。这些不是 GPU 实测；DeepEP、融合专家、CP、IndexCache、HiSparse、PD、MTP 和 Graph 模式不在该 profile 内。

示例同样使用 H800 名义 80GB、90% 峰值预算、1 GiB 安全余量、8192 token 上下文、并发/DP=1：

- 8 卡 FP8 权重仍 fail。
- 12 卡 TP4/PP3，优化分层 28/25/25，估算余量约 0.35 GiB；不跨 attention TP 时要求 **3 节点×4卡**，不能直接套用 vLLM 的 8+4 放置。
- 16 卡 TP8/PP2 可作为余量更大的候选；对应固定分层或优化分层的完整数值见 `examples/sglang-comparison.json`。

`nnodes` 和每个 rank 的节点编号随结果输出；多节点启动仍需按真实节点设置 `node_rank` 和 `dist_init_addr`。更正：SGLang 0.5.10 原生启动器要求 PP stages 与节点数形成整齐分组，TP4/PP3 不能按 2 节点×6卡启动，即使开启跨节点 TP。可设 `available_nodes` 限定真实可用节点数；未提供时返回的是所需节点数，不是现有库存验证。

官方源码合同（参数名、缓存常数、FP32 norm、页和节点约束）及 SHA256 存在 `data/sglang-contract.json`，不需要安装 SGLang/PyTorch 即可离线校验。

## H800 + GLM-5.1 的 vLLM 示例结果

场景：每卡名义容量 **80,000,000,000 bytes**，利用率上限 90%，额外安全余量 1 GiB；8192 token 总上下文，每个 DP attention 副本 1 个活跃请求；chunked prefill，batch token budget 2048；eager；无 offload/MTP。其他运行开销采用 `RuntimeProfile` 的显式假设。

**80GB 不是这里的 80GiB。** 应通过 `memory_bytes` 传入实际设备可见容量；组件没有把商品名当成实测值。

| 权重 / KV | TP / DP / EP / PP | GPU 数 | PP 层数 | 结果 | 最紧张卡余量 GiB |
|---|---|---:|---|---|---:|
| FP8 / BF16 | 8 / 1 / 8 / 1 | 8 | 78 | fail | -29.09 |
| FP8 / BF16 | 4 / 1 / 4 / 3 | 12 | 26,26,26 | fail | -0.91 |
| FP8 / BF16 | 4 / 1 / 4 / 3 | 12 | 28,25,25 | pass，估算候选 | 1.06 |
| FP8 / BF16 | 8 / 1 / 8 / 2 | 16 | 39,39 | pass，估算候选 | 13.92 |
| FP8 / FP8 MLA | 8 / 1 / 8 / 2 | 16 | 39,39 | pass，估算候选 | 13.92 |
| BF16 / BF16 | 8 / 1 / 8 / 2 | 16 | 39,39 | fail | -31.58 |
| BF16 / BF16 | 8 / 1 / 8 / 4 | 32 | 20,20,19,19 | pass，估算候选 | 13.18 |

12 卡候选使用两台至多 8 卡的节点，需要节点间网络和真实分布式启动配置。它的估算余量较小，实际运行开销偏差可能改变判断。搜索排序按 GPU 数、DP 数和余量进行；最少卡数不代表最佳吞吐或延迟。

完整逐卡输出和搜索结果位于 `examples/*-result.json`、`examples/comparison.json`。

### GLM-5.1 权重校验

固定官方版本：

- BF16：`26e1bd6e011feb778d25ae34b09b07074139d92d`
- FP8：`f396cf805182f4ca10fa675e1a99815b3ca384db`

读取官方 config、权重索引及少量 HTTP Range 文件头，未下载张量正文。张量模板的 shape、dtype、出现次数保存在包内，并与结构适配器逐项对照。

| 口径 | BF16 bytes | FP8 bytes，包含 scale 和保留的 BF16/F32 张量 |
|---|---:|---:|
| 官方完整 checkpoint，包含 MTP | 1,507,728,316,928 | 756,162,687,872 |
| 本组件默认基础模型，不启用 MTP | 1,487,822,475,264 | 746,130,054,912 |

第一行的结构推导结果与官方索引 `metadata.total_size` 完全一致。这个一致性验证的是 checkpoint 张量清单，不意味着运行时 HBM 就等于文件大小；组件还单独计入复制、分片及 MLA BF16 投影副本等开销。

### GLM-5.1 单 token 状态

MLA/DSA 使用潜在状态，不能按 config 中的 64 个 KV heads 套普通 GQA 公式。

每层、每个驻留 token：

- BF16 MLA：`512*2 + 64*2 = 1152 bytes`。
- FP8 MLA：`512 + 16(scale) + 64*2(RoPE) = 656 bytes`。
- DSA 索引：`128(FP8 key) + 4(scale) = 132 bytes`，不乘 `index_n_heads`。

因此 78 层逻辑模型是：

- BF16：`78*(1152+132) = 100,152 bytes/token`。
- FP8：`78*(656+132) = 61,464 bytes/token`。

MLA 和索引状态在 TP ranks 上复制，不能再除以 TP。PP 只保存该 stage 的层；TP8/PP2 的每卡数值分别为 50,076 和 30,732 bytes/token。这里的单 token 大小包括索引状态；分项在 `kv_breakdown_per_token` 中保留。

这些是物理缓存 payload 的单 token 大小，不包含激活，不表示每生成一个 token 都会立刻分配同样多的显存。真实缓存按块申请；组件按每请求向上取整，并计入一个保留 null block。

## 输入约定

JSON 顶层可包含 `model / hardware / parallelism / workload / runtime / weight_quantization / kv_quantization / search`。未知字段报错，避免拼写错误被忽略。配置相对路径相对于场景文件。

| 输入 | 关键约定 |
|---|---|
| `model` | `"glm-5.1"` 或 `{ "config_path": "...", "id": "...", "revision": "..." }`；也支持内联 `config` |
| `parallelism` | `num_gpus = tp * pp * dp`；EP 不额外乘卡数 |
| `ep` | 在当前 vLLM adapter 中，1 表示关闭；启用时必须等于 `tp*dp` |
| `stage_layers` | 可选的连续分层数列表，长度为 PP、总和为模型层数 |
| `weight_quantization` | `bf16`、`fp8_block`；`fp8` 为后者别名。通用 dense adapter 另支持 `fp16` |
| `kv_quantization` | GLM DSA：`bf16`、`fp8_ds_mla`；`fp8` 会映射为该特定格式 |
| `context_tokens` | 每请求驻留的总上下文预算，包含输入和输出；不额外再加输出 |
| `concurrency_per_dp` | 每个 attention DP 副本的并发，不是整个部署的总并发 |
| `max_batch_tokens` | 每个引擎 step 的 token budget，影响激活估计；不是驻留 token 总数 |
| `hardware.memory_bytes` | 实际可见容量；内置示例只使用 H800 80GB 名义值 |
| `external_used_bytes_per_gpu` | 当前逐卡相同的外部占用假设；本版不支持逐卡异构容量 |
| `allow_cross_node_tp` | 默认 false。连续放置导致 TP 组跨节点时拒绝；显式打开后仍需网络 |

vLLM mesh 的逻辑 rank 顺序为 `(dp, pp, tp)`，TP 最快变化。`node = rank // gpus_per_node` 是连续放置；不隐式插入 GPU 空洞。

vLLM 的普通 DP 复制完整权重副本。EP 打开后，routed experts 分布到 TP*DP mesh，而 attention 按 TP 分片、按 DP 复制；对 vLLM 默认 allgather/reducescatter 后端，TP>1 且 DP>1 时的 shared experts 使用 sequence parallel，在卡上复制而不是继续除以 TP。

## 显存模型与输出

对 GPU r：

```text
budget_r = min(memory_bytes * utilization,
               memory_bytes - external_used_bytes_per_gpu) - safety_margin_bytes

serving_peak_r = weights_r + kv_required_r + runtime_peak_r
peak_r = max(load_peak_r, serving_peak_r)
headroom_r = budget_r - peak_r
```

`runtime_breakdown` 包含 CUDA context、通信、allocator、激活工作区、DSA prefill 工作区、FP8 权重的 MLA BF16 投影副本、block table 和额外预留。加载峰值包含加载前已存在的通信缓冲；加载临时空间与服务阶段取最大值，不把互斥阶段全部叠加。KV 量化减少缓存需求，但当加载阶段是瓶颈时，整体峰值余量可能保持不变。

核心输出：

| 字段 | 含义 |
|---|---|
| `status` | `pass / fail / invalid / unsupported` |
| `memory_pass` | true / false；非法或不支持时为 null |
| `deployability` | pass 时为 `static_candidate`，始终不是实机验证 |
| `summary.minimum_headroom_bytes` | 最紧张 GPU 的预算余量 |
| `ranks[*].weight_bytes` | 按实际分片/复制和量化 scale 计算的常驻权重 |
| `ranks[*].kv_bytes_per_token` | 该卡保存的单 token 状态 payload |
| `summary.kv_bytes_per_token_logical` | 完整逻辑模型、无 TP 复制的单 token 状态 |
| `summary.kv_bytes_per_token_per_dp_replica_physical` | 一个 DP 副本跨 TP/PP 的物理状态总和，包含复制 |
| `ranks[*].kv_capacity_tokens` | 扣除非 KV 开销和 null block 后的可用缓存 token 槽位 |
| `ranks[*].physical_slack_at_peak_bytes` | 算术物理余量；不等于引擎预分配 KV 池后的设备空闲 |
| `summary.engine_args / engine_env` | 候选需要的参数，包括显式 PP 分层；需用户负责集群 bootstrap |

引擎参数中的利用率会扣除安全余量，匹配预算口径；单独提供 `requested_utilization` 保留原始输入。引擎可能预分配整个 KV 池，设备显示空闲不等于逻辑缓存空闲。

### 搜索与 PP 优化

`search` 可限制 TP/DP/EP/PP 候选、允许卡数及最大卡数。默认 TP 为 1/2/4/8、DP 为 1/2/4、PP 为 1..16；vLLM EP 枚举关闭和完整 mesh，SGLang EP 枚举 native TP 的合法约数。返回最少卡数只对公开的 `search_space` 成立，不能声称全世界所有框架、量化及拓扑的全局最少卡数。

PP 优化用 O(PP × layers²) 动态规划，在当前显存模型下，精确寻找固定并行度的最优连续非空层分配，使最紧张卡的加载/服务峰值最小。它不会优化吞吐，也不会改变模型计算图。按引擎输出 `VLLM_PP_LAYER_PARTITION` 或 `SGLANG_PP_LAYER_PARTITION`，使估计分层可被实际引擎复现。

### 运行开销假设

默认 vLLM `RuntimeProfile`：CUDA context 1 GiB、通信 2 GiB、allocator 1 GiB、固定激活工作区 1 GiB；另加 `max_batch_tokens * hidden_size * 2 * activation_copies`（默认 16 份）以及 DSA 工作区 envelope。加载另用 2 GiB 临时空间。它们都是可覆写的假设值，不是 H800 profiling 结果。

默认仅覆盖 eager、batch token budget <=8192、并发/DP <=128。超出范围会返回 unsupported。`extra_bytes`、`kv_pool_overhead_ratio` 可用于更保守的预留；本版尚未提供实测 profile 的采集/拟合 adapter。

## 支持边界

- 主测试架构是 GLM-5/5.1 `glm_moe_dsa`：压缩 MLA、DSA 索引、混合 FP8、EP 和 PP。
- 附带普通 dense Llama/Qwen2/Qwen3 adapter，支持 MHA/GQA；这些没有像 GLM 一样用真实 checkpoint 做元数据对照，任意自定义变体仍需新增适配器。
- 固定能力口径为 vLLM 0.19.0 和 SGLang 0.5.10；GLM runtime 仅覆盖 Hopper SM90、64 token cache page 和 BF16 compute，具体注意力后端由适配器显式输出。
- 本版不支持 INT4/NVFP4 权重、INT4 KV、CUDA Graph、MTP、offload、prefix sharing、滑窗/混合注意力、冗余专家或任意 EP 子网格；相关输入会拒绝或需专用 adapter，不静默降级。
- 互联只有可连接性和 TP 跨节点策略检查，没有带宽、吞吐或延迟模型。
- 内核临时空间、通信峰值、分配器行为仍可能超出假设，因此最小候选应优先做实机验证。

## 校验与构建

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m build
```

119 项测试（包含新增旗舰结构用例）覆盖官方 BF16/FP8 元数据对照、MLA/DSA 单 token 大小、TP KV 复制、DP/EP、sequence-parallel shared experts、PP 逐卡不均衡及最优分层、缓存块/null block、容量边界、加载峰值、非法组合、搜索和 CLI，以及 SGLang 的 native 参数、部分 EP、DP attention、等额节点放置和真实池形状。

模型配置、精简张量元数据和来源记录位于 `src/inferfit/data/`，足够离线运行及测试。完整权重索引、上游源码副本和本地研究报告不随仓库或发行包分发。

## 主要依据

- [GLM-5.1 官方 config](https://huggingface.co/zai-org/GLM-5.1/blob/26e1bd6e011feb778d25ae34b09b07074139d92d/config.json)
- [GLM-5.1 FP8 官方 config](https://huggingface.co/zai-org/GLM-5.1-FP8/blob/f396cf805182f4ca10fa675e1a99815b3ca384db/config.json)
- [vLLM v0.19.0 GLM/DeepSeek 模型实现](https://github.com/vllm-project/vllm/blob/v0.19.0/vllm/model_executor/models/deepseek_v2.py)
- [vLLM FP8 MLA 格式：656 bytes](https://github.com/vllm-project/vllm/blob/v0.19.0/vllm/v1/attention/backends/mla/flashmla_sparse.py)
- [vLLM 缓存池分配](https://github.com/vllm-project/vllm/blob/v0.19.0/vllm/v1/core/kv_cache_utils.py)
- [vLLM PP 显式分层](https://github.com/vllm-project/vllm/blob/v0.19.0/vllm/distributed/utils.py)
- [NVIDIA H800 80GB 官方列举](https://docs.nvidia.com/ai-enterprise/release-7/latest/infra-software/vgpu/reference/hopper.html)

SGLang 依据：

- [固定提交的 GLM 模型](https://github.com/sgl-project/sglang/blob/1519acf37c23f2189adb93f57ca9cd2db1bebf18/python/sglang/srt/models/glm4_moe.py)
- [MLA/NSA 实际内存池](https://github.com/sgl-project/sglang/blob/1519acf37c23f2189adb93f57ca9cd2db1bebf18/python/sglang/srt/mem_cache/memory_pool.py)
- [内存容量与 request 映射计算](https://github.com/sgl-project/sglang/blob/1519acf37c23f2189adb93f57ca9cd2db1bebf18/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py)
- [固定版本 ServerArgs 约束](https://github.com/sgl-project/sglang/blob/1519acf37c23f2189adb93f57ca9cd2db1bebf18/python/sglang/srt/server_args.py)
