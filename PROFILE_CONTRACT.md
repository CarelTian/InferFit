# InferFit 输入合同与扩展指南（0.5.0）

预检层接收统一的 `ModelProfile + Hardware + RuntimeProfile + Parallelism + Workload`。
`check / search / optimize_pipeline / check_pd` 不依赖模型商品名或硬件 catalog 名称。

```text
官方 config + 权重 header → 模型转换器 → model.json ──┐
设备规格 / 设备查询结果              → hardware.json ├→ 预检 / 搜索 / PP 优化 / PD 容量检查
引擎构建 + 运行开销假设              → runtime.json ─┘
```

## 你需要提供什么

| 数据 | 必填或推荐内容 | 从哪里获取 |
|---|---|---|
| 模型身份 | model_id、固定 revision、memory_family | 模型仓库和具体提交 |
| 基础结构 | 层数、hidden size、Q heads、最大上下文 | config.json |
| GQA 状态 | KV heads、head dimension | config / 模型实现 |
| MoE 常驻权重 | 专家总数；每专家完整权重，非激活专家数 | config + 权重 header |
| 张量清单 | 名称、shape、dtype、所在层、分片方式、角色、重复数量 | safetensors header / index + 引擎分片实现 |
| 权重格式 | BF16/FP16 或目前支持的 128×128 FP8 block；哪些张量保留高精度 | 量化配置 / header |
| 硬件 | 名称、实际可见 bytes、平台、CUDA compute capability、每节点卡数及可选 available_nodes | 设备查询；规格值须标记 nominal |
| 运行时 | 固定 engine/version、激活/通信/加载/allocator 等预留 | 对应构建的参数及 profiling；初期可使用显式假设 |
| 场景 | TP/DP/EP/PP、卡数、上下文、每 DP 并发、batch tokens、权重与 KV 精度 | 用户输入 |

只有参数量或只有 config.json 时，通常还不足以确定真实权重布局和运行时开销。
精确推导权重清单通常只需 header，无需将完整权重加载到 GPU。

## 1. model.json

所有 profile 使用 `schema_version: 1`，未知字段和未知版本报错；JSON 不执行表达式，也不自动导入 Python 模块。

完整真实示例：`examples/profiles/qwen3-235b.model.json` 和 `glm51.model.json`。
可以从已有适配器导出，建立新模型的数据模板：

```python
import json
from inferfit import load_qwen3_235b, model_profile_to_dict, load_model_profile

manifest = model_profile_to_dict(load_qwen3_235b())
# 将身份、结构和整个张量清单替换为新模型事实，不能只改名称。
manifest['reference_weight_bytes'] = {'bf16': 470187269120}
with open('model.json', 'w') as f:
    json.dump(manifest, f, indent=2)
model = load_model_profile('model.json')
```

顶层字段：`schema_version / kind="model" / model_id / revision / memory_family / config / tensors / provenance`。
`reference_weight_bytes` 可选，用独立的 checkpoint 元数据核对清单字节数。它验证大小一致性，不证明清单与真实模型逐张量完全相符。

当前可复用内存家族：

- `gqa`：普通全注意力 MHA/GQA。
- `gqa_moe`：全注意力 GQA + MoE。
- `mla_dsa`：当前固定后端的 MLA + DSA；受 512/64/128 等实际后端限制。

GQA 最小 config 为 `num_hidden_layers / hidden_size / num_attention_heads / num_key_value_heads / head_dim / max_position_embeddings`。
MoE 另需 `num_experts`；MLA/DSA 另需 `kv_lora_rank / qk_rope_head_dim / qk_nope_head_dim / v_head_dim / index_head_dim / n_routed_experts`。
没有 tying 信息默认视为不共享；需要共享 embedding/输出时明确 `tie_word_embeddings` 并提供正确清单。

每条 tensor 数据：

```json
{
  "name": "any_real_tensor_name",
  "shape": [1536, 4096],
  "dtype": "BF16",
  "layer": 0,
  "placement": "expert_column",
  "multiplicity": 128,
  "quantizable": true,
  "role": "routed_expert",
  "pp_owner": null
}
```

- shape 表示**单份张量**；multiplicity 表示相同 shape 的重复份数，例如 128 个常驻专家。
- dtype 为源权重 BF16/F16/F32。`quantizable=true` 表示选择 fp8_block 时允许转换，不表示当前 header 已是 FP8。
- placement：`replicated / column / row / vocab / kv_column / expert_column / expert_row`。column/row 是输出/输入维分片；kv_column 会处理 KV head 复制；专家维与剩余 expert TP 由引擎 adapter 决定。
- role：`embedding / output / attention / dense_ffn / shared_expert / routed_expert / norm / index_norm / index / generic`。外部清单必须显式填写，不能使用 auto；核心不再依赖张量名称子串判断角色。
- 普通层张量写 layer 编号，pp_owner=null；全局张量写 layer=null，并明确 pp_owner=`first` 或 `last`。权重重复、复制或 tying 关系由调用方如实描述。

Loader 校验层范围、层覆盖、重复名称、正整数 shape、dtype、专家重复数和参考字节数等。
它无法从元数据判断你是否把某个真实张量漏报；清单的来源与正确性仍需要对照 header 和模型实现。

## 2. hardware.json

无需修改内置 catalog：

```json
{
  "schema_version": 1,
  "kind": "hardware",
  "hardware": {
    "name": "new-board",
    "memory_bytes": 80000000000,
    "platform": "cuda",
    "compute_capability": 90,
    "gpus_per_node": 8
  },
  "provenance": {"kind": "device_query"}
}
```

`available_nodes` 是真实可用节点数的上限，建议填写，避免将“理论需要三台”误认为“现有两台能放下”。

可以添加 Hardware 中已有的 utilization、外部占用、安全余量和互联策略字段。
新 CUDA 卡可以复用现有计算，但驱动/内核和固定后端的能力检查仍有效；例如 A100 不会因为改了名称就被当作 Hopper。
新 PPU/ROCm/其他平台可以用 profile 描述库存（非 CUDA 的 compute_capability 必须为 null），当前 CUDA adapter 仍返回 unsupported。
**提供硬件容量不等于完成了该平台的运行时适配。**

## 3. runtime.json

```json
{
  "schema_version": 1,
  "kind": "runtime",
  "runtime": {
    "engine": "sglang",
    "version": "0.5.10",
    "provenance": "assumed",
    "communication_bytes": 2147483648,
    "activation_fixed_bytes": 2147483648,
    "extra_bytes": 1073741824
  }
}
```

省略的字段使用对应工厂默认值。支持固定 vLLM 0.19.0 / SGLang 0.5.10；JSON 不能扩充或绕过引擎能力规则。
当前仅支持 assumed profile，尚未提供 measured profile 采集与校准 adapter。
相同模型家族只说明内存公式可复用，仍需确认实际引擎已实现相应模型和 kernels。

## 4. 使用统一接口

```python
from inferfit import *

model = load_model_profile('model.json')
hardware = load_hardware_profile('hardware.json')
runtime = load_runtime_profile('runtime.json')
result = check(model, hardware=hardware, runtime=runtime,
               parallelism=Parallelism(tp=8, dp=1, ep=8, pp=1, num_gpus=8),
               weight_quantization='bf16')
layouts = search(model, hardware=hardware, runtime=runtime,
                 weight_quantization='bf16', max_gpus=32)
```

CLI 场景通过相对于场景文件的路径引用，不静默下载、合并或覆盖 profile：

```json
{
  "model": {"profile_path": "model.json"},
  "hardware": {"profile_path": "hardware.json"},
  "runtime": {"profile_path": "runtime.json"},
  "parallelism": {"tp": 8, "dp": 1, "ep": 8, "pp": 1, "num_gpus": 8},
  "weight_quantization": "bf16"
}
```

同一引用方式适用于 search，以及 pd-check 的 model 和两个池内的 hardware/runtime。
`profile_path` 与内联字段互斥，避免隐藏覆盖。Python API 的 PDPool 直接传加载后的对象。

## 5. 从官方 config 自动生成 profile

同一家族的新模型若不想手写清单，可以在应用进程里注册一次**可信 Python 转换器**：

```python
from inferfit import register_model_adapter, from_config, load_model_profile

def convert_vendor_config(config, **identity):
    # 应用侧转换逻辑：结合该架构的 config/header，生成上面的完整 manifest。
    manifest = build_manifest_from_verified_metadata(config, **identity)
    return load_model_profile(manifest)

register_model_adapter('vendor_model_type', convert_vendor_config)
model = from_config({'model_type': 'vendor_model_type', ...})
# 此后 check/search/optimize_pipeline/check_pd 均无需修改。
```

上例的 build_manifest_from_verified_metadata 是调用方实现的转换逻辑，不是包内 API。
注册键必须唯一，不能替换内置模型类型。注册仅在当前 Python 进程生效；CLI 可直接使用转换结果 JSON。

## 何时必须写新代码

新张量排列如果可用现有 placements/roles 表达，只需数据或 config 转换器。
但滑窗、跨层共享缓存、Engram、压缩状态、非线性缓存增长、新量化块格式，以及新设备平台/引擎后端，需要新增状态、量化或引擎 adapter 并做回归；本版不会用一个任意 bytes/token 常数将其伪装为已适配。
因此 DeepSeek V4.1 和 ZW810 的已有拒绝边界仍保留。Pass 继续是数据及运行时假设下的容量判断，不是硬件启动验证。
