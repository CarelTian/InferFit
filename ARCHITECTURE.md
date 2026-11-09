# InferFit 架构与长期维护

0.8.0 将模型清单、状态布局、引擎规则与规划算法分开。旧 SDK 和
`inferfit.adapters`、`inferfit.qwen_flash_next` 导入路径继续可用。

```text
config + pinned tensor headers
          │
          ▼
model_families/ ──► ModelProfile ◄── data/*.json / external model profile
                         │
                 check / search / optimize_pipeline
                         │
              ┌──────────┼───────────┐
              ▼          ▼           ▼
         weights.py    states/     engines/
         存储字节数    KV/状态池    切分/参数/能力
              └──────────┼───────────┘
                         ▼
                逐 rank 峰值、余量、候选
```

## 模块边界

| 位置 | 职责 | 新增内容的典型情形 |
|---|---|---|
| `models.py` | 公共 loader、显式 model_type 注册表 | 注册新转换器，保留 SDK 兼容性 |
| `model_families/` | config/header → 具有明确角色和 placement 的张量清单 | 新模型结构；同家族优先复用 builder |
| `flagships.py`、`snapshots.py`、`data/` | 离线版本目录、配置/清单摘要校验、来源 | 新快照、变体或量化 checkpoint |
| `weights.py` | dtype、已打包字节数、量化格式约束 | 新编码或 scale 格式 |
| `states/` | 单 token 几何、精确池分配、固定状态、容量反解 | 新缓存或 recurrent state 布局 |
| `engine_versions.py`、`engines/` | 独立引擎合同、版本能力、TP/EP 放置、dtype 转换、启动参数 | 引擎新版本或新 kernel/backend |
| `estimator.py` | 权重/加载/服务峰值组合、逐卡预算、结果结构 | 通用预算规则变化 |
| `partition.py`、`search.py` | 连续 PP 最小最大值优化、有限组合搜索 | 搜索算法变化 |

SGLang 和 vLLM 都实现中立的 `EngineAdapter`，两者之间没有继承关系。
旧实现让 SGLang 继承 vLLM 是为了复用普通 rank/张量规则；Python 语法允许，
但语义上的 “SGLang is a vLLM” 不成立。未覆盖的方法会意外继承其他引擎的
缓存、rank 或量化规则，因此不能作为长期维护的边界。
同一引擎的版本特化仍可继承该引擎实现，但必须显式限制适用模型、版本和后端。
新增引擎无法靠继承另一引擎跳过抽象接口中的必需方法。

`StateAllocation` 同时给出缓存池字节数和运行状态分项。
`kv_per_token()` 是几何描述；V4 返回明确标记的、向上取整的分页摊销速率。
`check()` 使用精确的池分配，非线性布局的容量通过重新分配和二分反解计算，已有线性家族保留原容量算法。
PP 优化也调用同一个分配接口，避免优化器与预检使用两套近似公式。

## 新增模型需要提供什么

1. **身份与结构**：完整 config、不可变 revision、模型 ID；不能只给参数总量。
2. **真实存储清单**：tensor shapes/dtypes、scale、FP4 打包维度、专家数、MTP/视觉/PLE 等驻留范围；来源为 index/header。
3. **状态规则**：每层 attention 类型、共享/压缩关系、页大小、padding、请求槽位、状态 dtype 和固定 ring。
4. **引擎合同**：固定版本、backend、TP/EP/DP/PP 映射、加载后转换、offload/graph/MTP 是否启用。
5. **独立校验值**：checkpoint 总字节数、至少一个实际分配形状或实测值、失败边界和不支持的组合。

已有家族、存储与引擎合同均适用时，只新增数据 profile/快照和回归用例。
仅 model_type 不同，可以通过 `register_model_adapter()` 注册可信 Python 转换器。
新缓存结构放进 `states/` 并登记 family；新引擎放进 `engines/`。
JSON profile 不会导入任意代码，也不能把未知后端变成受支持后端。

硬件接入单独提供 `Hardware` 或 hardware profile：实际可用显存、平台/SM、
每节点卡数、可用节点数、互联、外部显存占用和安全余量。
新平台还需要对应的引擎/kernel 合同，修改硬件名称或容量不足以完成适配。

## 快照与验证流程

开发期使用 `tools/collect_snapshot.py MODEL_ID REVISION /tmp/cache` 只收集
配置、索引和 HTTP Range 文件头。服务端不返回精确 206 时拒绝读取。
再使用 `--compact /tmp/cache /tmp/model-metadata.json` 合并形状相同的专家行，
保留 multiplicity，并核对每个 tensor 的 offsets/dtype 和完整 index 字节数。
原始源码、完整索引与本地研究目录不进入发行包。

将精简清单与配置放到 `data/`，在目录中登记 config/metadata SHA-256、revision、
scope、support 和 blockers；来源链接与上游源码摘要保留在 metadata。
不要把 BF16 profile 的 dtype 直接改为 FP8：原生量化的 exclusions、scale 和打包
规则必须单独核对。`native` 表示复用 checkpoint 的混合布局，不进行二次量化。

验证包括：独立总字节数/池形状、页与容量边界、版本与硬件拒绝、PP 优化对照穷举、
搜索、JSON roundtrip、CLI、旧场景回归，以及构建后的离线 wheel 安装和 sdist 测试。
真实 GPU 启动、峰值采样和性能验证另记；未执行时保持 `deployment_verified=false`。

当前 V4 支持原始 Flash/Pro，V4.1 的跨层共享、Engram 和 32×32 量化属于另一份合同。
原始 V4 的成功预检不能作为 V4.1 的支持依据。

## SGLang 版本选择

`sglang_current()` 明确选择当前维护的 0.5.23 模型合同；`sglang_legacy()`
保留 0.5.10。旧 `sglang()` 工厂保持兼容，不会静默改变现有场景的能力规则。
`current` 是本包已核对的固定版本，更新上游版本需要重新检查布局与参数并增加
回归测试，不会运行时跟随 `latest`。0.5.23 当前仅覆盖本包列明的模型与后端。
