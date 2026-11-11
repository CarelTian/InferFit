# InferFit 架构与长期维护

## 容量与后端兼容性分离

新增 `analyze / search_plans`；旧 `check / search / optimize_pipeline / check_pd`
保持严格入口。新入口将显存供给、内存需求、执行能力和实机验证分别表达：

```text
Hardware(容量/身份/拓扑) + RuntimeProfile(构建/预留) + 模型/负载/并行
                  │
                  ▼
          backends.resolve_backend             每次 search 解析一次
          ┌───────┴──────────┐
     inspected                  unknown
          │                         │
     严格 check                 通用 GQA 或显式参考布局
          │                         │
          └──────────┬──────────────┘
                     ▼
       共享 estimator / states / weights / PP 成本编译
                     │
       capacity(pass/fail/unknown/invalid)
       compatibility(supported/unsupported/unknown)
       verification(not_run)
```

命中已检查的完整合同后，沿用精确后端规则；容量不通过也可同时报告后端受支持。
未命中时保留容量分析能力。已知不支持不会因降级而变成受支持；输入无效也不会变成
条件通过。未知缓存/转换规则不按零计费，而是返回 missing_data；复杂模型须显式选择
已经固定版本的参考布局。引用某布局是容量假设，不继承它对目标硬件的运行保证。

分析结果定义在 `analysis_types.py`，不向旧 CheckResult/SearchResult JSON 强行加入字段。
`analysis.py` 负责组合两条结果轴和搜索策略；`engines/generic.py` 是容量布局，
其 validate 始终拒绝将自身用作执行后端；`estimator._check(capacity_only=True)`
复用相同的逐 rank 数学路径，绕过硬件算子/launcher 限制，不伪造 NVIDIA 设备。
通用输入校验仍先执行，参考布局的页大小/专家所有权约束仍需满足。

`backends/` 是设备与构建的解析边界：区分厂商、device family、框架、distribution、
image、build revision、SDK。设备 API 的 `cuda` 字符串不足以识别 NVIDIA。
硬件目录仅保留 NVIDIA 规格；后端目录列出维护中的上游 CUDA 合同。
未知设备与厂商构建不会提升为 inspected。未来新设备若只需容量预检，提供容量与
已知模型/布局即可；需要部署候选时再增加独立后端合同及校验。

搜索 `policy=capacity` 接受条件容量通过；默认 `policy=static` 还要求兼容性受支持。
两者都不声明实机启动成功。结果包含两轴计数与聚合拒绝原因，区分无枚举组合、
显存不足、布局缺失、执行不支持。PP 优化使用区间成本编译和最终峰值校验。
容量 fallback 不生成 engine_args/env，不把算术节点打包当成可启动放置。

本阶段不引入性能仿真、在线下载或自动执行部署；新的设备合同与 PD 传输校准
仍需指定构建及实机证据。

当前架构将模型清单、状态布局、引擎规则与规划算法分开。兼容转发模块已移除；
每项实现使用其所属模块的规范路径。

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
| `models.py` | 公共 loader、显式 model_type 注册表 | 注册新转换器 |
| `model_families/` | config/header → 具有明确角色和 placement 的张量清单 | 新模型结构；同家族优先复用 builder |
| `flagships.py`、`snapshots.py`、`data/` | 离线版本目录、配置/清单摘要校验、来源 | 新快照、变体或量化 checkpoint |
| `weights.py` | dtype、已打包字节数、量化格式约束 | 新编码或 scale 格式 |
| `states/` | 单 token 几何、精确池分配、固定状态、容量反解 | 新缓存或 recurrent state 布局 |
| `engine_versions.py`、`engines/` | 独立引擎合同、版本能力、TP/EP 放置、dtype 转换、启动参数 | 引擎新版本或新 kernel/backend |
| `estimator.py` | 权重/加载/服务峰值组合、逐卡预算、结果结构 | 通用预算规则变化 |
| `partition.py`、`search.py` | 连续 PP 最小最大值优化、有限组合搜索 | 搜索算法变化 |

SGLang 和 vLLM 都实现中立的 `EngineAdapter`，两者之间没有继承关系。
SGLang 统一由单个 SglangAdapter 维护；状态家族在同一个固定版本合同内明确限制适用范围。
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

## PP 成本编译与性能约束

`prepare_interval_cost()` 在一次优化开始时编译状态成本：标准家族使用原始每块
KV 字节数、固定状态的前缀和；V4 从正常 allocator 提取各池每个 owner 的原始
字节系数，再使用 C4/C128 owner 前缀计数。标准 KV 保留区间合并后的每块取整，
V4 保留各物理池独立取整，固定 metadata 每 stage 只计一次。
未知或非线性布局使用作用域仅限本次调用的精确 memoized fallback。

`partition.py` 为可行区间构建一次峰值成本表，DP 内层只比较整数和读取表。
分层仍为精确最小最大值目标，时间复杂度 O(PP × L²)，额外成本表空间 O(L²)。
第一层直接初始化，前驱按原有升序和严格小于更新，保持同分时的选择顺序。
优化后仍调用完整 `check()` 校验实际峰值与 DP 预测一致。

性能测试使用 `tools/benchmark_partition.py`，模型加载不进入计时，先 warmup 后
报告多次调用的中位数。`--profile-dir` 可显式保存 cProfile 数据和状态分配调用数。
正确性测试覆盖所有区间与原 allocator 的逐字节一致、非整数 pool overhead、
无压缩 owner 的 stage、新布局 fallback、穷举分层、加载/服务峰值及同分行为。

## 滚动版本与公开接口

唯一版本常量是 `engine_versions.SGLANG_VERSION`，上游工厂为
`RuntimeProfile.sglang()`。未知构建不会自动替换成上游版本。版本匹配失败仍拒绝严格检查；
分析接口可以给出条件容量。项目版本统一由 `_version.py` 定义为 0.1.0，构建元数据从该模块读取。

`__init__.py` 使用一个明确的 `__all__`。不再维护新旧导出列表、重复工厂和适配器
别名。真正的模型转换、显存检查、组合搜索与 PD 检查仍是独立职责，不因函数较早
引入就删除。静态检查会拒绝 package 中未使用的导入、重复定义和局部变量。

场景输入与独立测试保留，生成结果通过 CLI 或回归工具按需输出；生成文件不作为
当前行为的权威来源。新增 backend/状态规则必须更新独立形状校验与负向边界测试。
