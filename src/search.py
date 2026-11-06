from pathlib import Path

from inferfit import (
    GiB, Hardware, RuntimeProfile, Workload,
    load_model_profile, search,
)

model = load_model_profile(
    Path(__file__).resolve().parents[1] / "reports/glm5-fp8-h800/model.profile.json"
)

result = search(
    model,
    hardware=Hardware(
        memory_bytes=80 * GiB,  # 示例假设，替换为实际可见容量
        gpus_per_node=8,
        available_nodes=10,
    ),
    runtime=RuntimeProfile.sglang(),
    workload=Workload(
        context_tokens=8192,
        concurrency_per_dp=1,
        max_batch_tokens=2048,
    ),
    weight_quantization="fp8",
    kv_quantization="bf16",

    tp_values=range(1,17),
    dp_values=range(1, 17),
    pp_values=range(1, 17),
    ep_values=None,       # 自动枚举该引擎允许的 EP
    max_gpus=100,
    optimize_pp=True,     # 同时优化 PP 层分配
    limit=100,           # 最多返回 100 个通过的候选
)

print("评估次数：", result.evaluated)
print("状态统计：", result.counts)

for candidate in result.candidates:
    print(
        candidate.parallelism,
        "PP 分层：", candidate.summary["stage_layers"],
        "最小余量 GiB：",
        candidate.summary["minimum_headroom_bytes"] / GiB,
    )
