from inferfit import (
    GiB, Hardware, RuntimeProfile, Workload,
    load_model, search,
)

model = load_model("glm-5.1")

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
print("容量统计：", result.capacity_counts)
print("兼容性统计：", result.compatibility_counts)

for candidate in result.candidates:
    print(
        candidate.parallelism,
        "PP 分层：", candidate.parallelism.stage_layers,
        "最小余量 GiB：",
        candidate.capacity.summary["minimum_headroom_bytes"] / GiB,
    )
