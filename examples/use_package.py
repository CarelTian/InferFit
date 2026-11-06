"""Run after installing the wheel, or with PYTHONPATH=src."""
import json

from inferfit import Hardware, Parallelism, Workload, check, load_glm51, search

model = load_glm51()
hardware = Hardware(memory_bytes=80_000_000_000)
workload = Workload(context_tokens=8192, concurrency_per_dp=1)
result = check(
    model, hardware=hardware, workload=workload,
    parallelism=Parallelism(tp=8, dp=1, ep=8, pp=2, num_gpus=16),
    weight_quantization="fp8_block", kv_quantization="fp8",
)
print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
layouts = search(model, hardware=hardware, workload=workload, gpu_counts=[8, 12, 16, 24, 32])
print("Minimum GPUs within search space:", layouts.minimum_num_gpus)
