"""Reproducible CPU planner benchmarks; model loading is outside the timers.

PYTHONPATH=src python tools/benchmark_partition.py --iterations 7
Use --case to compare only models available in an older source checkout.
"""
from __future__ import annotations

import argparse
import cProfile
import json
from pathlib import Path
import platform
import statistics
import time

from inferfit import (__version__, Parallelism, RuntimeProfile, Workload, load_glm51,
                      load_model, optimize_pipeline, search)

CASES = ("glm-sglang-pp8", "glm-vllm-pp8", "qwen-hybrid-pp8", "v4-pro-pp4", "glm-search")


def prepare(name):
    if name.startswith("glm"):
        model = load_glm51()
        runtime = RuntimeProfile() if name == "glm-vllm-pp8" else RuntimeProfile.sglang()
        options = dict(runtime=runtime, weight_quantization="fp8", kv_quantization="bf16")
        if name == "glm-search":
            return lambda: search(model, tp_values=[1, 2, 4, 8], dp_values=[1],
                                  pp_values=range(1, 9), max_gpus=32, limit=20, **options)
        plan = Parallelism(tp=4, ep=4, pp=8, num_gpus=32)
    elif name == "qwen-hybrid-pp8":
        model = load_model("qwen3.8-2.4t-a95b")
        options = dict(runtime=RuntimeProfile.sglang(), weight_quantization="bf16")
        plan = Parallelism(tp=4, ep=4, pp=8, num_gpus=32)
    else:
        model = load_model("deepseek-v4-pro")
        options = dict(runtime=RuntimeProfile.sglang(), weight_quantization="native",
                       kv_quantization="fp8", workload=Workload(block_size=256))
        plan = Parallelism(tp=4, ep=4, pp=4, num_gpus=16)
    return lambda: optimize_pipeline(model, parallelism=plan, **options)


def result_summary(result):
    if hasattr(result, "candidates"):
        return dict(evaluated=result.evaluated, counts=result.counts,
                    minimum_num_gpus=result.minimum_num_gpus,
                    candidates=[dict(parallelism=c.to_dict()["parallelism"],
                        stage_layers=c.summary["stage_layers"],
                        minimum_headroom_bytes=c.summary["minimum_headroom_bytes"])
                        for c in result.candidates])
    return dict(status=result.status, stage_layers=result.summary["stage_layers"],
                peak_bytes=max(rank.peak_bytes for rank in result.ranks),
                minimum_headroom_bytes=result.summary["minimum_headroom_bytes"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=CASES)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--profile-dir", type=Path, help="Optional directory for cProfile files")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    results = []
    for name in args.case or CASES:
        run = prepare(name)
        run()  # Warm geometry/storage caches consistently for every version.
        samples = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            result = run()
            samples.append(time.perf_counter() - start)
        row = dict(case=name, median_ms=statistics.median(samples) * 1000,
                   samples_ms=[sample * 1000 for sample in samples], result=result_summary(result))
        if args.profile_dir:
            args.profile_dir.mkdir(parents=True, exist_ok=True)
            profile = cProfile.Profile()
            profile.runcall(run)
            profile.dump_stats(str(args.profile_dir / (name + ".prof")))
            row["state_allocation_calls"] = sum(entry.callcount for entry in profile.getstats()
                if getattr(entry.code, "co_name", None) == "allocate")
        results.append(row)
    print(json.dumps(dict(inferfit_version=__version__, python=platform.python_version(),
                         platform=platform.platform(), iterations=args.iterations, cases=results), indent=2))


if __name__ == "__main__":
    main()
