"""SDK examples: python examples/native_models.py [--search]."""
import argparse
import json

from inferfit import Hardware, Parallelism, RuntimeProfile, Workload, check, load_model, search


def run(include_search=False):
    for name, tp, pp, page, kv in (
        ("qwen3.8-flash-next-fp8", 4, 1, 64, "bf16"),
        ("deepseek-v4-flash", 4, 1, 256, "fp8"),
        ("deepseek-v4-pro", 8, 2, 256, "fp8"),
    ):
        model = load_model(name)
        options = dict(hardware=Hardware(), runtime=RuntimeProfile.sglang(),
                       workload=Workload(context_tokens=8192, block_size=page),
                       weight_quantization="native", kv_quantization=kv)
        result = check(model, parallelism=Parallelism(tp=tp, ep=tp, pp=pp, num_gpus=tp*pp), **options)
        output = dict(model=name, status=result.status,
                      minimum_headroom_gib=result.summary.get("minimum_headroom_bytes", 0) / 1024**3,
                      kv_bytes_per_token_per_rank=result.summary.get("kv_bytes_per_token_per_rank_max"),
                      deployment_verified=result.summary.get("deployment_verified", False))
        if include_search:
            layouts = search(model, tp_values=[1, 2, 4, 8], dp_values=[1],
                             pp_values=[1] if page == 64 else range(1, 17), max_gpus=32,
                             limit=3, **options)
            output.update(minimum_num_gpus=layouts.minimum_num_gpus, evaluated=layouts.evaluated,
                          candidates=[dict(tp=c.parallelism.tp, ep=c.parallelism.ep,
                              dp=c.parallelism.dp, pp=c.parallelism.pp,
                              stage_layers=c.summary["stage_layers"],
                              engine_args=c.summary["engine_args"], engine_env=c.summary["engine_env"])
                              for c in layouts.candidates])
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", action="store_true")
    run(parser.parse_args().search)
