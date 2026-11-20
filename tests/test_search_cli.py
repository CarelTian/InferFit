"""Tool-facing search results retain ranked candidates within a small response."""
import json
from pathlib import Path
import subprocess
import sys

from inferfit.cli import evaluate_document


SCENARIO = Path(__file__).resolve().parents[1] / "examples/h800-glm51-sglang-search.json"


def test_cli_search_returns_reusable_layouts_within_small_response(tmp_path):
    command = [sys.executable, "-m", "inferfit", "search", str(SCENARIO)]
    default = json.loads(subprocess.check_output(command))
    document = json.loads(SCENARIO.read_text())
    document["search"]["limit"] = 3
    scenario = tmp_path / "limited.json"
    scenario.write_text(json.dumps(document))
    response = subprocess.check_output([*command[:-1], str(scenario)])
    limited = json.loads(response)
    assert len(response) < 4096
    assert not limited["deployment_verified"]
    assert limited["minimum_num_gpus"] == default["minimum_num_gpus"] == 16
    assert limited["evaluated"] == default["evaluated"] == 132
    assert limited["passing_candidates_total"] == default["passing_candidates_total"] == 87
    assert len(default["candidates"]) == 10
    assert limited["candidates"] == default["candidates"][:3]
    document.pop("search")
    assert {brief["sglang_dp_attention"] for brief in default["candidates"]} == {True, False}
    for brief in default["candidates"]:
        dp = brief["parallelism"]["dp"]
        assert brief["sglang_dp_attention"] == (dp > 1)
        assert brief["concurrency_per_dp"] == (3 + dp - 1) // dp
        checked = evaluate_document({
            **document, "parallelism": brief["parallelism"],
            "runtime": {**document["runtime"], "sglang_dp_attention": brief["sglang_dp_attention"]},
            "workload": {**document["workload"], "concurrency_per_dp": brief["concurrency_per_dp"]},
        })
        assert brief["capacity"]["status"] == "pass"
        assert brief["compatibility"]["status"] == "supported"
        assert brief["static_candidate"]
        brief_memory = brief["capacity"]
        assert checked["memory_pass"] and checked["ranks"]
        assert "ranks" not in brief_memory
        assert brief["parallelism"]["stage_layers"] == checked["summary"]["stage_layers"]
        assert brief_memory["summary"]["minimum_headroom_bytes"] == checked["summary"]["minimum_headroom_bytes"]
        assert checked["summary"]["engine_args"]["attn_dp_size"] == dp
        assert "engine_args" not in brief_memory["summary"]
