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
    assert limited["minimum_num_gpus"] == default["minimum_num_gpus"] == 12
    assert limited["evaluated"] == default["evaluated"] == 94
    assert limited["passing_candidates_total"] == default["passing_candidates_total"] == 26
    assert len(default["candidates"]) == 10
    assert limited["candidates"] == default["candidates"][:3]
    document.pop("search")
    for brief in limited["candidates"]:
        checked = evaluate_document({**document, "parallelism": brief["parallelism"]})
        assert brief["capacity"]["status"] == "pass"
        assert brief["compatibility"]["status"] == "supported"
        assert brief["static_candidate"]
        brief_memory = brief["capacity"]
        assert checked["memory_pass"] and checked["ranks"]
        assert "ranks" not in brief_memory
        assert brief["parallelism"]["stage_layers"] == checked["summary"]["stage_layers"]
        assert brief_memory["summary"]["minimum_headroom_bytes"] == checked["summary"]["minimum_headroom_bytes"]
        assert "engine_args" not in brief_memory["summary"]
