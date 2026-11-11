"""Collect reproducible HF *headers*, never tensor payloads (development only).

python tools/collect_snapshot.py MODEL_ID REVISION OUTPUT_DIRECTORY
python tools/collect_snapshot.py --compact CACHE_DIRECTORY OUTPUT_JSON
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import urllib.request

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8,
               "I8": 1, "U8": 1, "F8_E4M3": 1, "F8_E8M0": 1}


def compact(directory: Path) -> dict:
    source = json.loads((directory / "source.json").read_text())
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    tensors, digests = {}, {}
    for shard in sorted(set(index["weight_map"].values())):
        payload = (directory / "headers" / shard.replace(".safetensors", ".json")).read_bytes()
        header = json.loads(payload)
        digests[shard] = hashlib.sha256(json.dumps(header, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        for name, row in header.items():
            if name == "__metadata__":
                continue
            if name in tensors or index["weight_map"].get(name) != shard:
                raise ValueError("Duplicate tensor or shard/index disagreement")
            expected = math.prod(row["shape"]) * DTYPE_BYTES[row["dtype"]]
            if row["data_offsets"][1] - row["data_offsets"][0] != expected:
                raise ValueError(f"Unexpected encoded storage: {name}")
            tensors[name] = row
    if set(tensors) != set(index["weight_map"]):
        raise ValueError("Incomplete checkpoint inventory")
    groups, ids = {}, collections.defaultdict(set)
    total = mtp = 0
    dtype_bytes = collections.Counter()
    for name, row in sorted(tensors.items()):
        size = math.prod(row["shape"]) * DTYPE_BYTES[row["dtype"]]
        total += size
        dtype_bytes[row["dtype"]] += size
        if name.startswith("mtp."):
            mtp += size
        match = re.search(r"\.experts\.(\d+)\.", name)
        key = name if match is None else name[:match.start(1)] + "{expert}" + name[match.end(1):]
        item = {"name": key, "shape": row["shape"], "dtype": row["dtype"]}
        if key in groups and any(groups[key][k] != item[k] for k in item):
            raise ValueError("Expert tensors have heterogeneous shapes or dtypes")
        groups[key] = item
        if match:
            ids[key].add(int(match[1]))
    for key, expert_ids in ids.items():
        if expert_ids != set(range(len(expert_ids))):
            raise ValueError("Expert IDs are not contiguous")
        groups[key]["multiplicity"] = len(expert_ids)
    if total != index["metadata"]["total_size"]:
        raise ValueError("Header bytes disagree with checkpoint index")
    return {"model_id": source["model_id"], "revision": source["revision"],
            "checkpoint_bytes": total, "mtp_bytes": mtp,
            "tensor_count": len(tensors), "shard_count": len(digests), "tensor_payload_downloaded": False,
            "dtype_bytes": dict(dtype_bytes), "canonical_header_sha256": digests,
            "index_sha256": hashlib.sha256((directory / "model.safetensors.index.json").read_bytes()).hexdigest(),
            "tensors": list(groups.values())}


def download(model_id: str, revision: str, directory: Path) -> None:
    """Pin the revision before calling; Range responses must be exact HTTP 206."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "headers").mkdir(exist_ok=True)
    (directory / "source.json").write_text(json.dumps({"model_id": model_id, "revision": revision}))
    base = f"https://huggingface.co/{model_id}/resolve/{revision}/"
    for filename in ("config.json", "model.safetensors.index.json"):
        with urllib.request.urlopen(base + filename, timeout=30) as response:
            (directory / filename).write_bytes(response.read())
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    for shard in sorted(set(index["weight_map"].values())):
        target = directory / "headers" / shard.replace(".safetensors", ".json")
        if target.exists():
            continue
        for attempt in range(4):
            try:
                def part(start, end, key):
                    request = urllib.request.Request(base + shard + f"?inferfit_{key}={attempt}",
                                                     headers={"Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(request, timeout=30) as response:
                        if response.status != 206:
                            raise ValueError("Server did not honor header-only Range request")
                        value = response.read(end - start + 2)
                    if len(value) != end - start + 1:
                        raise ValueError("Incorrect Range response length")
                    return value
                length = struct.unpack("<Q", part(0, 7, "length"))[0]
                if not 0 < length < 20_000_000:
                    raise ValueError("Unreasonable safetensors header length")
                target.write_text(json.dumps(json.loads(part(8, 7 + length, "header")), separators=(",", ":")))
                break
            except Exception:
                if attempt == 3:
                    raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    if args.compact:
        cache, output = map(Path, args.paths)
        output.write_text(json.dumps(compact(cache), indent=2) + "\n")
    else:
        model, revision, output = args.paths
        download(model, revision, Path(output))
