#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crcbenchmark.config import load_yaml
from crcbenchmark.io import read_jsonl
from crcbenchmark.models.registry import build_model
from crcbenchmark.utils import parse_choice_response, parse_json_object_response, parse_label_list_response


def valid_answer(item, response):
    track = item["track"]
    if track == "t1":
        return parse_label_list_response(response, "ABCDEFGHIJKL", 5) is not None
    if track == "t2":
        answer = parse_json_object_response(response)
        if not isinstance(answer, dict):
            return False
        point, box = answer.get("point"), answer.get("box")
        return (
            isinstance(point, list) and len(point) == 2
            and isinstance(box, list) and len(box) == 4
            and all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1000 for x in point + box)
        )
    if track == "t3":
        return parse_label_list_response(response, "ABCDEFGHI") is not None
    return parse_choice_response(response, item["choices"]) is not None


def main():
    p = argparse.ArgumentParser(description="Load one VLM and run one real benchmark item.")
    p.add_argument("--model", required=True)
    p.add_argument("--models-config", default="configs/models.yaml")
    p.add_argument("--benchmark-config", default="configs/benchmark.yaml")
    p.add_argument("--model-root", default=None)
    p.add_argument("--manifest", default="manifests/benchmark_msd.jsonl")
    p.add_argument("--care-manifest", default=None)
    p.add_argument("--strict", action="store_true", help="Check one answer per available track for required output format")
    a = p.parse_args()

    mcfg = load_yaml(a.models_config)
    bcfg = load_yaml(a.benchmark_config)
    model_root = a.model_root or bcfg["paths"]["model_root"]

    rows = read_jsonl(a.manifest)
    if not rows:
        raise SystemExit(f"Smoke-test manifest is empty: {a.manifest}")

    print(f"[smoke] model={a.model}")
    if a.strict:
        by_track = {row["track"]: row for row in reversed(rows)}
        if a.care_manifest:
            for row in read_jsonl(a.care_manifest):
                if row["track"] == "t4":
                    by_track["t4"] = row
                    break
        items = [by_track[t] for t in ("t1", "t2", "t3", "t4", "t5") if t in by_track]
    else:
        items = [rows[0]]

    model, local = build_model(a.model, mcfg, model_root)
    print(f"[smoke] loaded={local}")

    failures = []
    for item in items:
        response = str(model.generate(item["image_path"], item["prompt"])).strip()
        valid = bool(response) and (not a.strict or valid_answer(item, response))
        print(f"[smoke] {item['track']} item={item['item_id']} format={'PASS' if valid else 'FAIL'}")
        print("[smoke] response:", response[:500])
        if not valid:
            failures.append(item["track"])
    if failures:
        raise RuntimeError(f"{a.model} failed output format on tracks: {', '.join(failures)}")
    print(f"[smoke] PASS: {a.model}, {len(items)} track samples")


if __name__ == "__main__":
    main()
