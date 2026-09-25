#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crcbenchmark.config import load_yaml
from crcbenchmark.io import read_jsonl
from crcbenchmark.models.registry import build_model


def main():
    p = argparse.ArgumentParser(description="Load one VLM and run one real benchmark item.")
    p.add_argument("--model", required=True)
    p.add_argument("--models-config", default="configs/models.yaml")
    p.add_argument("--benchmark-config", default="configs/benchmark.yaml")
    p.add_argument("--model-root", default=None)
    p.add_argument("--manifest", default="manifests/benchmark_msd.jsonl")
    a = p.parse_args()

    mcfg = load_yaml(a.models_config)
    bcfg = load_yaml(a.benchmark_config)
    model_root = a.model_root or bcfg["paths"]["model_root"]

    rows = read_jsonl(a.manifest)
    if not rows:
        raise SystemExit(f"Smoke-test manifest is empty: {a.manifest}")

    item = rows[0]
    print(f"[smoke] model={a.model}")
    print(f"[smoke] item={item['item_id']}")
    print(f"[smoke] image={item['image_path']}")

    model, local = build_model(a.model, mcfg, model_root)
    print(f"[smoke] loaded={local}")

    response = model.generate(
        item["image_path"],
        item["prompt"],
        max_new_tokens=32,
    )
    response = str(response).strip()
    if not response:
        raise RuntimeError(f"{a.model} returned an empty response.")

    print("[smoke] response:", response[:500])
    print(f"[smoke] PASS: {a.model}")


if __name__ == "__main__":
    main()
