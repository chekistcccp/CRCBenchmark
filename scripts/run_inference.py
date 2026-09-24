#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crcbenchmark.config import load_yaml
from crcbenchmark.io import read_jsonl
from crcbenchmark.models.registry import build_model
from crcbenchmark.inference import run_manifest

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--models-config", default="configs/models.yaml")
p.add_argument("--benchmark-config", default="configs/benchmark.yaml")
p.add_argument("--manifest", default="manifests/benchmark_v1.jsonl")
p.add_argument("--output", default=None)
p.add_argument("--model-root", default=None)
p.add_argument("--shard-index", type=int, default=0)
p.add_argument("--num-shards", type=int, default=1)
p.add_argument("--track", default=None, choices=[None,"t1","t2","t3","t4","t5"])
p.add_argument("--no-resume", action="store_true")
a = p.parse_args()
mcfg = load_yaml(a.models_config)
bcfg = load_yaml(a.benchmark_config)
model_root = a.model_root or bcfg["paths"]["model_root"]
rows = read_jsonl(a.manifest)
if a.track:
    rows = [r for r in rows if r["track"] == a.track]
out = a.output or str(Path(bcfg["paths"]["prediction_root"]) / a.model / f"shard_{a.shard_index:02d}.jsonl")
model, local = build_model(a.model, mcfg, model_root)
print(f"Loaded {a.model} from local ModelScope snapshot: {local}")
run_manifest(model, rows, out, a.shard_index, a.num_shards, resume=not a.no_resume)
print(out)
