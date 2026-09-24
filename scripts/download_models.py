#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crcbenchmark.config import load_yaml
from crcbenchmark.models.modelscope import ensure_model

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/models.yaml")
p.add_argument("--model-root", default="models")
p.add_argument("--models", nargs="*", default=None)
a = p.parse_args()
cfg = load_yaml(a.config)
keys = a.models or list(cfg["models"])
for key in keys:
    spec = cfg["models"][key]
    print(f"[ModelScope] {key}: {spec['modelscope_id']}")
    path = ensure_model(spec["modelscope_id"], a.model_root, spec.get("revision"))
    print(f"  -> {path}")
