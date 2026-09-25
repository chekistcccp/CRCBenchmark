#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crcbenchmark.config import load_yaml
from crcbenchmark.models.modelscope import ensure_model


p = argparse.ArgumentParser(
    description="Download ModelScope snapshots for all configured benchmark models."
)
p.add_argument("--config", default="configs/models.yaml")
p.add_argument("--model-root", default="models")
p.add_argument(
    "--models",
    nargs="*",
    default=None,
    help="Optional debugging subset. Omit to download ALL configured models.",
)
p.add_argument("--retries", type=int, default=3)
p.add_argument("--retry-wait", type=float, default=10.0)
a = p.parse_args()

cfg = load_yaml(a.config)
all_keys = list(cfg["models"])
keys = a.models or all_keys

unknown = sorted(set(keys) - set(all_keys))
if unknown:
    raise SystemExit(f"Unknown model keys: {unknown}")

root = Path(a.model_root)
root.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("ModelScope automatic model download")
print("Model root:", root.resolve())
print("Models:", ", ".join(keys))
print("=" * 72)

failures = []
downloaded = []

for i, key in enumerate(keys, 1):
    spec = cfg["models"][key]
    model_id = spec["modelscope_id"]
    revision = spec.get("revision")
    print(f"\n[{i}/{len(keys)}] {key}")
    print(f"  ModelScope ID: {model_id}")

    last_error = None
    for attempt in range(1, max(a.retries, 1) + 1):
        try:
            path = ensure_model(model_id, root, revision)
            print(f"  READY -> {path}")
            downloaded.append((key, str(path)))
            last_error = None
            break
        except Exception as e:
            last_error = e
            print(f"  attempt {attempt}/{max(a.retries,1)} failed: {e}", file=sys.stderr)
            if attempt < max(a.retries, 1):
                time.sleep(max(a.retry_wait, 0))

    if last_error is not None:
        failures.append((key, model_id, repr(last_error)))

print("\n" + "=" * 72)
print(f"Ready models: {len(downloaded)}/{len(keys)}")
for key, path in downloaded:
    print(f"  OK  {key}: {path}")

if failures:
    print("Failed models:", file=sys.stderr)
    for key, model_id, err in failures:
        print(f"  FAIL {key} ({model_id}): {err}", file=sys.stderr)
    raise SystemExit(1)

print("All requested model weights are ready.")
