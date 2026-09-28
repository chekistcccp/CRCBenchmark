#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crcbenchmark.config import load_yaml
from crcbenchmark.io import read_jsonl, write_jsonl, write_json
from crcbenchmark.inference import validate_evaluation_inputs
from crcbenchmark.evaluate import eval_items, eval_t5_groups, patient_aggregate
from crcbenchmark.stats import summarize_patient_rows


EXPERIMENTS = (
    "msd",
    "care",
)


def evaluate_one(manifest_path: Path, pred_path: Path, out_dir: Path, bootstrap: int):
    manifest = read_jsonl(manifest_path)
    preds = read_jsonl(pred_path)

    validate_evaluation_inputs(manifest, preds, pred_path)

    items = eval_items(manifest, preds) + eval_t5_groups(manifest, preds)
    patients = patient_aggregate(items)
    summary = summarize_patient_rows(patients, n_boot=bootstrap, seed=42)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "item_metrics.jsonl", items)
    write_jsonl(out_dir / "patient_metrics.jsonl", patients)
    write_json(out_dir / "summary.json", summary)

    print(
        f"[OK] {pred_path}: {len(preds)} predictions -> "
        f"{len(patients)} patient-track groups -> {out_dir}"
    )


def main():
    p = argparse.ArgumentParser(
        description="Re-evaluate all existing predictions without loading any VLM."
    )
    p.add_argument("--models-config", default="configs/models.yaml")
    p.add_argument("--predictions-root", default="predictions")
    p.add_argument("--results-root", default="results")
    p.add_argument("--manifest-root", default="manifests")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--skip-care", action="store_true", help="Evaluate MSD only")
    a = p.parse_args()

    cfg = load_yaml(a.models_config)
    failures = []

    for model in cfg["models"]:
        for exp in EXPERIMENTS[:1] if a.skip_care else EXPERIMENTS:
            manifest = Path(a.manifest_root) / f"benchmark_{exp}.jsonl"
            pred = Path(a.predictions_root) / model / exp / "all.jsonl"
            out = Path(a.results_root) / model / exp

            if not manifest.exists():
                failures.append(f"{model}/{exp}: missing manifest {manifest}")
                continue
            if not pred.exists():
                failures.append(f"{model}/{exp}: missing predictions {pred}")
                continue

            try:
                evaluate_one(manifest, pred, out, a.bootstrap)
            except Exception as e:
                failures.append(f"{model}/{exp}: {e!r}")

    if failures:
        print("\nEvaluation recovery completed with failures:", file=sys.stderr)
        for x in failures:
            print("  - " + x, file=sys.stderr)
        raise SystemExit(1)

    print("\nAll existing predictions were evaluated successfully.")


if __name__ == "__main__":
    main()
