#!/usr/bin/env python3
"""Development-only paired prompt experiment; all variants use identical images."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crcbenchmark.config import load_yaml
from crcbenchmark.evaluate import eval_t3, patient_aggregate
from crcbenchmark.inference import manifest_fingerprint, run_manifest, validate_resume_predictions
from crcbenchmark.io import read_jsonl, write_json, write_jsonl
from crcbenchmark.stats import summarize_patient_rows
from crcbenchmark.supplement import FORMAT_SUFFIXES, format_ablation_items, paired_comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="One dataset's development manifest")
    parser.add_argument("--model", required=True)
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--model-root", default="models")
    parser.add_argument("--output-dir", required=True, help="Dedicated supplement directory, separate from primary outputs")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    manifest = read_jsonl(args.manifest)
    if len({r["dataset"] for r in manifest}) != 1:
        raise ValueError("Use a single dataset per ablation run")
    rows = format_ablation_items(manifest)
    output = Path(args.output_dir)
    if output.resolve() == Path(args.manifest).resolve().parent:
        raise ValueError("Use a dedicated output directory")
    path = output / "format_predictions.jsonl"
    fingerprint = manifest_fingerprint(rows)
    existing = read_jsonl(path) if path.exists() else []
    validate_resume_predictions(existing, fingerprint, path)
    expected = {r["item_id"] for r in rows}
    observed = {r["item_id"] for r in existing}
    if len(observed) != len(existing) or not observed <= expected:
        raise ValueError("Duplicate or unexpected ablation predictions")
    metadata = {"model": args.model, "model_spec": load_yaml(args.models_config)["models"][args.model],
                "model_root": str(Path(args.model_root).resolve()),
                "manifest_fingerprint": fingerprint, "scope": "development_only",
                "variants": FORMAT_SUFFIXES, "selection_policy": "report all variants; no automatic winner"}
    metadata_path = output / "format_metadata.json"
    if metadata_path.exists():
        import json
        if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Ablation model/configuration changed; use a fresh output directory")
    elif existing:
        raise ValueError("Existing predictions lack model provenance; use a fresh output directory")
    write_json(metadata_path, metadata)
    write_jsonl(output / "format_manifest.jsonl", rows)
    if args.prepare_only:
        print(f"Prepared {len(rows)} development questions -> {output}")
        return
    if observed != expected:
        from crcbenchmark.models.registry import build_model
        model, _ = build_model(args.model, load_yaml(args.models_config), args.model_root)
        run_manifest(model, rows, path)
    predictions = read_jsonl(path)
    validate_resume_predictions(predictions, fingerprint, path)
    if len(predictions) != len(rows) or {r["item_id"] for r in predictions} != expected:
        raise ValueError("Incomplete ablation predictions")
    by_id = {r["item_id"]: r for r in predictions}
    report = {"model": args.model, "scope": "development_only", "variants": {}, "paired_differences": []}
    per_variant = {}
    for variant in FORMAT_SUFFIXES:
        scored = [{"track": "t3", "dataset": item["dataset"], "case_id": item["case_id"],
                   **eval_t3(item, by_id[item["item_id"]])} for item in rows if item["format_variant"] == variant]
        for row in scored:
            row["format_valid"] = 1 - row["invalid"]
        patients = patient_aggregate(scored)
        per_variant[variant] = {r["case_id"]: r for r in patients}
        report["variants"][variant] = summarize_patient_rows(patients)
    for variant in FORMAT_SUFFIXES:
        if variant == "current":
            continue
        paired = [{**row, "current_valid": per_variant["current"][case]["format_valid"],
                   "current_f1": per_variant["current"][case]["slice_f1"]}
                  for case, row in per_variant[variant].items()]
        report["paired_differences"].append({"variant": variant, "comparisons": paired_comparisons(
            paired, [("t3", "format_valid", "current_valid"), ("t3", "slice_f1", "current_f1")])})
    write_json(output / "format_summary.json", report)
    print(f"Wrote paired development format results -> {output}")


if __name__ == "__main__":
    main()
