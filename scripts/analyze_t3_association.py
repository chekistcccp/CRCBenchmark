#!/usr/bin/env python3
"""Offline, post-hoc association control using all completed dev format variants."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crcbenchmark.association import holm_adjust, patient_block_association
from crcbenchmark.config import load_yaml
from crcbenchmark.inference import manifest_fingerprint, validate_resume_predictions
from crcbenchmark.io import read_jsonl, write_json
from crcbenchmark.supplement import FORMAT_SUFFIXES, format_ablation_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="runs/protocol_v2_5")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    root = Path(args.run_root)
    models = load_yaml(args.models_config)["models"]
    entries = []
    for dataset in ("msd", "care"):
        dev = read_jsonl(root / "manifests" / f"benchmark_{dataset}_dev.jsonl")
        evaluation = read_jsonl(root / "manifests" / f"benchmark_{dataset}.jsonl")
        if {r["case_id"] for r in dev} & {r["case_id"] for r in evaluation}:
            raise ValueError("Development/evaluation patients overlap")
        expected = format_ablation_items(dev)
        fingerprint = manifest_fingerprint(expected)
        for model in models:
            folder = root / "supplement" / "format" / model / dataset
            rows = read_jsonl(folder / "format_manifest.jsonl")
            if rows != expected:
                raise ValueError(f"Manifest differs from frozen development variants: {folder}")
            metadata = json.loads((folder / "format_metadata.json").read_text(encoding="utf-8"))
            if (metadata.get("model") != model or metadata.get("model_spec") != models[model]
                    or metadata.get("manifest_fingerprint") != fingerprint
                    or metadata.get("scope") != "development_only"
                    or metadata.get("variants") != FORMAT_SUFFIXES):
                raise ValueError(f"Model/manifest provenance mismatch: {folder}")
            path = folder / "format_predictions.jsonl"
            predictions = read_jsonl(path)
            validate_resume_predictions(predictions, fingerprint, path)
            by_id = {r["item_id"]: r for r in predictions}
            if len(by_id) != len(predictions) or set(by_id) != {r["item_id"] for r in rows}:
                raise ValueError(f"Incomplete/duplicate/unexpected predictions: {path}")
            for variant in FORMAT_SUFFIXES:
                subset = [r for r in rows if r["format_variant"] == variant]
                result = patient_block_association(subset, [by_id[r["item_id"]] for r in subset],
                                                   args.permutations, args.seed)
                entries.append({"model": model, "dataset": dataset, "variant": variant,
                                "manifest_sha256": fingerprint,
                                "predictions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                **result})
    adjusted = holm_adjust([r["metrics"]["balanced_accuracy"]["p_greater"] for r in entries])
    for row, p_value in zip(entries, adjusted):
        row["metrics"]["balanced_accuracy"]["p_holm"] = p_value
    report = {
        "analysis": "posthoc_dev_patient_block_association_v1", "primary_results_modified": False,
        "inference": "BA one-sided Monte Carlo permutation; Holm across all model/dataset/variant tests",
        "family_size": len(entries), "null_interval": "2.5/97.5 percentiles of permutation null, NOT a confidence interval",
        "exchangeability": "patient response blocks within entry/exit availability strata; identical prompts",
        "limitations": "Observed answer-label association, not causal image reliance; no significance does not prove images ignored. Development cohort reused; exploratory only.",
        "entries": entries,
    }
    target = root / "supplement" / "t3_association.json"
    write_json(target, report)
    print(f"Wrote {len(entries)} association controls -> {target}")


if __name__ == "__main__":
    main()
