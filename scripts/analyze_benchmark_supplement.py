#!/usr/bin/env python3
"""Analyze existing predictions without rebuilding images or loading models."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crcbenchmark.benchmark_audit import audit_benchmark_cohorts
from crcbenchmark.evaluate import patient_aggregate
from crcbenchmark.inference import manifest_fingerprint, validate_evaluation_inputs
from crcbenchmark.io import read_jsonl, write_json
from crcbenchmark.stats import summarize_patient_rows
from crcbenchmark.supplement import fit_position_prior, paired_comparisons, t3_diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="runs/protocol_v2_5")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    root = Path(args.run_root)
    output = {"analysis": "posthoc_v1", "primary_results_modified": False,
              "confidence_intervals": "patient bootstrap, 95%, exploratory, no multiplicity correction",
              "datasets": {}, "entries": []}
    for dataset in ("msd", "care"):
        folder = root / "manifests"
        evaluation = read_jsonl(folder / f"benchmark_{dataset}.jsonl")
        dev = read_jsonl(folder / f"benchmark_{dataset}_dev.jsonl")
        audit = audit_benchmark_cohorts(read_jsonl(folder / f"cases_{dataset}_dev.jsonl"),
                                      read_jsonl(folder / f"cases_{dataset}_eval.jsonl"), dev, evaluation)
        prior = fit_position_prior(dev)
        output["datasets"][dataset] = {"coverage": audit, "dev_position_prior": prior,
                                         "dev_manifest_fingerprint": manifest_fingerprint(dev),
                                         "manifest_fingerprint": manifest_fingerprint(evaluation)}
        paths = sorted((root / "predictions").glob(f"*/{dataset}/all.jsonl"))
        if not paths:
            raise ValueError(f"No predictions for {dataset}")
        for path in paths:
            predictions = read_jsonl(path)
            validate_evaluation_inputs(evaluation, predictions, path)
            patients = patient_aggregate(t3_diagnostics(evaluation, predictions, prior))
            comparisons = [("t3", metric, f"{baseline}_{metric}")
                           for metric in ("slice_f1", "balanced_accuracy", "exact_set")
                           for baseline in ("all", "center", "dev_prior")]
            output["entries"].append({"model": path.parts[-3], "dataset": dataset,
                                      "summary": summarize_patient_rows(patients, args.bootstrap),
                                      "paired_differences": paired_comparisons(patients, comparisons, args.bootstrap)})
    target = root / "supplement" / "diagnostics.json"
    write_json(target, output)
    print(f"Wrote {len(output['entries'])} model/dataset diagnostics -> {target}")


if __name__ == "__main__":
    main()
