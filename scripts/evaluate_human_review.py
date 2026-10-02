#!/usr/bin/env python3
"""Score complete reader files against the frozen key; never filter main scores."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crcbenchmark.inference import manifest_fingerprint, validate_resume_predictions
from crcbenchmark.io import read_jsonl, write_json
from crcbenchmark.publication import file_sha256, score_dev_items, validity_agreement
from crcbenchmark.stats import summarize_patient_rows


def validate_reader(rows, key):
    expected = {r["review_id"] for r in key["items"]}
    by_id = {r["review_id"]: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != expected:
        raise ValueError("Reader must answer every selected question exactly once")
    readers = {r.get("reviewer_id") for r in rows}
    if len(readers) != 1 or not next(iter(readers)):
        raise ValueError("Use one nonempty reviewer_id per file")
    for row in rows:
        if row.get("packet_sha256") != key["packet_sha256"]:
            raise ValueError("Reader packet fingerprint mismatch")
        if row.get("task_valid") not in ("yes", "no", "uncertain") or not row.get("raw_response", "").strip():
            raise ValueError("Incomplete reader response or validity rating")
    return next(iter(readers)), by_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", default="runs/protocol_v2_5/supplement/human_review/private/review_key.json")
    parser.add_argument("--readers", nargs="+", required=True)
    parser.add_argument("--output", default="runs/protocol_v2_5/supplement/human_review/human_summary.json")
    parser.add_argument("--evidence-root", help="Optional completed evidence-control root for same-subset model comparison")
    args = parser.parse_args()
    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    if key.get("scope") != "development_only":
        raise ValueError("Expected development review key")
    items = [r["item"] for r in key["items"]]
    for item in items:
        if file_sha256(item["image_path"]) != item["source_image_sha256"]:
            raise ValueError("Reader image changed since packet preparation")
        if item.get("gt_mask_path") and file_sha256(item["gt_mask_path"]) != item["gt_mask_sha256"]:
            raise ValueError("Reader scoring mask changed")
    report = {"scope": "development_only_posthoc", "packet_sha256": key["packet_sha256"],
              "n_items": len(items), "readers": [], "agreement": [], "matched_subset_models": [],
              "limitations": "A small development feasibility study, not a population human ceiling; no main item exclusions. Kappa concerns item-level task-validity ratings, not diagnosis agreement."}
    observed = []
    for path in args.readers:
        reader, answers = validate_reader(read_jsonl(path), key)
        if reader in [x[0] for x in observed]:
            raise ValueError("Reviewer IDs must be distinct across files")
        predictions = [{"item_id": r["item"]["item_id"], "raw_response": answers[r["review_id"]]["raw_response"]}
                       for r in key["items"]]
        report["readers"].append({"reviewer_id": reader, "summary": summarize_patient_rows(score_dev_items(items, predictions)),
                                  "task_valid_counts": {v: sum(a["task_valid"] == v for a in answers.values()) for v in ("yes", "no", "uncertain")}})
        observed.append((reader, answers))
    for i, (first, a) in enumerate(observed):
        for second, b in observed[i + 1:]:
            ids = [r["review_id"] for r in key["items"]]
            report["agreement"].append({"reviewers": [first, second], **validity_agreement(
                [a[k]["task_valid"] for k in ids], [b[k]["task_valid"] for k in ids])})
    if args.evidence_root:
        root = Path(args.evidence_root)
        for model in sorted(p.name for p in root.iterdir() if p.is_dir()):
            for dataset in sorted({r["dataset"].lower() for r in items}):
                folder = root / model / dataset
                metadata = json.loads((folder / "evidence_metadata.json").read_text())
                predictions = read_jsonl(folder / "evidence_predictions.jsonl")
                validate_resume_predictions(predictions, metadata["manifest_sha256"], folder)
                expected = read_jsonl(folder / "evidence_manifest.jsonl")
                if manifest_fingerprint(expected) != metadata["manifest_sha256"] or metadata.get("model") != model:
                    raise ValueError("Evidence manifest/model provenance mismatch")
                if len(predictions) != len(expected) or {r["item_id"] for r in predictions} != {r["item_id"] for r in expected}:
                    raise ValueError("Incomplete evidence predictions")
                original = {r["source_item_id"]: r for r in expected if r["evidence_condition"] == "original"}
                by_id = {r["item_id"]: r for r in predictions}
                subset = [r for r in items if r["dataset"].lower() == dataset]
                paired = []
                for item in subset:
                    source = original[item["item_id"]]
                    if (source["source_image_sha256"] != item["source_image_sha256"]
                            or source.get("gt_mask_sha256") != item.get("gt_mask_sha256")
                            or source["prompt"] != item["prompt"] or source["gt"] != item["gt"]):
                        raise ValueError("Human/model subset inputs differ")
                    paired.append({**by_id[source["item_id"]], "item_id": item["item_id"]})
                report["matched_subset_models"].append({"model": model, "dataset": dataset,
                    "summary": summarize_patient_rows(score_dev_items(subset, paired))})
    write_json(args.output, report)
    print(f"Scored {len(observed)} readers -> {args.output}")


if __name__ == "__main__":
    main()
