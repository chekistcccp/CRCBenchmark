#!/usr/bin/env python3
"""Paired original / text-only / neutral-image controls on development T1-T4."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crcbenchmark.config import load_yaml
from crcbenchmark.inference import manifest_fingerprint, run_manifest, validate_resume_predictions
from crcbenchmark.io import read_jsonl, write_json, write_jsonl
from crcbenchmark.publication import (CONDITIONS, evidence_items, evidence_report, file_sha256,
                                     relocate_items, runtime_provenance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evaluation-manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--model-root", default="models")
    parser.add_argument("--artifact-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    source = read_jsonl(args.manifest)
    evaluation = read_jsonl(args.evaluation_manifest)
    if not evaluation or any(r.get("benchmark_split") != "eval" for r in evaluation):
        raise ValueError("Supply the frozen evaluation manifest for isolation checks")
    if len({r["dataset"] for r in source + evaluation}) != 1:
        raise ValueError("Use a single dataset")
    if {r["case_id"] for r in source} & {r["case_id"] for r in evaluation}:
        raise ValueError("Development/evaluation patient overlap")
    output = Path(args.output_dir)
    rows = evidence_items(relocate_items(source, args.artifact_root), output / "neutral_images")
    cfg = load_yaml(args.models_config)
    spec = cfg["models"][args.model]
    metadata = {"model": args.model, "model_spec": spec, "scope": "development_only",
                "source_manifest_sha256": manifest_fingerprint(source),
                "evaluation_manifest_sha256": manifest_fingerprint(evaluation),
                "manifest_sha256": manifest_fingerprint(rows), "conditions": CONDITIONS,
                "runtime": runtime_provenance(), "model_root": str(Path(args.model_root).resolve()),
                "weight_identity_status": "requested revision and config are recorded; exact weight identity requires a separately archived weight checksum/revision"}
    path = output / "evidence_predictions.jsonl"
    existing = read_jsonl(path) if path.exists() else []
    validate_resume_predictions(existing, metadata["manifest_sha256"], path)
    expected = {r["item_id"] for r in rows}
    observed = {r["item_id"] for r in existing}
    if len(observed) != len(existing) or not observed <= expected:
        raise ValueError("Duplicate or unexpected control predictions")
    meta_path = output / "evidence_metadata.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old != json.loads(json.dumps(metadata)):
            raise ValueError("Inputs/model/runtime changed; use a fresh output directory")
    elif existing:
        raise ValueError("Existing predictions lack provenance")
    write_json(meta_path, metadata)
    write_jsonl(output / "evidence_manifest.jsonl", rows)
    if args.prepare_only:
        print(f"Prepared {len(rows)} questions, all conditions -> {output}")
        return
    if observed != expected:
        from crcbenchmark.models.registry import build_model
        model, local = build_model(args.model, cfg, args.model_root)
        import torch
        hardware = {"torch_cuda_runtime": torch.version.cuda,
                    "devices": [{"name": torch.cuda.get_device_name(i),
                                 "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory}
                                for i in range(torch.cuda.device_count())]}
        hardware_path = output / "hardware.json"
        if existing and (not hardware_path.exists() or json.loads(hardware_path.read_text()) != hardware):
            raise ValueError("GPU environment changed or missing; use a fresh output directory")
        write_json(hardware_path, hardware)
        snapshot = {name: file_sha256(Path(local) / name) for name in
                    ("config.json", "generation_config.json", "model.safetensors.index.json")
                    if (Path(local) / name).exists()}
        provenance = {"local_snapshot": str(local), "metadata_files_sha256": snapshot}
        snapshot_path = output / "snapshot_metadata.json"
        if existing and (not snapshot_path.exists() or json.loads(snapshot_path.read_text()) != provenance):
            raise ValueError("Snapshot metadata changed or missing; use a fresh output directory")
        write_json(snapshot_path, provenance)
        try:
            run_manifest(model, rows, path)
        except Exception as exc:
            raise RuntimeError("Evidence control failed; completed rows are resumable. A model/adapter error is not a scored wrong answer. Inspect whether text-only inputs are supported.") from exc
    predictions = read_jsonl(path)
    validate_resume_predictions(predictions, metadata["manifest_sha256"], path)
    report = evidence_report(rows, predictions)
    write_json(output / "evidence_summary.json", {"model": args.model, **report})
    print(f"Wrote complete paired evidence results -> {output}")


if __name__ == "__main__":
    main()
