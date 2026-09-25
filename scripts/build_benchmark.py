#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crcbenchmark.config import load_yaml
from crcbenchmark.io import read_jsonl, write_jsonl, write_json
from crcbenchmark.cases import load_case
from crcbenchmark.tracks import build_all_tracks


p = argparse.ArgumentParser()
p.add_argument("--cases", default="manifests/cases.jsonl")
p.add_argument("--config", default="configs/benchmark.yaml")
p.add_argument("--output", default="manifests/benchmark_v1.jsonl")
p.add_argument("--artifact-root", default=None)
p.add_argument("--limit", type=int, default=None)
p.add_argument("--experiment-name", default="benchmark_v1")
a = p.parse_args()

cfg = load_yaml(a.config)
records = read_jsonl(a.cases)
if a.limit:
    records = records[: a.limit]

out_root = Path(a.artifact_root or cfg["paths"]["artifact_root"])
rows = []
failures = []

for rec in tqdm(records, desc=f"Building {a.experiment_name}"):
    try:
        case = load_case(rec)
        built = build_all_tracks(case, out_root, cfg["tracks"], int(cfg["seed"]))
        for item in built:
            item["experiment_name"] = a.experiment_name
            if rec.get("dataset", "").upper() == "CARE":
                item["care_label_mapping"] = {
                    "tumor_label_id": rec.get("tumor_label_id"),
                    "normal_label_id": rec.get("normal_label_id"),
                    "canonicalization": "raw_label_gt_2_to_2",
                    "semantic_status": "sensitivity_branch_unresolved",
                }
                item["care_split"] = rec.get("split")
                item["care_index_source"] = rec.get("care_index_source")
        rows.extend(built)
    except Exception as e:
        failures.append(
            {
                "experiment_name": a.experiment_name,
                "dataset": rec.get("dataset"),
                "case_id": rec.get("case_id"),
                "error": repr(e),
            }
        )

write_jsonl(a.output, rows)
write_json(Path(a.output).with_suffix(".failures.json"), failures)
print(
    f"Built {len(rows)} benchmark items from {len(records)} cases "
    f"for experiment={a.experiment_name}. Failures={len(failures)}"
)
