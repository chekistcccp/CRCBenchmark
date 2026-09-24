#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crcbenchmark.io import read_jsonl, write_jsonl, write_json
from crcbenchmark.evaluate import eval_items, eval_t5_groups, patient_aggregate
from crcbenchmark.stats import summarize_patient_rows
p=argparse.ArgumentParser()
p.add_argument("--manifest",default="manifests/benchmark_v1.jsonl")
p.add_argument("--predictions",required=True)
p.add_argument("--output-dir",required=True)
p.add_argument("--bootstrap",type=int,default=2000)
a=p.parse_args()
manifest=read_jsonl(a.manifest); preds=read_jsonl(a.predictions)
items=eval_items(manifest,preds)+eval_t5_groups(manifest,preds)
patients=patient_aggregate(items)
summary=summarize_patient_rows(patients,n_boot=a.bootstrap,seed=42)
out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
write_jsonl(out/"item_metrics.jsonl",items); write_jsonl(out/"patient_metrics.jsonl",patients); write_json(out/"summary.json",summary)
print(f"Wrote metrics for {len(patients)} patient-track groups -> {out}")
