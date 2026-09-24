#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crcbenchmark.indexing import build_case_index
from crcbenchmark.io import write_jsonl

p = argparse.ArgumentParser()
p.add_argument("--msd-root", default=None)
p.add_argument("--care-root", default=None)
p.add_argument("--care-mapping", default=None, help="Optional CSV: case_id,slice_index,npz_path,split[,spacing fields]")
p.add_argument("--output", default="manifests/cases.jsonl")
a = p.parse_args()
rows = build_case_index(a.msd_root, a.care_root, a.care_mapping)
if not rows:
    raise SystemExit("No cases indexed. Check dataset paths and CARE mapping/filename convention.")
write_jsonl(a.output, rows)
print(f"Indexed {len(rows)} patient-level records -> {a.output}")
