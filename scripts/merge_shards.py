#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crcbenchmark.io import read_jsonl, write_jsonl
p=argparse.ArgumentParser(); p.add_argument("--dir",required=True); p.add_argument("--output",required=True); a=p.parse_args()
rows={}
for f in sorted(Path(a.dir).glob("shard_*.jsonl")):
    for r in read_jsonl(f): rows[r["item_id"]]=r
write_jsonl(a.output, rows.values()); print(f"Merged {len(rows)} rows -> {a.output}")
