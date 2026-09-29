#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crcbenchmark.benchmark_audit import audit_benchmark_cohorts
from crcbenchmark.io import read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(description="Audit patient isolation and track coverage before inference.")
    parser.add_argument("--cases-dev", required=True)
    parser.add_argument("--cases-eval", required=True)
    parser.add_argument("--manifest-dev", required=True)
    parser.add_argument("--manifest-eval", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit_benchmark_cohorts(
        read_jsonl(args.cases_dev), read_jsonl(args.cases_eval),
        read_jsonl(args.manifest_dev), read_jsonl(args.manifest_eval),
    )
    write_json(args.output, report)
    print(f"Audited benchmark cohorts and track coverage -> {args.output}")
    for split, data in report.items():
        print(f"  {split}: {data['n_cases']} cases, {data['n_items']} items; "
              f"patients/track={data['patients_by_track']}; T3 slots={data['t3_boundary_slots']}")


if __name__ == "__main__":
    main()
