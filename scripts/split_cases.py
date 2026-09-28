#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crcbenchmark.io import read_jsonl, write_jsonl
from crcbenchmark.splits import split_patient_cases


def main():
    parser = argparse.ArgumentParser(description="Freeze patient-disjoint benchmark dev/eval cohorts.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--dev-size", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--required-dev-case", action="append", default=[])
    parser.add_argument("--dev-output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--split-report", required=True)
    args = parser.parse_args()

    cases = read_jsonl(args.cases)
    dev, evaluation = split_patient_cases(cases, args.dev_size, args.seed, args.required_dev_case)
    write_jsonl(args.dev_output, dev)
    write_jsonl(args.eval_output, evaluation)
    report = {
        "seed": args.seed,
        "source_cases": args.cases,
        "required_dev_cases": args.required_dev_case,
        "dev": sorted(r["case_id"] for r in dev),
        "eval": sorted(r["case_id"] for r in evaluation),
    }
    output = Path(args.split_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Patient split: {len(dev)} dev, {len(evaluation)} eval -> {output}")


if __name__ == "__main__":
    main()
