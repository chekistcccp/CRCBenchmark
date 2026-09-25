#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", default="results")
    p.add_argument("--output", default="results/all_experiments_summary.json")
    a = p.parse_args()

    root = Path(a.results_root)
    entries = []

    for summary_path in sorted(root.glob("*/*/summary.json")):
        rel = summary_path.relative_to(root)
        if len(rel.parts) != 3:
            continue
        model, experiment, _ = rel.parts
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as e:
            entries.append(
                {
                    "model": model,
                    "experiment": experiment,
                    "summary_path": str(summary_path),
                    "error": repr(e),
                }
            )
            continue

        entries.append(
            {
                "model": model,
                "experiment": experiment,
                "summary_path": str(summary_path),
                "summary": summary,
            }
        )

    output = {
        "experiments": {
            "msd": {
                "description": "MSD Task10 Colon, fixed label semantics",
            },
            "care_tumor1_normal2": {
                "tumor_label_id": 1,
                "normal_label_id": 2,
                "semantic_status": "unresolved_sensitivity_branch",
            },
            "care_tumor2_normal1": {
                "tumor_label_id": 2,
                "normal_label_id": 1,
                "semantic_status": "unresolved_sensitivity_branch",
            },
        },
        "warning": (
            "Do not determine CARE label semantics by choosing the branch with "
            "better model performance. Resolve semantics from annotation evidence."
        ),
        "entries": entries,
    }

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Collected {len(entries)} model/experiment summaries -> {out}")


if __name__ == "__main__":
    main()
