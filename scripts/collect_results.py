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
            "care": {
                "tumor_label_id": 2,
                "normal_label_id": 1,
                "raw_label_rule": "0=background, 1=normal, all other foreground=tumor",
                "semantic_status": "user_confirmed_mapping",
            },
        },
        "care_mapping_source": "user_confirmed; not independently verified from release documentation",
        "entries": entries,
    }

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Collected {len(entries)} model/experiment summaries -> {out}")


if __name__ == "__main__":
    main()
