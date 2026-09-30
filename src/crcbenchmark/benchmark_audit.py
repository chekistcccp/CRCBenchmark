from __future__ import annotations

from collections import Counter


def audit_benchmark_cohorts(dev_cases: list[dict], eval_cases: list[dict],
                            dev_items: list[dict], eval_items: list[dict]) -> dict:
    """Check benchmark isolation and report eligibility before model inference."""
    case_sets = {}
    report = {}
    for split, cases, items in (("dev", dev_cases, dev_items), ("eval", eval_cases, eval_items)):
        case_ids = [row["case_id"] for row in cases]
        item_ids = [row["item_id"] for row in items]
        if len(case_ids) != len(set(case_ids)) or len(item_ids) != len(set(item_ids)):
            raise ValueError(f"Duplicate case or item ID in {split} cohort")
        if any(row.get("benchmark_split") != split for row in cases + items):
            raise ValueError(f"Unexpected benchmark_split in {split} cohort")
        case_sets[split] = set(case_ids)
        represented = {row["case_id"] for row in items}
        if represented != case_sets[split]:
            raise ValueError(f"{split} manifest case coverage differs from its cohort")

        tracks = sorted({row["track"] for row in items})
        t3_slots = Counter()
        t3_side_slots = {}
        t3_positive_counts = Counter()
        for row in items:
            if row["track"] != "t3":
                continue
            gt = row["gt"]
            labels = gt["slice_labels"]
            slot = gt.get("boundary_slot")
            if slot is None or slot not in range(len(labels)):
                raise ValueError("T3 item has no valid boundary_slot")
            source = [labels[label] for label in sorted(labels)]
            if source != list(range(source[0], source[0] + len(source))):
                raise ValueError("T3 item uses nonconsecutive source slices")
            if source[slot] != gt["boundary_slice"] or sorted(labels)[slot] not in gt["positive_labels"]:
                raise ValueError("T3 boundary does not match its image labels")
            t3_slots[slot] += 1
            side = gt.get("side", "unspecified")
            t3_side_slots.setdefault(side, Counter())[slot] += 1
            t3_positive_counts[len(gt["positive_labels"])] += 1
        if split == "eval" and len(t3_slots) < 2:
            raise ValueError("Formal T3 cohort has a fixed boundary position")
        report[split] = {
            "n_cases": len(cases),
            "n_items": len(items),
            "items_by_track": dict(sorted(Counter(row["track"] for row in items).items())),
            "patients_by_track": {
                track: len({row["case_id"] for row in items if row["track"] == track})
                for track in tracks
            },
            "t3_boundary_slots": dict(sorted(t3_slots.items())),
            "t3_boundary_slots_by_side": {side: dict(sorted(counts.items()))
                                           for side, counts in sorted(t3_side_slots.items())},
            "t3_positive_slice_counts": dict(sorted(t3_positive_counts.items())),
        }
    if case_sets["dev"] & case_sets["eval"]:
        raise ValueError("Development and evaluation patients overlap")
    return report
