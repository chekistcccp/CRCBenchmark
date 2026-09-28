from __future__ import annotations

import hashlib


def split_patient_cases(records: list[dict], dev_size: int, seed: int, required_dev_case_ids=()):
    """Make a stable patient-disjoint development and evaluation split."""
    if not 0 < dev_size < len(records):
        raise ValueError(f"dev_size must be between 1 and {len(records) - 1}")
    keys = [(str(r["dataset"]), str(r["case_id"])) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate patient keys in case index")
    required = set(required_dev_case_ids)
    available = {case_id for _, case_id in keys}
    if not required <= available:
        raise ValueError(f"Required development cases are missing: {sorted(required - available)}")
    if len(required) > dev_size:
        raise ValueError("Required development cases exceed dev_size")

    def order_key(record):
        key = f'{seed}\0{record["dataset"]}\0{record["case_id"]}'.encode("utf-8")
        return hashlib.sha256(key).hexdigest()

    ranked = sorted(records, key=lambda r: (order_key(r), r["dataset"], r["case_id"]))
    dev_keys = {(r["dataset"], r["case_id"]) for r in ranked if r["case_id"] in required}
    for record in ranked:
        if len(dev_keys) == dev_size:
            break
        dev_keys.add((record["dataset"], record["case_id"]))
    dev, evaluation = [], []
    for record in records:
        row = dict(record)
        row["benchmark_split"] = "dev" if (row["dataset"], row["case_id"]) in dev_keys else "eval"
        (dev if row["benchmark_split"] == "dev" else evaluation).append(row)
    return dev, evaluation
