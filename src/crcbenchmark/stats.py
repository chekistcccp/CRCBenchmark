from __future__ import annotations
from collections import defaultdict
import numpy as np


def bootstrap_mean_ci(values, n_boot=2000, seed=42, alpha=0.05):
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        means[i] = sample.mean()
    lo, hi = np.quantile(means, [alpha/2, 1-alpha/2])
    return {"mean": float(vals.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": int(vals.size)}


def summarize_patient_rows(rows: list[dict], n_boot=2000, seed=42) -> list[dict]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["track"], r["dataset"])].append(r)
    output = []
    for (track, dataset), rs in sorted(grouped.items()):
        keys = sorted({k for r in rs for k, v in r.items() if isinstance(v, (int, float)) and k not in {"n_items"}})
        for k in keys:
            vals = [float(r[k]) for r in rs if isinstance(r.get(k), (int, float))]
            s = bootstrap_mean_ci(vals, n_boot=n_boot, seed=seed)
            output.append({"track": track, "dataset": dataset, "metric": k, **s})
    return output
