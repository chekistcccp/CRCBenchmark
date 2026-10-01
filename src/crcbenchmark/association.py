"""Exploratory patient-block response/label association, development only."""
from collections import Counter, defaultdict

import numpy as np

from .supplement import score_slice_set
from .utils import parse_label_list_response


def holm_adjust(values):
    """Holm family-wise correction, returned in original order."""
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("Invalid p-values")
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        values[order] * np.arange(len(values), 0, -1)))
    return adjusted.tolist()


def patient_block_association(items, predictions, n_permutations=9999, seed=20261002):
    """Permute complete patient response blocks within observed side-pattern strata.

    Prompts must be identical. No GT-dependent donor selection. Full uniform
    permutations include fixed points; the null is conditional exchangeability
    of patient response blocks, not a causal intervention on the model.
    """
    if not items or any(r.get("benchmark_split") != "dev" or r["track"] != "t3" for r in items):
        raise ValueError("Association requires development T3 items only")
    if len({r["dataset"] for r in items}) != 1 or len({r["prompt"] for r in items}) != 1:
        raise ValueError("Use one dataset and identical prompts")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    by_id = {r["item_id"]: r for r in predictions}
    if (len(by_id) != len(predictions) or len({r["item_id"] for r in items}) != len(items)
            or set(by_id) != {r["item_id"] for r in items}):
        raise ValueError("Predictions must uniquely and completely match items")
    patients = defaultdict(dict)
    parsed = {}
    counts = Counter()
    for row in items:
        case, side = row["case_id"], row["gt"]["side"]
        if side not in ("entry", "exit") or side in patients[case]:
            raise ValueError("Expected at most one entry and exit per patient")
        if set(row["gt"]["slice_labels"]) != set("ABCDEFGHI"):
            raise ValueError("Expected nine labels A-I")
        score_slice_set(row, [])  # Validate both classes even if responses are invalid.
        patients[case][side] = row
        answer = parse_label_list_response(by_id[row["item_id"]]["raw_response"], "ABCDEFGHI")
        parsed[row["item_id"]] = answer
        counts["INVALID" if answer is None else ",".join(sorted(answer))] += 1
    strata = defaultdict(list)
    for case in sorted(patients):
        strata[tuple(sorted(patients[case]))].append(case)
    eligible = sum(len(cases) for cases in strata.values() if len(cases) >= 2)
    if eligible == 0:
        raise ValueError("No exchangeable patient stratum")
    metrics = ("balanced_accuracy", "slice_f1", "exact_set")
    observed = np.zeros(len(metrics))
    null = np.zeros((n_permutations, len(metrics)))
    rng = np.random.default_rng(seed)
    for sides, cases in sorted(strata.items()):
        n = len(cases)
        if n < 2:
            continue
        scores = np.zeros((n, n, len(metrics)))
        for target, target_case in enumerate(cases):
            for donor, donor_case in enumerate(cases):
                for side in sides:
                    answer = parsed[patients[donor_case][side]["item_id"]]
                    if answer is not None:
                        result = score_slice_set(patients[target_case][side], answer)
                        scores[target, donor] += [result[key] / len(sides) for key in metrics]
        observed += scores[np.arange(n), np.arange(n)].sum(axis=0)
        for index in range(n_permutations):
            null[index] += scores[np.arange(n), rng.permutation(n)].sum(axis=0)
    observed /= eligible
    null /= eligible
    results = {}
    for column, key in enumerate(metrics):
        values = null[:, column]
        results[key] = {
            "matched_mean": float(observed[column]),
            "permutation_mean": float(values.mean()),
            "matched_minus_permutation": float(observed[column] - values.mean()),
            "null_q025": float(np.quantile(values, .025)),
            "null_q975": float(np.quantile(values, .975)),
        }
        # Only BA is an inferential endpoint; F1/exact-set are descriptive.
        if key == "balanced_accuracy":
            results[key]["p_greater"] = float(
                (1 + np.count_nonzero(values >= observed[column] - 1e-12)) / (n_permutations + 1))
    return {"n_patients": len(patients), "n_eligible_patients": eligible,
            "excluded_patients": [case for cases in strata.values() if len(cases) < 2 for case in cases],
            "strata": {"+".join(sides): len(cases) for sides, cases in sorted(strata.items())},
            "n_items": len(items), "response_counts_all_items": dict(sorted(counts.items())),
            "n_permutations": n_permutations, "seed": seed, "metrics": results}
