"""Post-hoc diagnostics; never used to replace the frozen primary scores."""
from collections import defaultdict

import numpy as np

from .metrics import binary_prf
from .stats import bootstrap_mean_ci
from .utils import parse_label_list_response


def fit_position_prior(dev_items):
    """Patient-weighted development prevalence, threshold fixed at 0.5."""
    if not dev_items or any(r.get("benchmark_split") != "dev" for r in dev_items):
        raise ValueError("Position prior requires development items only")
    by_patient = defaultdict(list)
    for item in dev_items:
        if item["track"] == "t3":
            by_patient[item["case_id"]].append([
                float(label in item["gt"]["positive_labels"]) for label in "ABCDEFGHI"
            ])
    if not by_patient:
        raise ValueError("No development T3 items")
    prevalence = np.mean([np.mean(x, axis=0) for x in by_patient.values()], axis=0)
    return {
        "n_dev_patients": len(by_patient),
        "prevalence": dict(zip("ABCDEFGHI", map(float, prevalence))),
        "labels": [label for label, value in zip("ABCDEFGHI", prevalence) if value >= 0.5],
    }


def score_slice_set(item, selected):
    """Balanced accuracy needs both positive and negative slices in a window."""
    universe = set(item["gt"]["slice_labels"])
    truth = set(item["gt"]["positive_labels"])
    negative = universe - truth
    if not truth or not negative:
        raise ValueError("T3 balanced accuracy requires both slice classes")
    selected = set(selected) & universe
    sensitivity = len(selected & truth) / len(truth)
    specificity = len(negative - selected) / len(negative)
    return {
        "slice_f1": binary_prf(selected, truth)[2],
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "exact_set": float(selected == truth),
    }


def t3_diagnostics(items, predictions, prior):
    pred_map = {r["item_id"]: r for r in predictions}
    output = []
    for item in items:
        if item["track"] != "t3":
            continue
        parsed = parse_label_list_response(pred_map[item["item_id"]]["raw_response"], "ABCDEFGHI")
        scores = score_slice_set(item, parsed or [])
        # Malformed output is a failure, never a rewarded all-negative decision.
        if parsed is None:
            scores = {key: 0.0 for key in scores}
        row = {"track": "t3", "dataset": item["dataset"], "case_id": item["case_id"],
               "item_id": item["item_id"], "invalid": float(parsed is None), **scores}
        for name, selected in (("all", "ABCDEFGHI"), ("center", "E"), ("dev_prior", prior["labels"])):
            row.update({f"{name}_{key}": value for key, value in score_slice_set(item, selected).items()})
        output.append(row)
    return output


def paired_comparisons(patient_rows, comparisons, n_boot=2000):
    output = []
    for track, metric, baseline in comparisons:
        eligible = [r for r in patient_rows if r["track"] == track]
        values = [r[metric] - r[baseline] for r in eligible
                  if metric in r and baseline in r and np.isfinite(r[metric]) and np.isfinite(r[baseline])]
        if values:
            output.append({"track": track, "metric": metric, "baseline": baseline,
                           "direction": "model_minus_baseline", "n_total_patients": len(eligible),
                           **bootstrap_mean_ci(values, n_boot=n_boot, seed=42)})
    return output


FORMAT_SUFFIXES = {
    "current": "",
    "example_ac": ' Format example only (not an answer): ["A","C"].',
    "example_gi": ' Format example only (not an answer): ["G","I"].',
}


def format_ablation_items(manifest):
    if not manifest or any(r.get("benchmark_split") != "dev" for r in manifest):
        raise ValueError("Format ablation requires development items only")
    if len({r["item_id"] for r in manifest}) != len(manifest):
        raise ValueError("Duplicate development item IDs")
    rows = []
    for item in manifest:
        if item["track"] != "t3":
            continue
        for variant, suffix in FORMAT_SUFFIXES.items():
            rows.append({**item, "item_id": f'{item["item_id"]}:format:{variant}',
                         "source_item_id": item["item_id"], "format_variant": variant,
                         "prompt": item["prompt"] + suffix})
    if not rows:
        raise ValueError("No development T3 items")
    return rows
