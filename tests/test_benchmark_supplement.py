import math

import pytest

from crcbenchmark.supplement import (fit_position_prior, format_ablation_items,
                                    paired_comparisons, score_slice_set, t3_diagnostics)


def item(case="one", split="dev", truth=None):
    return {"item_id": case, "case_id": case, "dataset": "CARE", "track": "t3",
            "benchmark_split": split, "image_path": "same.jpg", "prompt": "Return a JSON list.",
            "gt": {"slice_labels": dict(zip("ABCDEFGHI", range(9))),
                   "positive_labels": truth or list("EFGHI")}}


def test_balanced_accuracy_exposes_all_positive_shortcut():
    row = item()
    all_slices = score_slice_set(row, "ABCDEFGHI")
    assert all_slices["balanced_accuracy"] == .5
    assert all_slices["exact_set"] == 0
    assert score_slice_set(row, "EFGHI")["balanced_accuracy"] == 1
    assert score_slice_set(row, [])["balanced_accuracy"] == .5
    invalid = t3_diagnostics([row], [{"item_id": "one", "raw_response": "I think E"}], {"labels": []})[0]
    assert invalid["balanced_accuracy"] == 0 and invalid["invalid"] == 1


def test_prior_uses_dev_patients_with_equal_weight():
    rows = [item("one", truth=["A"]), {**item("one", truth=["A"]), "item_id": "second"},
            item("two", truth=["B"])]
    prior = fit_position_prior(rows)
    assert prior["prevalence"]["A"] == .5
    assert prior["prevalence"]["B"] == .5
    with pytest.raises(ValueError, match="development"):
        fit_position_prior([item(split="eval")])


def test_ablation_preserves_images_and_rejects_eval():
    source = item()
    variants = format_ablation_items([source])
    assert len(variants) == 3
    assert len({r["image_path"] for r in variants}) == 1
    assert len({r["item_id"] for r in variants}) == 3
    assert variants[0]["prompt"] == source["prompt"]
    assert "format_variant" not in source
    with pytest.raises(ValueError, match="development"):
        format_ablation_items([item(split="eval")])


def test_paired_differences_keep_patient_pairing():
    rows = [{"track": "t3", "value": 1., "base": .8},
            {"track": "t3", "value": .3, "base": .1}]
    result = paired_comparisons(rows, [("t3", "value", "base")], n_boot=100)[0]
    assert result["n"] == 2
    assert math.isclose(result["ci_low"], .2) and math.isclose(result["ci_high"], .2)
