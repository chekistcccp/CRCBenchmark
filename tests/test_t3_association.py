import json

import pytest

from crcbenchmark.association import holm_adjust, patient_block_association


def cohort(constant=None, perfect=True):
    rows, predictions = [], []
    for case in range(12):
        for side in ("entry", "exit"):
            truth = list("ABCDEFGHI")[case % 6 + 1:] if side == "entry" else list("ABCDEFGHI")[:case % 6 + 1]
            item_id = f"{case}:{side}"
            rows.append({"item_id": item_id, "case_id": str(case), "dataset": "MSD", "track": "t3",
                         "benchmark_split": "dev", "prompt": "Same prompt",
                         "gt": {"side": side, "slice_labels": list("ABCDEFGHI"), "positive_labels": truth}})
            response = truth if perfect else (list("EFGHI") if side == "entry" else list("ABCD"))
            predictions.append({"item_id": item_id, "raw_response": json.dumps(constant if constant is not None else response)})
    return rows, predictions


def test_perfect_alignment_is_detected_and_reproducible():
    rows, predictions = cohort()
    result = patient_block_association(rows, predictions, 999)
    metric = result["metrics"]["balanced_accuracy"]
    assert metric["matched_mean"] == 1
    assert metric["p_greater"] <= .01
    assert metric["matched_minus_permutation"] > .1
    assert result == patient_block_association(list(reversed(rows)), list(reversed(predictions)), 999)


@pytest.mark.parametrize("constant", [list("GI"), list("ABCDEFGHI"), []])
def test_constant_answers_have_no_association(constant):
    result = patient_block_association(*cohort(constant=constant), n_permutations=99)
    metric = result["metrics"]["balanced_accuracy"]
    assert metric["matched_minus_permutation"] == pytest.approx(0)
    assert metric["p_greater"] == 1


def test_side_only_shortcut_is_preserved_by_null():
    result = patient_block_association(*cohort(perfect=False), n_permutations=99)
    assert result["metrics"]["balanced_accuracy"]["p_greater"] == 1


def test_invalid_is_failure_and_singleton_is_reported():
    rows, predictions = cohort()
    rows.pop()
    predictions.pop()
    for prediction in predictions:
        prediction["raw_response"] = "Not JSON"
    result = patient_block_association(rows, predictions, 99)
    assert result["n_patients"] == 12 and result["n_eligible_patients"] == 11
    assert result["excluded_patients"] == ["11"]
    assert result["metrics"]["balanced_accuracy"]["matched_mean"] == 0


def test_reject_bad_inputs():
    rows, predictions = cohort()
    with pytest.raises(ValueError, match="completely"):
        patient_block_association(rows, predictions[:-1])
    rows[0]["prompt"] = "Different prompt"
    with pytest.raises(ValueError, match="identical"):
        patient_block_association(rows, predictions)
    rows[0]["benchmark_split"] = "eval"
    with pytest.raises(ValueError, match="development"):
        patient_block_association(rows, predictions)


def test_holm_preserves_original_order_and_monotonicity():
    assert holm_adjust([.04, .001, .03]) == pytest.approx([.06, .003, .06])
