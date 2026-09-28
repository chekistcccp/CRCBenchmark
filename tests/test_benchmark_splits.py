import pytest

from crcbenchmark.splits import split_patient_cases


def test_patient_split_is_stable_disjoint_and_ignores_input_order():
    cases = [{"dataset": "MSD", "case_id": f"case{i:03d}"} for i in range(30)]
    dev, evaluation = split_patient_cases(cases, 6, 42, ["case029"])
    dev_reversed, eval_reversed = split_patient_cases(list(reversed(cases)), 6, 42, ["case029"])
    dev_ids = {r["case_id"] for r in dev}
    eval_ids = {r["case_id"] for r in evaluation}
    assert len(dev_ids) == 6 and len(eval_ids) == 24
    assert not dev_ids & eval_ids
    assert "case029" in dev_ids
    assert dev_ids == {r["case_id"] for r in dev_reversed}
    assert eval_ids == {r["case_id"] for r in eval_reversed}
    assert {r["benchmark_split"] for r in dev} == {"dev"}
    assert {r["benchmark_split"] for r in evaluation} == {"eval"}
    assert all("benchmark_split" not in r for r in cases)


def test_patient_split_rejects_duplicate_and_empty_eval():
    cases = [{"dataset": "CARE", "case_id": "same"}] * 2
    with pytest.raises(ValueError, match="Duplicate patient"):
        split_patient_cases(cases, 1, 42)
    with pytest.raises(ValueError, match="dev_size"):
        split_patient_cases(cases, 2, 42)
    with pytest.raises(ValueError, match="Required development cases are missing"):
        split_patient_cases([{"dataset": "CARE", "case_id": "a"}, {"dataset": "CARE", "case_id": "b"}], 1, 42, ["not-here"])
