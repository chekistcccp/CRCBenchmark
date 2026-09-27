import numpy as np
import pytest

from PIL import Image

from crcbenchmark.evaluate import eval_t1, eval_t2, eval_t4, eval_t5_groups
from crcbenchmark.inference import manifest_fingerprint, validate_resume_predictions
from crcbenchmark.preprocess import VolumeCase
from crcbenchmark.tracks import build_t1
from crcbenchmark.utils import parse_choice_response, parse_json_object_response, parse_label_list_response, parse_t2_response


def test_strict_label_parsing_rejects_reasoning_and_accepts_final_list():
    assert parse_label_list_response('The answer is A or B', "ABCDEFGHIJKL") is None
    assert parse_label_list_response('```json\n["A", "C"]\n```<turn|>', "ABCDEFGHIJKL") == ["A", "C"]
    assert parse_label_list_response('Reasoning.\n```json\n["A", "C"]\n```\nEnd.', "ABCDEFGHIJKL") == ["A", "C"]
    assert parse_label_list_response('["A", "A"]', "ABCDEFGHIJKL") is None
    item = {"gt": {"negative_only": False, "positive_labels": ["A"]}}
    out = eval_t1(item, {"raw_response": "The answer is A or B", "parsed": None})
    assert out["invalid"] == 1 and out["hit_3"] == 0


def test_choice_parsing_rejects_explanations_but_accepts_transport_suffix():
    assert parse_choice_response("B<turn|>", ["A", "B"]) == "B"
    assert parse_choice_response("PRESENT<turn|>", ["PRESENT", "ABSENT"]) == "PRESENT"
    assert parse_choice_response("A looks suspicious but B wins", ["A", "B"]) is None
    assert parse_choice_response("The answer is <|begin_of_box|>A<|end_of_box|>.", ["A", "B"]) == "A"
    assert parse_choice_response("<|begin_of_box|>A<|end_of_box|> and <|begin_of_box|>B<|end_of_box|>", ["A", "B"]) is None
    item = {"choices": ["A", "B"], "gt": {"answer": "B"}}
    assert eval_t4(item, {"choice": "A", "raw_response": "B<turn|>"})["pairwise_acc"] == 1


def test_t2_requires_numeric_coordinates_and_structured_answer(tmp_path):
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[4:6, 4:6] = 255
    path = tmp_path / "mask.png"
    Image.fromarray(mask).save(path)
    item = {"gt_mask_path": str(path), "gt": {"point_norm": [500, 500], "box_norm": [400, 400, 600, 600]}}
    good = '{"point":[500,500],"box":[400,400,600,600]}'
    assert parse_json_object_response(f'```json\n{good}\n```') == {"point": [500, 500], "box": [400, 400, 600, 600]}
    assert eval_t2(item, {"raw_response": good})["pointing_acc"] == 1
    assert eval_t2(item, {"raw_response": '{"point":["bad",500],"box":[400,400,600,600]}'})["invalid"] == 1
    assert parse_t2_response('The location is <|begin_of_box|>{"point":[500,500],"box":[400,400,600,600]}<|end_of_box|>.') is not None
    assert parse_t2_response('{"point":[500,500],"box":[500,500,500,500]}') is None
    assert eval_t2(item, {"raw_response": '{"point":[500,500],"box":[500,500,500,500]}'})["invalid"] == 1
    assert parse_label_list_response('Visible slices are <|begin_of_box|>["B","C"]<|end_of_box|>.', "ABCDEFGHI") == ["B", "C"]


def test_t5_invalid_and_unrecognized_original_do_not_create_faithfulness_score():
    def item(condition):
        return {"item_id": condition, "group_id": "g", "track": "t5", "dataset": "MSD", "case_id": "case", "condition": condition, "choices": ["PRESENT", "ABSENT"]}
    manifest = [item(x) for x in ("original", "lesion_gaussian", "control_gaussian")]
    preds = [
        {"item_id": "original", "raw_response": "ABSENT", "choice": "ABSENT", "score_mode": "decision"},
        {"item_id": "lesion_gaussian", "raw_response": "ABSENT", "choice": "ABSENT", "score_mode": "decision"},
        {"item_id": "control_gaussian", "raw_response": "analysis PRESENT", "choice": "PRESENT", "score_mode": "decision"},
    ]
    result = eval_t5_groups(manifest, preds)[0]
    assert result["original_present"] == 0
    assert result["valid_fraction"] == 2 / 3
    assert np.isnan(result["faithfulness_gap"])
    assert np.isnan(result["decision_gap"])
    preds = [
        {"item_id": "original", "raw_response": "PRESENT", "score_mode": "decision"},
        {"item_id": "lesion_gaussian", "raw_response": "ABSENT", "score_mode": "decision"},
        {"item_id": "control_gaussian", "raw_response": "PRESENT", "score_mode": "decision"},
    ]
    result = eval_t5_groups(manifest, preds)[0]
    assert result["decision_gap"] == 1
    assert np.isnan(result["faithfulness_gap"])


def test_t1_short_negative_pool_keeps_case_buildable(tmp_path):
    image = np.zeros((11, 8, 8), dtype=np.float32)
    label = np.zeros_like(image, dtype=np.uint8)
    label[4:7, 2:5, 2:5] = 2
    case = VolumeCase("CARE", "short", image, label, (float("nan"),)*3, tumor_label_id=2, normal_label_id=1, intensity_mode="normalized")
    rows = build_t1(case, tmp_path, {"positives": 3, "negatives": 9, "permutations": 1}, np.random.default_rng(1))
    assert rows == []  # Eight negative slices cannot form a 12-slice question.


def test_t1_omits_only_negative_question_when_pool_is_9_to_11(tmp_path):
    image = np.zeros((14, 8, 8), dtype=np.float32)
    label = np.zeros_like(image, dtype=np.uint8)
    label[4:7, 2:5, 2:5] = 2
    case = VolumeCase("CARE", "partial", image, label, (float("nan"),)*3, tumor_label_id=2, normal_label_id=1, intensity_mode="normalized")
    rows = build_t1(case, tmp_path, {"positives": 3, "negatives": 9, "permutations": 1}, np.random.default_rng(1))
    assert len(rows) == 1
    assert rows[0]["gt"]["negative_only"] is False


def test_resume_rejects_predictions_from_changed_manifest(tmp_path):
    first = [{"item_id": "same", "gt": {"positive_labels": ["A"]}}]
    changed = [{"item_id": "same", "gt": {"positive_labels": ["B"]}}]
    old = [{"item_id": "same", "manifest_sha256": manifest_fingerprint(first)}]
    with pytest.raises(RuntimeError, match="fresh prediction output path"):
        validate_resume_predictions(old, manifest_fingerprint(changed), tmp_path / "old.jsonl")
    validate_resume_predictions(old, manifest_fingerprint(first), tmp_path / "old.jsonl")
