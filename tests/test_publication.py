import copy
import json
from pathlib import Path
import runpy

import numpy as np
from PIL import Image
import pytest

from crcbenchmark.inference import manifest_fingerprint
from crcbenchmark.publication import (evidence_items, evidence_report, relocate_items,
                                     resolve_artifact, select_human_items, validity_agreement)


def t3_item(tmp_path, case="one", truth="EFGHI"):
    path = tmp_path / f"{case}.png"
    Image.new("RGB", (768, 768), (240, 20, 20)).save(path)
    return {"item_id": f"t3:{case}", "case_id": case, "dataset": "MSD", "track": "t3",
            "benchmark_split": "dev", "image_path": str(path), "prompt": "Same prompt",
            "gt": {"slice_labels": dict(zip("ABCDEFGHI", range(9))), "positive_labels": list(truth),
                   "boundary_slice": 4, "side": "entry", "spacing_z_mm": None}}


def test_controls_preserve_question_and_hash_actual_pixels(tmp_path):
    source = t3_item(tmp_path)
    saved = copy.deepcopy(source)
    rows = evidence_items(relocate_items([source]), tmp_path / "neutral")
    assert source == saved
    assert len({r["item_id"] for r in rows}) == 3
    assert all(r["gt"] == saved["gt"] and r["prompt"] == saved["prompt"] for r in rows)
    assert rows[1]["image_path"] is None and rows[1]["input_image_sha256"] is None
    neutral = np.asarray(Image.open(rows[2]["image_path"]))
    assert np.all(neutral[100:200, 100:200] == 128)
    assert np.any(neutral[:25, :29] != 128)  # Only fixed label scaffolding remains.
    old = manifest_fingerprint(rows)
    Image.new("RGB", (768, 768), "blue").save(source["image_path"])
    assert manifest_fingerprint(evidence_items(relocate_items([source]), tmp_path / "neutral")) != old


def test_constant_answers_have_zero_paired_effect(tmp_path):
    sources = [t3_item(tmp_path, "one"), t3_item(tmp_path, "two", "DEFGHI")]
    rows = evidence_items(relocate_items(sources), tmp_path / "neutral")
    predictions = [{"item_id": r["item_id"], "raw_response": json.dumps(list("ABCDEFGHI"))} for r in rows]
    report = evidence_report(rows, predictions)
    for contrast in report["paired_differences"]:
        for result in contrast["comparisons"]:
            assert result["mean"] == 0 and result["n"] == 2
    with pytest.raises(ValueError, match="completely"):
        evidence_report(rows, predictions[:-1])


def test_eval_rejected_and_remap_cannot_escape_root(tmp_path):
    source = t3_item(tmp_path)
    source["benchmark_split"] = "eval"
    with pytest.raises(ValueError, match="development"):
        evidence_items([source], tmp_path / "neutral")
    with pytest.raises(ValueError, match="escapes"):
        resolve_artifact("/server/artifacts/../../secret.png", tmp_path)


def test_human_sample_is_reproducible_patient_limited_and_keeps_swaps(tmp_path):
    rows = []
    for case in ("one", "two", "three"):
        source = t3_item(tmp_path, case)
        for swap in (0, 1):
            rows.append({**source, "item_id": f"t4:{case}:{swap}", "track": "t4", "gt": {"swap": swap}})
        rows.append(source)
    selected = select_human_items(rows, patients_per_group=2)
    assert len([r for r in selected if r["track"] == "t4"]) == 4
    assert len([r for r in selected if r["track"] == "t3"]) == 2
    assert selected == select_human_items(list(reversed(rows)), patients_per_group=2)


def test_validity_agreement_reports_undefined_kappa():
    assert validity_agreement(["yes", "no"], ["yes", "no"])["cohen_kappa"] == 1
    assert validity_agreement(["yes"], ["yes"])["cohen_kappa"] is None


def test_reader_scoring_rejects_wrong_packet_and_missing_answers():
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "evaluate_human_review.py"))
    validate = namespace["validate_reader"]
    key = {"packet_sha256": "expected", "items": [{"review_id": "first"}, {"review_id": "second"}]}
    rows = [{"review_id": name, "packet_sha256": "expected", "reviewer_id": "doctor_1",
             "task_valid": "yes", "raw_response": "A"} for name in ("first", "second")]
    assert validate(rows, key)[0] == "doctor_1"
    with pytest.raises(ValueError, match="every selected"):
        validate(rows[:1], key)
    rows[0]["packet_sha256"] = "different"
    with pytest.raises(ValueError, match="fingerprint"):
        validate(rows, key)
