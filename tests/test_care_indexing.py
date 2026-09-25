from pathlib import Path

import pytest

from crcbenchmark.indexing import index_care


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def _make_release(root: Path):
    for split in ("train", "test"):
        npz_dir = root / split / f"{split}_npz"
        npz_dir.mkdir(parents=True, exist_ok=True)

    train_names = [
        "caseA_slice001",
        "caseA_slice002",
        "caseB_slice010",
    ]
    test_names = [
        "caseC_slice100",
        "caseC_slice101",
    ]

    for name in train_names:
        _touch(root / "train" / "train_npz" / f"{name}.npz")
    for name in test_names:
        _touch(root / "test" / "test_npz" / f"{name}.npz")

    (root / "train" / "train.txt").write_text("\n".join(train_names) + "\n", encoding="utf-8")
    (root / "test" / "test.txt").write_text("\n".join(test_names) + "\n", encoding="utf-8")

    (root / "train" / "train_bbox.txt").write_text("\n".join(train_names[:2]) + "\n", encoding="utf-8")
    (root / "test" / "test_bbox.txt").write_text(test_names[0] + "\n", encoding="utf-8")

    (root / "train" / "train_bbox.csv").write_text(
        "pid,pos\n" + "\n".join(f'{x},"[0,0,1,1]"' for x in train_names[:2]) + "\n",
        encoding="utf-8",
    )
    (root / "test" / "test_bbox.csv").write_text(
        "pid,pos\n" + f'{test_names[0]},"[0,0,1,1]"\n',
        encoding="utf-8",
    )


def test_primary_default_uses_test_txt_only(tmp_path):
    _make_release(tmp_path)
    rows = index_care(
        tmp_path,
        tumor_label_id=2,
        normal_label_id=1,
    )
    assert len(rows) == 1
    assert rows[0]["split"] == "test"
    assert rows[0]["case_id"] == "caseC"
    assert rows[0]["care_index_source"] == "txt"
    assert [x["slice_index"] for x in rows[0]["slices"]] == [100, 101]


def test_explicit_train_and_test_txt(tmp_path):
    _make_release(tmp_path)
    rows = index_care(
        tmp_path,
        tumor_label_id=2,
        normal_label_id=1,
        index_source="txt",
        splits=("train", "test"),
    )
    assert {(r["split"], r["case_id"]) for r in rows} == {
        ("train", "caseA"),
        ("train", "caseB"),
        ("test", "caseC"),
    }


def test_bbox_txt_is_supported_for_sensitivity_analysis(tmp_path):
    _make_release(tmp_path)
    rows = index_care(
        tmp_path,
        tumor_label_id=2,
        normal_label_id=1,
        index_source="bbox_txt",
        splits=("test",),
    )
    assert len(rows) == 1
    assert len(rows[0]["slices"]) == 1
    assert rows[0]["care_index_source"] == "bbox_txt"


def test_auto_refuses_disagreeing_txt_and_bbox_csv(tmp_path):
    _make_release(tmp_path)
    with pytest.raises(ValueError, match="competing index sources"):
        index_care(
            tmp_path,
            tumor_label_id=2,
            normal_label_id=1,
            index_source="auto",
            splits=("test",),
        )
