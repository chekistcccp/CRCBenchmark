from __future__ import annotations
from .preprocess import load_nifti_case, load_care_npz_series, VolumeCase


def load_case(record: dict) -> VolumeCase:
    fmt = record["format"]
    if fmt == "nifti":
        return load_nifti_case(
            record["image_path"], record["mask_path"], record["dataset"], record["case_id"], record.get("split")
        )
    if fmt == "care_npz_series":
        return load_care_npz_series(
            record["slices"],
            record["case_id"],
            record.get("split"),
            tumor_label_id=record.get("tumor_label_id"),
            normal_label_id=record.get("normal_label_id"),
        )
    raise ValueError(f"unknown format: {fmt}")
