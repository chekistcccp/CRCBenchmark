from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from .preprocess import infer_case_slice


def index_msd(root):
    root = Path(root)
    rows = []
    for image_path in sorted((root / "imagesTr").glob("*.nii.gz")):
        mask_path = root / "labelsTr" / image_path.name
        if mask_path.exists():
            rows.append({
                "dataset": "MSD",
                "case_id": image_path.name.replace(".nii.gz", ""),
                "format": "nifti",
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "split": "public_train",
                "tumor_label_id": 1,
                "normal_label_id": None,
            })
    return rows


def resolve_care_root(root: Path) -> Path:
    root = Path(root)
    candidates = [root] + [p for p in root.rglob("*") if p.is_dir()]
    for p in candidates:
        if (p / "train" / "train_npz").is_dir() or (p / "test" / "test_npz").is_dir():
            return p
    raise FileNotFoundError(
        f"Could not locate CARE train/train_npz or test/test_npz under {root}"
    )


def _read_bbox_csv_names(path: Path):
    names = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row and row[0].strip():
                names.append(row[0].strip())
    return names


def _read_txt_names(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return [line.strip() for line in f if line.strip()]


def _all_npz_names(npz_dir: Path):
    return [p.stem for p in sorted(npz_dir.glob("*.npz"))]


def _rows_from_names(names, npz_dir, split, index_source):
    out = []
    missing_npz = []
    unparsed = []
    seen = set()

    for name in names:
        if name in seen:
            raise ValueError(
                f"Duplicate CARE sample ID in {index_source}/{split}: {name}"
            )
        seen.add(name)

        npz_path = npz_dir / f"{name}.npz"
        if not npz_path.exists():
            missing_npz.append(name)
            continue

        parsed = infer_case_slice(name)
        if parsed is None:
            unparsed.append(name)
            continue

        case_id, slice_index = parsed
        out.append({
            "case_id": case_id,
            "slice_index": int(slice_index),
            "npz_path": str(npz_path),
            "split": split,
            "care_index_source": index_source,
        })

    if missing_npz:
        raise FileNotFoundError(
            f"CARE {index_source}/{split} references {len(missing_npz)} missing NPZ files; "
            f"examples={missing_npz[:10]}"
        )
    if unparsed:
        raise ValueError(
            f"CARE {index_source}/{split} contains {len(unparsed)} sample IDs that do not "
            f"prove case_id + slice_index; examples={unparsed[:10]}"
        )
    return out


def _select_care_source(root: Path, split: str, requested: str):
    npz_dir = root / split / f"{split}_npz"
    txt_path = root / split / f"{split}.txt"
    csv_path = root / split / f"{split}_bbox.csv"

    available = {
        "txt": txt_path.exists(),
        "bbox_csv": csv_path.exists(),
        "all_npz": npz_dir.exists(),
    }

    if requested != "auto":
        if requested not in available:
            raise ValueError(
                f"Unknown CARE index source {requested!r}; choose auto/txt/bbox_csv/all_npz"
            )
        if not available[requested]:
            raise FileNotFoundError(
                f"Requested CARE index source {requested!r} is unavailable for split={split}"
            )
        return requested

    # Scientific safeguard: when both release list and bbox CSV exist but differ,
    # do not silently choose one. Audit v3 must be reviewed and the source frozen.
    if txt_path.exists() and csv_path.exists():
        txt_names = _read_txt_names(txt_path)
        csv_names = _read_bbox_csv_names(csv_path)
        if txt_names != csv_names:
            raise ValueError(
                f"CARE has competing index sources for split={split}: "
                f"{split}.txt={len(txt_names)} records vs {split}_bbox.csv={len(csv_names)}. "
                "Run CARE audit v3, review paper-alignment results, then pass "
                "--care-index-source txt or --care-index-source bbox_csv explicitly."
            )
        return "txt"

    if txt_path.exists():
        return "txt"
    if csv_path.exists():
        return "bbox_csv"
    if npz_dir.exists():
        return "all_npz"
    raise FileNotFoundError(f"No CARE index source found for split={split} under {root}")


def _care_split_rows(root: Path, split: str, index_source: str):
    npz_dir = root / split / f"{split}_npz"
    selected = _select_care_source(root, split, index_source)

    if selected == "txt":
        names = _read_txt_names(root / split / f"{split}.txt")
    elif selected == "bbox_csv":
        names = _read_bbox_csv_names(root / split / f"{split}_bbox.csv")
    elif selected == "all_npz":
        names = _all_npz_names(npz_dir)
    else:
        raise AssertionError(selected)

    return _rows_from_names(names, npz_dir, split, selected), selected


def index_care(
    root,
    mapping_csv=None,
    tumor_label_id=None,
    normal_label_id=None,
    index_source="auto",
):
    if tumor_label_id is None:
        raise ValueError(
            "CARE label semantics have not been explicitly configured. Run "
            "scripts/inspect_care.py first, verify which canonical label ID is "
            "tumor/normal, then pass --care-tumor-label and --care-normal-label. "
            "No default CARE label semantics are assumed."
        )

    root = resolve_care_root(Path(root))
    slice_rows = []

    if mapping_csv:
        import pandas as pd

        df = pd.read_csv(mapping_csv)
        required = {"case_id", "slice_index", "npz_path"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CARE mapping missing columns: {sorted(missing)}")

        for row in df.to_dict("records"):
            p = Path(str(row["npz_path"]))
            p = p if p.is_absolute() else root / p
            if not p.exists():
                raise FileNotFoundError(f"CARE mapping references missing NPZ: {p}")
            row["npz_path"] = str(p)
            row.setdefault("split", "unknown")
            row["care_index_source"] = "explicit_mapping_csv"
            slice_rows.append(row)
    else:
        for split in ("train", "test"):
            rows, _ = _care_split_rows(root, split, index_source)
            slice_rows.extend(rows)

    groups = defaultdict(list)
    for row in slice_rows:
        groups[(str(row.get("split", "unknown")), str(row["case_id"]))].append(row)

    case_to_splits = defaultdict(set)
    for split, case_id in groups:
        case_to_splits[case_id].add(split)
    overlap = {k: sorted(v) for k, v in case_to_splits.items() if len(v) > 1}
    if overlap:
        examples = list(overlap.items())[:10]
        raise ValueError(f"CARE patient IDs overlap across splits: {examples}")

    records = []
    for (split, case_id), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda x: int(x["slice_index"]))
        slice_indices = [int(x["slice_index"]) for x in rows]
        if len(slice_indices) != len(set(slice_indices)):
            raise ValueError(f"Duplicate CARE source slice index in {split}/{case_id}")

        records.append({
            "dataset": "CARE",
            "case_id": case_id,
            "format": "care_npz_series",
            "split": split,
            "slices": rows,
            "tumor_label_id": int(tumor_label_id),
            "normal_label_id": (
                None if normal_label_id is None else int(normal_label_id)
            ),
            "patient_slice_mapping_source": (
                "explicit_mapping_csv" if mapping_csv else "filename_convention"
            ),
            "care_index_source": (
                "explicit_mapping_csv"
                if mapping_csv
                else rows[0].get("care_index_source")
            ),
        })

    return records


def build_case_index(
    msd_root,
    care_root,
    care_mapping=None,
    care_tumor_label=None,
    care_normal_label=None,
    care_index_source="auto",
):
    rows = []
    if msd_root:
        rows.extend(index_msd(msd_root))
    if care_root:
        rows.extend(
            index_care(
                care_root,
                mapping_csv=care_mapping,
                tumor_label_id=care_tumor_label,
                normal_label_id=care_normal_label,
                index_source=care_index_source,
            )
        )
    return rows
