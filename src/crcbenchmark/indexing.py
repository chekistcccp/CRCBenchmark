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
    raise FileNotFoundError(f"Could not locate CARE train/train_npz or test/test_npz under {root}")


def _care_csv_rows(csv_path, npz_dir, split):
    out = []
    total_existing = 0
    unparsed = []
    missing_npz = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        candidates = [] if header is None else [header]
        candidates.extend(reader)
        for row in candidates:
            if not row:
                continue
            name = row[0].strip()
            if name.lower() in {"filename", "file", "name"}:
                continue
            npz_path = npz_dir / f"{name}.npz"
            if not npz_path.exists():
                missing_npz.append(name)
                continue
            total_existing += 1
            parsed = infer_case_slice(name)
            if parsed is None:
                unparsed.append(name)
                continue
            case_id, slice_index = parsed
            out.append({"case_id": case_id, "slice_index": slice_index, "npz_path": str(npz_path), "split": split})
    return out, total_existing, unparsed, missing_npz


def index_care(root, mapping_csv=None, tumor_label_id=None, normal_label_id=None):
    if tumor_label_id is None:
        raise ValueError(
            "CARE label semantics have not been explicitly configured. Run `python scripts/inspect_care.py ...` first, "
            "verify which label ID is tumor/normal from the official release, then pass --care-tumor-label and "
            "--care-normal-label. No default CARE label semantics are assumed."
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
        for r in df.to_dict("records"):
            p = Path(str(r["npz_path"]))
            p = p if p.is_absolute() else root / p
            if not p.exists():
                raise FileNotFoundError(f"CARE mapping references missing NPZ: {p}")
            r["npz_path"] = str(p)
            r.setdefault("split", "unknown")
            slice_rows.append(r)
    else:
        diagnostics = []
        for split in ("train", "test"):
            csv_path = root / split / f"{split}_bbox.csv"
            npz_dir = root / split / f"{split}_npz"
            if csv_path.exists() and npz_dir.exists():
                parsed, total, unparsed, missing_npz = _care_csv_rows(csv_path, npz_dir, split)
                slice_rows.extend(parsed)
                diagnostics.append((split, total, unparsed, missing_npz))
        if not diagnostics:
            raise FileNotFoundError(f"CARE bbox CSV / NPZ directories not found under {root}")
        bad = [(s, total, unparsed) for s, total, unparsed, _ in diagnostics if unparsed]
        if bad:
            examples = {s: xs[:10] for s, _, xs in bad}
            counts = {s: {"existing_npz": total, "unparsed": len(xs)} for s, total, xs in bad}
            raise ValueError(
                "CARE filenames do not fully prove patient identity + slice order. "
                f"Diagnostics={counts}; examples={examples}. Provide an explicit --care-mapping CSV before patient-level benchmarking."
            )

    groups = defaultdict(list)
    for r in slice_rows:
        groups[(str(r.get("split", "unknown")), str(r["case_id"]))].append(r)

    records = []
    for (split, case_id), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda x: int(x["slice_index"]))
        records.append({
            "dataset": "CARE",
            "case_id": case_id,
            "format": "care_npz_series",
            "split": split,
            "slices": rows,
            "tumor_label_id": int(tumor_label_id),
            "normal_label_id": None if normal_label_id is None else int(normal_label_id),
            "patient_slice_mapping_source": "explicit_csv" if mapping_csv else "filename_convention",
        })
    return records


def build_case_index(msd_root, care_root, care_mapping=None, care_tumor_label=None, care_normal_label=None):
    rows = []
    if msd_root:
        rows.extend(index_msd(msd_root))
    if care_root:
        rows.extend(index_care(care_root, care_mapping, care_tumor_label, care_normal_label))
    return rows
