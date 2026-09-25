#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageDraw


def to_u8(img):
    a = np.asarray(img, dtype=np.float32)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros_like(a, dtype=np.uint8)
    if finite.min() >= -1e-4 and finite.max() <= 1.0001:
        return (np.clip(a, 0, 1) * 255).astype(np.uint8)
    lo, hi = np.percentile(finite, [0.5, 99.5])
    if hi <= lo:
        return np.zeros_like(a, dtype=np.uint8)
    return (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def overlay(base_u8, mask, rgb):
    base = np.stack([base_u8] * 3, axis=-1).astype(np.float32)
    m = mask.astype(bool)
    color = np.asarray(rgb, dtype=np.float32)
    base[m] = 0.45 * base[m] + 0.55 * color
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def labeled_panel(img, title):
    img = img.convert("RGB")
    canvas = Image.new("RGB", (img.width, img.height + 34), "black")
    canvas.paste(img, (0, 34))
    d = ImageDraw.Draw(canvas)
    d.text((8, 8), title, fill="white")
    return canvas


def montage(rows, cols):
    if not rows:
        raise ValueError("No review samples were found.")
    w = max(x.width for row in rows for x in row)
    h = max(x.height for row in rows for x in row)
    canvas = Image.new("RGB", (cols * w, len(rows) * h), "black")
    for rr, row in enumerate(rows):
        for cc, im in enumerate(row):
            canvas.paste(im, (cc * w, rr * h))
    return canvas


def _detect_zip_root(names, split):
    suffix = f"/{split}/{split}.txt"
    for name in names:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    raise FileNotFoundError(f"Cannot locate {split}/{split}.txt in CARE ZIP")


def iter_from_zip(zip_path: Path, split: str):
    z = zipfile.ZipFile(zip_path)
    names = [x.filename for x in z.infolist() if not x.is_dir()]
    root = _detect_zip_root(names, split)
    prefix = f"{root}/" if root else ""
    list_name = f"{prefix}{split}/{split}.txt"
    list_text = z.read(list_name).decode("utf-8-sig", errors="replace")
    sample_ids = [x.strip() for x in list_text.splitlines() if x.strip()]

    try:
        for sample_id in sample_ids:
            npz_name = f"{prefix}{split}/{split}_npz/{sample_id}.npz"
            raw_bytes = z.read(npz_name)
            with np.load(io.BytesIO(raw_bytes), allow_pickle=False) as obj:
                yield sample_id, np.asarray(obj["image"]), np.asarray(obj["label"]), npz_name
    finally:
        z.close()


def _resolve_extracted_root(root: Path, split: str) -> Path:
    for p in [root] + [x for x in root.rglob("*") if x.is_dir()]:
        if (p / split / f"{split}_npz").is_dir() and (p / split / f"{split}.txt").is_file():
            return p
    raise FileNotFoundError(f"Cannot locate CARE {split}/{split}_npz under {root}")


def iter_from_dir(root: Path, split: str):
    data = _resolve_extracted_root(root, split)
    list_file = data / split / f"{split}.txt"
    npz_dir = data / split / f"{split}_npz"
    sample_ids = [x.strip() for x in list_file.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    for sample_id in sample_ids:
        path = npz_dir / f"{sample_id}.npz"
        with np.load(path, allow_pickle=False) as obj:
            yield sample_id, np.asarray(obj["image"]), np.asarray(obj["label"]), str(path)


def main():
    p = argparse.ArgumentParser(
        description="Create a semantics-neutral CARE class-1/class-2 visual review sheet."
    )
    p.add_argument("care_source", help="CARE.zip or extracted CARE directory")
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--samples", type=int, default=12)
    p.add_argument("--output-dir", default="artifacts/care_label_review")
    a = p.parse_args()

    source = Path(a.care_source)
    if source.is_dir():
        iterator = iter_from_dir(source, a.split)
        source_type = "extracted_directory"
    elif source.suffix.lower() == ".zip":
        iterator = iter_from_zip(source, a.split)
        source_type = "zip"
    else:
        raise SystemExit("care_source must be CARE.zip or an extracted CARE directory")

    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = []
    rows = []

    for sample_id, image, raw, npz_path in iterator:
        label = np.rint(raw).astype(np.int16)
        label[label > 2] = 2

        has1 = bool(np.any(label == 1))
        has2 = bool(np.any(label == 2))
        if not (has1 and has2):
            continue

        u8 = to_u8(image)
        base = Image.fromarray(u8, mode="L").convert("RGB")
        p1 = overlay(u8, label == 1, (255, 80, 80))
        p2 = overlay(u8, label == 2, (80, 255, 80))

        area1 = int((label == 1).sum())
        area2 = int((label == 2).sum())

        rows.append([
            labeled_panel(base, f"{sample_id} | original"),
            labeled_panel(p1, f"canonical class 1 | pixels={area1} | semantics UNKNOWN"),
            labeled_panel(p2, f"canonical class 2 | pixels={area2} | semantics UNKNOWN"),
        ])
        selected.append({
            "sample_id": sample_id,
            "npz_path": npz_path,
            "raw_values": sorted(int(x) for x in np.unique(np.rint(raw))),
            "canonical_values": sorted(int(x) for x in np.unique(label)),
            "class1_pixels": area1,
            "class2_pixels": area2,
        })

        if len(selected) >= a.samples:
            break

    if not selected:
        raise SystemExit(
            "No samples containing both canonical classes 1 and 2 were found."
        )

    sheet = montage(rows, 3)
    sheet_path = out_dir / f"care_{a.split}_class_review.jpg"
    sheet.save(sheet_path, quality=94)

    json_path = out_dir / f"care_{a.split}_class_review.json"
    json_path.write_text(
        json.dumps(
            {
                "warning": "Class 1/2 medical semantics are intentionally not assigned by this script.",
                "source": str(source),
                "source_type": source_type,
                "split": a.split,
                "sample_count": len(selected),
                "samples": selected,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("Saved:", sheet_path)
    print("Saved:", json_path)
    print("Upload the JPG for visual semantic review before setting CARE_TUMOR_LABEL / CARE_NORMAL_LABEL.")


if __name__ == "__main__":
    main()
