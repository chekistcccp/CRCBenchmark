#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def resolve_root(root: Path) -> Path:
    root = Path(root)
    for p in [root] + [x for x in root.rglob("*") if x.is_dir()]:
        if (p / "test" / "test_npz").is_dir():
            return p
    raise FileNotFoundError(f"Cannot locate CARE test/test_npz under {root}")


def to_u8(img):
    a = np.asarray(img, dtype=np.float32)
    if a.min() >= -1e-4 and a.max() <= 1.0001:
        return (np.clip(a, 0, 1) * 255).astype(np.uint8)
    lo, hi = np.percentile(a[np.isfinite(a)], [0.5, 99.5])
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
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            canvas.paste(im, (c * w, r * h))
    return canvas


def main():
    p = argparse.ArgumentParser(
        description="Create a semantics-neutral CARE class-1/class-2 visual review sheet."
    )
    p.add_argument("care_root", help="Extracted CARE directory; wrapper directory is allowed.")
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--samples", type=int, default=12)
    p.add_argument("--require-both", action="store_true", default=True)
    p.add_argument("--output-dir", default="artifacts/care_label_review")
    a = p.parse_args()

    root = resolve_root(Path(a.care_root))
    list_file = root / a.split / f"{a.split}.txt"
    npz_dir = root / a.split / f"{a.split}_npz"
    if not list_file.exists():
        raise FileNotFoundError(list_file)

    names = [x.strip() for x in list_file.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = []
    rows = []

    for name in names:
        path = npz_dir / f"{name}.npz"
        with np.load(path, allow_pickle=False) as z:
            image = np.asarray(z["image"])
            raw = np.asarray(z["label"])

        label = np.rint(raw).astype(np.int16)
        label[label > 2] = 2

        has1 = bool(np.any(label == 1))
        has2 = bool(np.any(label == 2))
        if a.require_both and not (has1 and has2):
            continue

        u8 = to_u8(image)
        base = Image.fromarray(u8, mode="L").convert("RGB")
        p1 = overlay(u8, label == 1, (255, 80, 80))
        p2 = overlay(u8, label == 2, (80, 255, 80))

        area1 = int((label == 1).sum())
        area2 = int((label == 2).sum())
        rows.append([
            labeled_panel(base, f"{name} | original"),
            labeled_panel(p1, f"canonical class 1 | pixels={area1} | semantics UNKNOWN"),
            labeled_panel(p2, f"canonical class 2 | pixels={area2} | semantics UNKNOWN"),
        ])
        selected.append({
            "sample_id": name,
            "npz_path": str(path),
            "raw_values": sorted(int(x) for x in np.unique(np.rint(raw))),
            "canonical_values": sorted(int(x) for x in np.unique(label)),
            "class1_pixels": area1,
            "class2_pixels": area2,
        })
        if len(selected) >= a.samples:
            break

    if not selected:
        raise SystemExit(
            "No samples containing both canonical classes 1 and 2 were found in the requested scan."
        )

    sheet = montage(rows, 3)
    sheet_path = out_dir / f"care_{a.split}_class_review.jpg"
    sheet.save(sheet_path, quality=94)

    json_path = out_dir / f"care_{a.split}_class_review.json"
    json_path.write_text(
        json.dumps(
            {
                "warning": "Class 1/2 medical semantics are intentionally not assigned by this script.",
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
