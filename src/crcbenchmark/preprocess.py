from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation


@dataclass
class VolumeCase:
    dataset: str
    case_id: str
    image: np.ndarray
    label: np.ndarray
    spacing_zyx: tuple[float, float, float]
    split: str | None = None
    intensity_mode: str = "hu"
    tumor_label_id: int | None = None
    normal_label_id: int | None = None

    def _require_tumor_label(self) -> int:
        if self.tumor_label_id is None:
            raise ValueError(
                f"Tumor label semantics are not verified for {self.dataset}/{self.case_id}. "
                "Run scripts/inspect_care.py and pass explicit CARE label IDs during indexing."
            )
        return int(self.tumor_label_id)

    def tumor_mask(self):
        return self.label == self._require_tumor_label()

    def normal_mask(self):
        if self.normal_label_id is None:
            return np.zeros_like(self.label, dtype=bool)
        return self.label == int(self.normal_label_id)


def canonical_nifti(path):
    return nib.as_closest_canonical(nib.load(str(path)))


def load_nifti_case(image_path, mask_path, dataset, case_id, split=None):
    ii, mi = canonical_nifti(image_path), canonical_nifti(mask_path)
    image = np.asarray(ii.dataobj, dtype=np.float32)
    label = np.asarray(mi.dataobj)
    if image.shape != label.shape:
        raise ValueError(f"shape mismatch for {case_id}: {image.shape} vs {label.shape}")
    image = np.moveaxis(image, -1, 0)
    label = np.moveaxis(label, -1, 0)
    z = ii.header.get_zooms()[:3]
    return VolumeCase(
        dataset, case_id, image, label,
        (float(z[2]), float(z[1]), float(z[0])),
        split, "hu", tumor_label_id=1, normal_label_id=None,
    )


def infer_case_slice(filename):
    """Conservative parser: never interprets a bare integer as patient+slice."""
    stem = Path(filename).stem
    for pat in (
        r"^(?P<case>.+?)[_-](?:slice|z)?(?P<slice>\d+)$",
        r"^(?P<case>.+?)[_-](?P<slice>\d+)$",
    ):
        m = re.match(pat, stem, flags=re.IGNORECASE)
        if m and m.group("case"):
            return m.group("case"), int(m.group("slice"))
    return None


def _optional_spacing(row, key):
    v = row.get(key, None)
    if v is None or v == "":
        return float("nan")
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if np.isfinite(x) and x > 0 else float("nan")


def load_care_npz_series(rows, case_id, split=None, tumor_label_id=None, normal_label_id=None):
    if tumor_label_id is None:
        raise ValueError(
            "CARE tumor label ID is not configured. Run the CARE audit first; then index with "
            "--care-tumor-label and --care-normal-label only after verifying the official semantics."
        )
    rows = sorted(rows, key=lambda x: int(x["slice_index"]))
    images, labels, indices = [], [], []
    for r in rows:
        with np.load(r["npz_path"], allow_pickle=False) as o:
            if "image" not in o or "label" not in o:
                raise KeyError(f"CARE NPZ missing image/label keys: {r['npz_path']}")
            img = np.asarray(o["image"], dtype=np.float32)
            raw = np.asarray(o["label"])
            if not np.all(np.isfinite(raw)):
                raise ValueError(f"CARE label has non-finite values: {r['npz_path']}")
            rounded = np.rint(raw)
            if not np.allclose(raw, rounded):
                raise ValueError(f"CARE label is not integer-valued: {r['npz_path']}")
            # Match the official U-SAM CARE loader:
            # raw label values > 2 are collapsed to canonical class 2.
            lab = rounded.astype(np.int16)
            lab[lab > 2] = 2
            images.append(img)
            labels.append(lab)
        indices.append(int(r["slice_index"]))

    image = np.stack(images)
    label = np.stack(labels)
    mode = "normalized" if image.min() >= -1e-4 and image.max() <= 1.0001 else "preprocessed"

    # The packaged NPZ release does not itself prove physical spacing.
    # Missing spacing stays NaN instead of being silently replaced by 1 mm.
    dz = _optional_spacing(rows[0], "slice_spacing")
    dy = _optional_spacing(rows[0], "pixel_spacing_y")
    dx = _optional_spacing(rows[0], "pixel_spacing_x")

    c = VolumeCase(
        "CARE", case_id, image, label, (dz, dy, dx), split, mode,
        tumor_label_id=int(tumor_label_id),
        normal_label_id=None if normal_label_id is None else int(normal_label_id),
    )
    c.slice_indices = indices
    c.source_slice_indices = indices
    c.care_label_rule = "raw_gt2_to_2"
    return c


def window_to_uint8(image, level=50, width=400, intensity_mode="hu"):
    a = image.astype(np.float32)
    if intensity_mode in {"normalized", "preprocessed"} or (a.min() >= -1e-4 and a.max() <= 1.0001):
        if a.min() >= -1e-4 and a.max() <= 1.0001:
            return (np.clip(a, 0, 1) * 255).astype(np.uint8)
        lo, hi = np.nanpercentile(a, [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        if hi <= lo:
            return np.zeros_like(a, dtype=np.uint8)
        return (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    lo, hi = level - width / 2, level + width / 2
    a = np.clip(a, lo, hi)
    return ((a - lo) / (hi - lo) * 255).astype(np.uint8)


def to_rgb_pil(slice2d, level=50, width=400, intensity_mode="hu", size=None):
    im = Image.fromarray(window_to_uint8(slice2d, level, width, intensity_mode), mode="L").convert("RGB")
    if size is not None:
        if isinstance(size, int):
            size = (size, size)
        im = im.resize(size, Image.Resampling.BILINEAR)
    return im


def make_montage(images, labels, rows, cols, tile_size=256, margin=8):
    canvas = Image.new("RGB", (cols * tile_size, rows * tile_size), "black")
    draw = ImageDraw.Draw(canvas)
    for i, (im, lab) in enumerate(zip(images, labels)):
        r, c = divmod(i, cols)
        canvas.paste(im.resize((tile_size, tile_size), Image.Resampling.BILINEAR), (c * tile_size, r * tile_size))
        draw.rectangle((c * tile_size, r * tile_size, c * tile_size + 28, r * tile_size + 24), fill="black")
        draw.text((c * tile_size + 6, r * tile_size + 3), lab, fill="white")
    return canvas


def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    return None if len(xs) == 0 else [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def centroid_from_mask(mask):
    ys, xs = np.where(mask > 0)
    return None if len(xs) == 0 else [float(xs.mean()), float(ys.mean())]


def normalize_point(p, width, height):
    return [int(round(p[0] / max(width - 1, 1) * 1000)), int(round(p[1] / max(height - 1, 1) * 1000))]


def normalize_box(b, width, height):
    x1, y1, x2, y2 = b
    return [int(round(x1 / width * 1000)), int(round(y1 / height * 1000)), int(round(x2 / width * 1000)), int(round(y2 / height * 1000))]


def crop_center_physical(image, center_xy, spacing_yx, fov_mm):
    h, w = image.shape
    sx = max(int(round((fov_mm / spacing_yx[1]) / 2)), 1)
    sy = max(int(round((fov_mm / spacing_yx[0]) / 2)), 1)
    cx, cy = center_xy
    x1, x2 = int(round(cx)) - sx, int(round(cx)) + sx
    y1, y2 = int(round(cy)) - sy, int(round(cy)) + sy
    pl, pr = max(0, -x1), max(0, x2 - w)
    pt, pb = max(0, -y1), max(0, y2 - h)
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    crop = image[y1:y2, x1:x2]
    return np.pad(crop, ((pt, pb), (pl, pr)), mode="edge") if any((pl, pr, pt, pb)) else crop


def crop_center_pixels(image, center_xy, size_px):
    h, w = image.shape
    half = max(int(round(float(size_px) / 2)), 1)
    cx, cy = center_xy
    x1, x2 = int(round(cx)) - half, int(round(cx)) + half
    y1, y2 = int(round(cy)) - half, int(round(cy)) + half
    pl, pr = max(0, -x1), max(0, x2 - w)
    pt, pb = max(0, -y1), max(0, y2 - h)
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    crop = image[y1:y2, x1:x2]
    return np.pad(crop, ((pt, pb), (pl, pr)), mode="edge") if any((pl, pr, pt, pb)) else crop


def body_mask_from_hu(image):
    return binary_dilation(image > -500, iterations=2)
