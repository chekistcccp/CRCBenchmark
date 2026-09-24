from __future__ import annotations
import numpy as np


def recall_at_k(ranked: list[str], positives: set[str], k: int) -> float:
    if not positives:
        return float("nan")
    return float(any(x in positives for x in ranked[:k]))


def reciprocal_rank(ranked: list[str], positives: set[str]) -> float:
    for i, x in enumerate(ranked, 1):
        if x in positives:
            return 1.0 / i
    return 0.0


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = aa + ba - inter
    return inter / den if den > 0 else 0.0


def pointing_hit(point_xy: list[float], mask: np.ndarray) -> float:
    x, y = point_xy
    x = int(round(x)); y = int(round(y))
    if x < 0 or y < 0 or y >= mask.shape[0] or x >= mask.shape[1]:
        return 0.0
    return float(mask[y, x] > 0)


def binary_prf(pred: set[str], truth: set[str]) -> tuple[float, float, float]:
    tp = len(pred & truth)
    fp = len(pred - truth)
    fn = len(truth - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def normalized_distance(point_xy: list[float], centroid_xy: list[float], width: int, height: int) -> float:
    p = np.asarray(point_xy, float); c = np.asarray(centroid_xy, float)
    return float(np.linalg.norm(p - c) / np.sqrt(width * width + height * height))
