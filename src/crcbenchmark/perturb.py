from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt


def feather_mask(mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    alpha = gaussian_filter(mask.astype(np.float32), sigma=sigma)
    if alpha.max() > 0:
        alpha /= alpha.max()
    return np.clip(alpha, 0.0, 1.0)


def gaussian_suppress(image: np.ndarray, mask: np.ndarray, sigma: float = 12.0, feather_sigma: float = 2.0) -> np.ndarray:
    blurred = gaussian_filter(image.astype(np.float32), sigma=sigma)
    alpha = feather_mask(mask, feather_sigma)
    return image.astype(np.float32) * (1 - alpha) + blurred * alpha


def median_replace(image: np.ndarray, mask: np.ndarray, ring_mask: np.ndarray | None = None, feather_sigma: float = 2.0) -> np.ndarray:
    if ring_mask is None or not np.any(ring_mask):
        ring_mask = ~mask
    vals = image[ring_mask]
    if vals.size == 0:
        return image.copy()
    fill = float(np.median(vals))
    repl = np.full_like(image, fill, dtype=np.float32)
    alpha = feather_mask(mask, feather_sigma)
    return image.astype(np.float32) * (1 - alpha) + repl * alpha


def nested_fraction_mask(mask: np.ndarray, fraction: float) -> np.ndarray:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0,1]")
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return mask.copy()
    dist = distance_transform_edt(mask)
    values = dist[mask]
    n = max(1, int(round(values.size * fraction)))
    thresh = np.partition(values, values.size - n)[values.size - n]
    return mask & (dist >= thresh)


def shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    ys, xs = np.where(mask)
    xs2, ys2 = xs + dx, ys + dy
    keep = (xs2 >= 0) & (xs2 < w) & (ys2 >= 0) & (ys2 < h)
    out[ys2[keep], xs2[keep]] = True
    return out


def choose_matched_control(image, lesion_mask, allowed_mask, forbidden_mask, trials, rng):
    if not np.any(lesion_mask) or not np.any(allowed_mask):
        return None
    ys, xs = np.where(lesion_mask)
    cy, cx = ys.mean(), xs.mean()
    lesion_mean = float(np.mean(image[lesion_mask]))
    ay, ax = np.where(allowed_mask & ~forbidden_mask)
    if len(ax) == 0:
        return None
    best, best_score = None, float("inf")
    for _ in range(trials):
        j = int(rng.integers(0, len(ax)))
        dx, dy = int(round(ax[j] - cx)), int(round(ay[j] - cy))
        candidate = shift_mask(lesion_mask, dx, dy)
        if candidate.sum() < max(1, int(0.9 * lesion_mask.sum())):
            continue
        if np.any(candidate & forbidden_mask) or not np.all(allowed_mask[candidate]):
            continue
        score = abs(float(np.mean(image[candidate])) - lesion_mean)
        if score < best_score:
            best_score, best = score, candidate
    return best
