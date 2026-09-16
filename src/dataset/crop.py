import cv2
import numpy as np

from src.dataset.content_filter import _content_mask, content_bbox_fast

CROP_SIZE = 1024
PATCH_GRID = 10
PATCH_INK_EPS = 0.01
PATCH_CONTENT_MIN_FRAC = 0.80
_Y_STRIDE = 32


def _patch_coverage_at(integral: np.ndarray, y0: int, size: int, grid: int) -> float:
    step = size // grid
    area = step * step
    hits = 0
    for i in range(grid):
        row = integral[y0 + i * step + step] - integral[y0 + i * step]
        for j in range(grid):
            x = j * step
            if row[x + step] - row[x] > area * PATCH_INK_EPS:
                hits += 1
    return hits / (grid * grid)


def _bbox_strip(gray: np.ndarray, bbox, size: int) -> np.ndarray:
    by0, by1, bx0, bx1 = bbox
    content = gray[by0:by1 + 1, bx0:bx1 + 1]
    ch, cw = content.shape
    scale = size / cw
    sh = max(1, round(ch * scale))
    return cv2.resize(content, (size, sh),
                      interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)


def crop_from_bbox(gray: np.ndarray, bbox_y0: int, bbox_x0: int, bbox_h: int, bbox_w: int,
                   crop_y0: int, size: int = CROP_SIZE) -> np.ndarray:
    bbox = (bbox_y0, bbox_y0 + bbox_h - 1, bbox_x0, bbox_x0 + bbox_w - 1)
    strip = _bbox_strip(gray, bbox, size)
    return strip[crop_y0:crop_y0 + size, 0:size]


def select_window_in_mask(mask: np.ndarray, rng, size: int = CROP_SIZE,
                          min_patch_frac: float = PATCH_CONTENT_MIN_FRAC):
    """Vertical crop offset into a size-wide content-occupancy mask (any
    source: pixel BlackHat/TopHat, PDF text-block occupancy, ...) whose
    8x8/10x10 patch grid clears min_patch_frac coverage. None if the mask
    is shorter than size, or no offset qualifies."""
    sh = mask.shape[0]
    if sh < size:
        return None
    integral = cv2.integral(mask.astype(np.uint8))
    max_y0 = sh - size
    candidates = list(range(0, max_y0 + 1, _Y_STRIDE))
    if candidates[-1] != max_y0:
        candidates.append(max_y0)
    qualifying = [y0 for y0 in candidates
                 if _patch_coverage_at(integral, y0, size, PATCH_GRID) >= min_patch_frac]
    if not qualifying:
        return None
    return int(rng.choice(qualifying))


def select_crop_1024(gray: np.ndarray, rng, bbox=None, size: int = CROP_SIZE,
                     min_patch_frac: float = PATCH_CONTENT_MIN_FRAC):
    if bbox is None:
        bbox, _ = content_bbox_fast(gray)
    if bbox is None:
        return None
    by0, by1, bx0, bx1 = bbox
    bbox_h, bbox_w = by1 - by0 + 1, bx1 - bx0 + 1
    strip = _bbox_strip(gray, bbox, size)
    crop_y0 = select_window_in_mask(_content_mask(strip), rng, size, min_patch_frac)
    if crop_y0 is None:
        return None
    return dict(bbox_y0=by0, bbox_x0=bx0, bbox_h=bbox_h, bbox_w=bbox_w, crop_y0=crop_y0)
