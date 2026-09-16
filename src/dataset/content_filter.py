import cv2
import numpy as np

_LINE_KSIZE = 15
_LINE_THRESH = 36
_SCAN_MAX_SIDE = 512
MIN_CONTENT_AREA_FRAC = 0.01
MAX_MARGIN_FRAC = 0.50
MIN_CONTENT_FRAC = 0.01


def _content_mask(gray: np.ndarray, ksize: int = _LINE_KSIZE, thresh: int = _LINE_THRESH) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    whitehat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    return cv2.max(blackhat, whitehat) > thresh


def _bbox_from_mask(mask: np.ndarray):
    rows = mask.any(axis=1)
    if not rows.any():
        return None
    cols = mask.any(axis=0)
    y0, y1 = int(np.argmax(rows)), len(rows) - 1 - int(np.argmax(rows[::-1]))
    x0, x1 = int(np.argmax(cols)), len(cols) - 1 - int(np.argmax(cols[::-1]))
    return y0, y1, x0, x1


def content_bbox_fast(gray: np.ndarray):
    h, w = gray.shape
    scale = min(1.0, _SCAN_MAX_SIDE / max(h, w))
    small = (cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            if scale < 1.0 else gray)
    mask = _content_mask(small)
    bbox = _bbox_from_mask(mask)
    if bbox is None:
        return None, float(mask.mean())
    content_frac = float(mask.mean())
    if scale == 1.0:
        return bbox, content_frac
    y0, y1, x0, x1 = bbox
    inv = 1.0 / scale
    return (int(y0 * inv), min(h - 1, int(y1 * inv) + 1),
            int(x0 * inv), min(w - 1, int(x1 * inv) + 1)), content_frac


def bbox_keep(bbox, h: int, w: int, min_area_frac: float = MIN_CONTENT_AREA_FRAC,
             max_margin_frac: float = MAX_MARGIN_FRAC) -> bool:
    """Margin/area keep-check for a content bbox in an h x w page --
    shared by any mask source (pixel BlackHat/TopHat, PDF text-block
    occupancy, ...), not just the raster path below."""
    y0, y1, x0, x1 = bbox
    bbox_area_frac = ((y1 - y0 + 1) * (x1 - x0 + 1)) / (h * w)
    v_margin_frac = 1.0 - (y1 - y0 + 1) / h
    h_margin_frac = 1.0 - (x1 - x0 + 1) / w
    return (bbox_area_frac >= min_area_frac and v_margin_frac <= max_margin_frac
           and h_margin_frac <= max_margin_frac)


def has_enough_content(gray: np.ndarray, min_area_frac: float = MIN_CONTENT_AREA_FRAC,
                       max_margin_frac: float = MAX_MARGIN_FRAC,
                       min_content_frac: float = MIN_CONTENT_FRAC):
    bbox, content_frac = content_bbox_fast(gray)
    if bbox is None:
        return None, content_frac, False
    keep = (bbox_keep(bbox, *gray.shape, min_area_frac, max_margin_frac)
           and content_frac >= min_content_frac)
    return bbox, content_frac, keep
