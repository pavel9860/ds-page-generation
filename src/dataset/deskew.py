"""Projection-profile deskew (ported from smartcrop-pdf-web's deskew.ts),
for raster sources without a reliable text layer: standalone scans/images,
and PDF pages found to have no extractable text.

Coarse-to-fine angle search on a downscaled, Otsu-binarized copy: the angle
whose rotated row-sum profile has max variance (text lines aligning into
rows) is the deskew angle. Applied to the full-resolution image afterward.
"""
import cv2
import numpy as np

_DOWNSCALE_PX = 800
_MAX_DEG = 15.0
_COARSE_STEP = 1.0
# Below this we treat the page as already flat -- skip the warpAffine call
# (and the noise it would otherwise add) entirely.
_SKIP_ANGLE_DEG = 0.15


def _row_variance_at_angle(bw: np.ndarray, angle_deg: float) -> float:
    h, w = bw.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    rotated = cv2.warpAffine(bw, m, (w, h), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    row_sums = rotated.sum(axis=1).astype(np.float64)
    return float(row_sums.var())


def estimate_deskew_angle(gray: np.ndarray, downscale_px: int = _DOWNSCALE_PX,
                          max_deg: float = _MAX_DEG) -> float:
    """gray: uint8 2D. -> angle in degrees (rotate by +angle to deskew)."""
    scale = min(1.0, downscale_px / max(gray.shape))
    small = (cv2.resize(gray, (round(gray.shape[1] * scale), round(gray.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA) if scale < 1.0 else gray)
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    best_angle, best_var = 0.0, -1.0
    a = -max_deg
    while a <= max_deg + 1e-9:
        v = _row_variance_at_angle(bw, a)
        if v > best_var:
            best_var, best_angle = v, a
        a += _COARSE_STEP
    refine_step = _COARSE_STEP / 10
    a = best_angle - _COARSE_STEP
    while a <= best_angle + _COARSE_STEP + 1e-9:
        v = _row_variance_at_angle(bw, a)
        if v > best_var:
            best_var, best_angle = v, a
        a += refine_step
    return best_angle


def needs_deskew(angle_deg: float) -> bool:
    return abs(angle_deg) >= _SKIP_ANGLE_DEG


def rotate_full_res(img: np.ndarray, angle_deg: float, fill_value: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=fill_value)
