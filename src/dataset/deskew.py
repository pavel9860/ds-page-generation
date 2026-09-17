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
_N_STRIPS = 5
# Below this we treat the page as already flat -- skip the warpAffine call
# (and the noise it would otherwise add) entirely. Well above the coarse
# search's own step so a single stray strip can't pass it on noise alone.
_SKIP_ANGLE_DEG = 0.25
# Above this the estimate is treated as a false detection (unrelated
# structure dominating the variance metric, not real page skew) and
# dropped rather than applied.
_MAX_APPLIED_DEG = 10.0


def _row_variance_at_angle(bw: np.ndarray, angle_deg: float) -> float:
    h, w = bw.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    rotated = cv2.warpAffine(bw, m, (w, h), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    row_sums = rotated.sum(axis=1).astype(np.float64)
    return float(row_sums.var())


def _best_angle(bw: np.ndarray, max_deg: float) -> float:
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


def estimate_deskew_angle(gray: np.ndarray, downscale_px: int = _DOWNSCALE_PX,
                          max_deg: float = _MAX_DEG, n_strips: int = _N_STRIPS) -> float:
    """gray: uint8 2D. -> angle in degrees (rotate by +angle to deskew),
    the median of independent per-strip estimates -- a single region's
    spurious peak (a table border, a stray mark) skews the whole-page
    variance metric, but can't move the median of several strips."""
    scale = min(1.0, downscale_px / max(gray.shape))
    small = (cv2.resize(gray, (round(gray.shape[1] * scale), round(gray.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA) if scale < 1.0 else gray)
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h = bw.shape[0]
    strip_h = max(1, h // n_strips)
    angles = [_best_angle(bw[i * strip_h:min(h, (i + 1) * strip_h)], max_deg)
             for i in range(n_strips)]
    return float(np.median(angles))


def needs_deskew(angle_deg: float) -> bool:
    return _SKIP_ANGLE_DEG <= abs(angle_deg) <= _MAX_APPLIED_DEG


def detect_rotation(gray: np.ndarray) -> int:
    """Content-based 90-degree orientation: text lines produce sharp
    horizontal-band row variance only when the page is actually upright,
    so picking whichever of (as-is, +90, -90) maximizes it identifies the
    true orientation directly. Aspect ratio alone can't tell a genuinely
    landscape document from a portrait page rotated sideways -- that
    conflation was rotating correctly-oriented landscape pages into
    sideways ones. Run once at manifest-build time, before bbox/crop
    selection, since a 90-degree rotation moves content far enough that
    any bbox/crop computed before it would no longer apply; the decision
    is stored in the manifest and replayed as-is at page-building time.
    Returns one of 0, 90, -90 (degrees to rotate clockwise)."""
    scale = min(1.0, _DOWNSCALE_PX / max(gray.shape))
    small = (cv2.resize(gray, (round(gray.shape[1] * scale), round(gray.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA) if scale < 1.0 else gray)
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    candidates = [(_row_variance_at_angle(bw, 0.0), 0),
                 (_row_variance_at_angle(cv2.rotate(bw, cv2.ROTATE_90_CLOCKWISE), 0.0), 90),
                 (_row_variance_at_angle(cv2.rotate(bw, cv2.ROTATE_90_COUNTERCLOCKWISE), 0.0), -90)]
    return max(candidates)[1]


def rotate90(gray: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 90:
        return cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    if degrees == -90:
        return cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return gray


def rotate_full_res(img: np.ndarray, angle_deg: float, fill_value: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=fill_value)
