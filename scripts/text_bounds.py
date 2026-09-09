"""Text-body bounds detection, ported from smartcrop-pdf-web's TS/OpenCV.js pipeline
(core/detection_service.ts, pdf/imaging.ts, pdf/loader.ts).

Two paths, matching the original's NORMAL vs SCANNED modes:
  - detect_text_box_pdf: native PDF text layer (word boxes union), no rasterisation.
  - detect_content_box_scan: raster path — illumination-flatten + Sauvola binarize +
    despeckle + morphological line-merge + connected components, same constants as
    core/constants.ts (DETECT_MAX_PX, SAUVOLA_*, BW_STRENGTH[2], DETECT_CLOSE_*, ...).
"""
from __future__ import annotations

import cv2
import numpy as np

# --- constants, mirrored from smartcrop-pdf-web/src/core/constants.ts ---
DETECT_MAX_PX = 1400
BORDER_FRAC = 0.02
MIN_COMP_FRAC = 2.5e-4
DETECT_CLOSE_W = 9
DETECT_CLOSE_H = 3
CC_CONNECTIVITY = 8
SAUVOLA_R = 127.5
SAUVOLA_WINDOW = 51
BG_KERNEL_SIZE = 51
BG_DOWNSCALE = 4
BW_STRENGTH_2 = {"k": 0.110, "minArea": 40}


def sauvola_ink_mask(flat: np.ndarray, window: int, k: float) -> np.ndarray:
    """T = mean*(1-k) + (k/R)*mean*std over a window x window box filter; ink = flat < T."""
    win = window | 1
    f32 = flat.astype(np.float32)
    mean = cv2.boxFilter(f32, cv2.CV_32F, (win, win), borderType=cv2.BORDER_REFLECT_101)
    sq_mean = cv2.boxFilter(f32 * f32, cv2.CV_32F, (win, win), borderType=cv2.BORDER_REFLECT_101)
    variance = np.clip(sq_mean - mean * mean, 0, None)
    std = np.sqrt(variance)
    threshold = mean * (1 - k) + (k / SAUVOLA_R) * mean * std
    mask = (f32 < threshold).astype(np.uint8) * 255
    return mask


def morph_close_background(gray: np.ndarray, kernel: int) -> np.ndarray:
    """Background estimate via morphological close on a 1/BG_DOWNSCALE copy, then upscaled."""
    scale = BG_DOWNSCALE
    sh, sw = round(gray.shape[0] / scale), round(gray.shape[1] / scale)
    k_small = round(kernel / scale) | 1
    if scale <= 1 or k_small < 3 or sw < 1 or sh < 1:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
        return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se)
    small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_small, k_small))
    bg_small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, se)
    return cv2.resize(bg_small, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)


def illumination_flatten(gray: np.ndarray, kernel_size: int) -> np.ndarray:
    bg = morph_close_background(gray, kernel_size | 1)
    ratio = gray.astype(np.float32) / (bg.astype(np.float32) + 1e-6) * 255
    return np.clip(ratio, 0, 255).astype(np.uint8)


def clean_document_bilevel(gray: np.ndarray, k: float, min_area: float, window: int,
                            bg_kernel: int) -> np.ndarray:
    """Returns bilevel image: ink=0, background=255 (imaging.py/cv.ts equivalent)."""
    flat = illumination_flatten(gray, bg_kernel)
    ink = sauvola_ink_mask(flat, window, k)

    if min_area > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=CC_CONNECTIVITY)
        keep_label = stats[:, cv2.CC_STAT_AREA] >= min_area
        keep_label[0] = False
        ink = (keep_label[labels] * 255).astype(np.uint8)

    hi = np.full_like(ink, 255)
    hi[ink > 0] = 0
    return hi


def detect_content_box_scan(bgr: np.ndarray, region: tuple[float, float, float, float] | None = None):
    """Content-box detection for a scanned/rasterised page. `bgr` is the full-resolution page
    raster (as it would come off a scanner/renderer). Returns (x0, y0, x1, y1) in `bgr`'s own
    pixel coordinates. `region`, if given, scopes detection to that pixel sub-rectangle."""
    ph, pw = bgr.shape[:2]
    rx0, ry0, rx1, ry1 = region if region else (0, 0, pw, ph)
    rw, rh = rx1 - rx0, ry1 - ry0

    scale = min(1.0, DETECT_MAX_PX / max(rw, rh))
    dw, dh = max(1, round(rw * scale)), max(1, round(rh * scale))

    crop = bgr[round(ry0):round(ry0 + rh), round(rx0):round(rx0 + rw)]
    small = cv2.resize(crop, (dw, dh), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    bilevel = clean_document_bilevel(gray, BW_STRENGTH_2["k"], BW_STRENGTH_2["minArea"],
                                      SAUVOLA_WINDOW, BG_KERNEL_SIZE)
    ink = (bilevel < 127).astype(np.uint8) * 255

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DETECT_CLOSE_W, DETECT_CLOSE_H))
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, se)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=CC_CONNECTIVITY)

    page_area = dw * dh
    min_area = max(8, MIN_COMP_FRAC * page_area)
    border_x = round(BORDER_FRAC * min(dw, dh))
    border_y = border_x

    def boxes(require_interior: bool):
        out = []
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if a < min_area:
                continue
            if require_interior and (x <= border_x or y <= border_y
                                      or x + w >= dw - border_x or y + h >= dh - border_y):
                continue
            out.append((x, y, w, h))
        return out

    kept = boxes(True) or boxes(False)

    if not kept:
        return rx0, ry0, rx1, ry1

    bx0 = min(b[0] for b in kept)
    by0 = min(b[1] for b in kept)
    bx1 = max(b[0] + b[2] for b in kept)
    by1 = max(b[1] + b[3] for b in kept)

    return (rx0 + bx0 / scale, ry0 + by0 / scale, rx0 + bx1 / scale, ry0 + by1 / scale)


def arxiv_sidebar_bbox(page) -> tuple[float, float, float, float] | None:
    """Bounding box of arXiv's injected vertical sidebar identifier block (rotated text split
    across several words in one block: "arXiv:2609.xxxxx", "[gr-qc]", day, month, year), or
    None if the page has none. Used to blank the block out of the raster BEFORE rendering, so
    it never reaches the image at all (not just excluded from the detected text box)."""
    words = page.get_text("words")
    arxiv_blocks = {w[5] for w in words if "arxiv:" in w[4].lower()}
    if not arxiv_blocks:
        return None
    block_words = [w for w in words if w[5] in arxiv_blocks]
    x0 = min(w[0] for w in block_words)
    y0 = min(w[1] for w in block_words)
    x1 = max(w[2] for w in block_words)
    y1 = max(w[3] for w in block_words)
    return x0, y0, x1, y1


def detect_text_box_pdf(page) -> tuple[float, float, float, float] | None:
    """Content-box detection for a native (text-layer) PDF page, via PyMuPDF word boxes —
    equivalent to loader.ts's detect_text_box but without the clip-path/justification-run
    corrections (PyMuPDF's word boxes are already trimmed to visible glyphs).

    Drops the whole text block containing "arxiv:" (case-insensitive) before computing the
    union — arXiv's injected vertical sidebar identifier, not part of the paper's actual text
    body. Callers should also blank it out of the raster via arxiv_sidebar_bbox before
    rendering, so it's excluded from the image, not just from this box."""
    words = page.get_text("words")
    arxiv_blocks = {w[5] for w in words if "arxiv:" in w[4].lower()}
    words = [w for w in words if w[5] not in arxiv_blocks]
    if not words:
        return None
    x0 = min(w[0] for w in words)
    y0 = min(w[1] for w in words)
    x1 = max(w[2] for w in words)
    y1 = max(w[3] for w in words)
    return x0, y0, x1, y1
