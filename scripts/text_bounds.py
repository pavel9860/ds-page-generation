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


def _content_components_scan(bgr: np.ndarray, region: tuple[float, float, float, float] | None):
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
    return kept, ink, scale, rx0, ry0, rx1, ry1


def detect_content_box_scan(bgr: np.ndarray, region: tuple[float, float, float, float] | None = None):
    """Content-box detection for a scanned/rasterised page. `bgr` is the full-resolution page
    raster (as it would come off a scanner/renderer). Returns (x0, y0, x1, y1) in `bgr`'s own
    pixel coordinates. `region`, if given, scopes detection to that pixel sub-rectangle."""
    kept, _, scale, rx0, ry0, rx1, ry1 = _content_components_scan(bgr, region)
    if not kept:
        return rx0, ry0, rx1, ry1

    bx0 = min(b[0] for b in kept)
    by0 = min(b[1] for b in kept)
    bx1 = max(b[0] + b[2] for b in kept)
    by1 = max(b[1] + b[3] for b in kept)

    return (rx0 + bx0 / scale, ry0 + by0 / scale, rx0 + bx1 / scale, ry0 + by1 / scale)


def content_box_scan_coverage(bgr: np.ndarray, region: tuple[float, float, float, float] | None = None):
    """Same box as detect_content_box_scan, plus what fraction of the box's rows/columns
    actually contain ink pixels — catches pages where a couple of isolated marks produce a
    big, mostly-empty union box."""
    kept, ink, scale, rx0, ry0, rx1, ry1 = _content_components_scan(bgr, region)
    if not kept:
        return rx0, ry0, rx1, ry1, 0.0, 0.0

    bx0 = min(b[0] for b in kept)
    by0 = min(b[1] for b in kept)
    bx1 = max(b[0] + b[2] for b in kept)
    by1 = max(b[1] + b[3] for b in kept)

    region_ink = ink[by0:by1, bx0:bx1] > 0
    y_cov = float(np.count_nonzero(region_ink.any(axis=1))) / max(1, by1 - by0)
    x_cov = float(np.count_nonzero(region_ink.any(axis=0))) / max(1, bx1 - bx0)

    return (rx0 + bx0 / scale, ry0 + by0 / scale, rx0 + bx1 / scale, ry0 + by1 / scale, x_cov, y_cov)


PATCH_GRID_COLS = 8
PATCH_GRID_ROWS = 12


def patch_fill_ratio_boxes(boxes: list, box: tuple[float, float, float, float]) -> float:
    """Fraction of a PATCH_GRID_COLS x PATCH_GRID_ROWS grid over `box` touched by at least
    one of `boxes` (each a (x0, y0, x1, y1, ...) tuple — PDF word boxes, image regions, or
    both mixed together) — computed straight from PDF metadata, no rasterisation. A patch
    spanning a text line plus its normal leading still counts as occupied, so inter-line
    spacing is never mistaken for empty space; only genuinely unused regions (dead margins,
    blank paragraphs, half-empty pages) show up as empty patches."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0:
        return 0.0

    occupied = set()
    for b in boxes:
        bx0, by0, bx1, by1 = max(b[0], x0), max(b[1], y0), min(b[2], x1), min(b[3], y1)
        if bx1 <= bx0 or by1 <= by0:
            continue
        c0 = int((bx0 - x0) / bw * PATCH_GRID_COLS)
        c1 = int((bx1 - x0) / bw * PATCH_GRID_COLS - 1e-9)
        r0 = int((by0 - y0) / bh * PATCH_GRID_ROWS)
        r1 = int((by1 - y0) / bh * PATCH_GRID_ROWS - 1e-9)
        for r in range(max(0, r0), min(PATCH_GRID_ROWS - 1, r1) + 1):
            for c in range(max(0, c0), min(PATCH_GRID_COLS - 1, c1) + 1):
                occupied.add((r, c))
    return len(occupied) / (PATCH_GRID_ROWS * PATCH_GRID_COLS)


STROKE_SCALE_FRAC = 0.006
MAX_THICK_INK_FRAC = 0.15


def classify_patches_scan(bgr: np.ndarray, box: tuple[float, float, float, float],
                           dark_thresh: int = 180, min_dark_frac: float = 0.01) -> tuple[int, int, int]:
    """Classifies every grid patch over `box` as empty / photo / text in one pass, returning
    (n_empty, n_photo, n_text). A patch is empty if it has no meaningful dark-pixel content.
    Otherwise it's classified by ink thickness via distance transform: text/line-art strokes
    are thin (a pixel deep inside the ink is still close to a background edge), photographs
    are large continuous-tone blobs (many ink pixels sit far from any edge). The thickness
    threshold scales with the image's own resolution (not a fixed pixel count), so it stays
    correct across sources at very different DPI — a fixed absolute or edge-density-ratio
    threshold misclassifies dense text as "photo" once strokes get physically wider at high
    DPI (the interior of a thick stroke has more non-edge ink pixels)."""
    x0, y0, x1, y1 = [round(v) for v in box]
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0:
        return (PATCH_GRID_ROWS * PATCH_GRID_COLS, 0, 0)

    gray = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    ink = (gray < dark_thresh).astype(np.uint8)
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    stroke_thresh = max(1.0, max(bgr.shape[:2]) * STROKE_SCALE_FRAC)
    thick = dist > stroke_thresh

    n_empty = n_photo = n_text = 0
    for r in range(PATCH_GRID_ROWS):
        py0, py1 = round(r * bh / PATCH_GRID_ROWS), round((r + 1) * bh / PATCH_GRID_ROWS)
        for c in range(PATCH_GRID_COLS):
            px0, px1 = round(c * bw / PATCH_GRID_COLS), round((c + 1) * bw / PATCH_GRID_COLS)
            patch_ink = ink[py0:py1, px0:px1]
            if patch_ink.size == 0:
                n_empty += 1
                continue
            dark = patch_ink.mean()
            if dark <= min_dark_frac:
                n_empty += 1
                continue
            thick_frac = thick[py0:py1, px0:px1].sum() / max(1, patch_ink.sum())
            if thick_frac <= MAX_THICK_INK_FRAC:
                n_text += 1
            else:
                n_photo += 1
    return n_empty, n_photo, n_text


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


def oceanofpdf_watermark_bboxes(page) -> list[tuple[float, float, float, float]]:
    """Bounding box(es) of "OceanofPDF.com" watermark blocks injected into pirated book
    PDFs (often the only thing on an otherwise blank page). Used to blank them out of the
    raster BEFORE rendering, the same way arxiv_sidebar_bbox strips the arXiv sidebar."""
    words = page.get_text("words")
    blocks = {w[5] for w in words if "oceanofpdf" in w[4].lower()}
    if not blocks:
        return []
    out = []
    for block_no in blocks:
        block_words = [w for w in words if w[5] == block_no]
        x0 = min(w[0] for w in block_words)
        y0 = min(w[1] for w in block_words)
        x1 = max(w[2] for w in block_words)
        y1 = max(w[3] for w in block_words)
        out.append((x0, y0, x1, y1))
    return out


def detect_text_box_pdf(page) -> tuple[float, float, float, float] | None:
    """Content-box detection for a native (text-layer) PDF page, via PyMuPDF word boxes —
    equivalent to loader.ts's detect_text_box but without the clip-path/justification-run
    corrections (PyMuPDF's word boxes are already trimmed to visible glyphs).

    Drops the whole text block containing "arxiv:" (case-insensitive) before computing the
    union — arXiv's injected vertical sidebar identifier, not part of the paper's actual text
    body. Callers should also blank it out of the raster via arxiv_sidebar_bbox before
    rendering, so it's excluded from the image, not just from this box."""
    box, _ = _text_box_pdf_words(page)
    return box


def _text_box_pdf_words(page):
    words = page.get_text("words")
    drop_blocks = {w[5] for w in words if "arxiv:" in w[4].lower() or "oceanofpdf" in w[4].lower()}
    words = [w for w in words if w[5] not in drop_blocks]
    if not words:
        return None, None
    x0 = min(w[0] for w in words)
    y0 = min(w[1] for w in words)
    x1 = max(w[2] for w in words)
    y1 = max(w[3] for w in words)
    return (x0, y0, x1, y1), words
