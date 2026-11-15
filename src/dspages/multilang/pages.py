"""Page composition: hwsynth lines on real PDF pages (forms filled, text gaps) and on handwritten pages"""

import hashlib
from dataclasses import dataclass

import cv2
import numpy as np
import pymupdf

from dspages.pagegen.paper import blend, degrade, free_mask, page_segments  # noqa: F401  (degrade: part of this module's API)

from .config import Config


@dataclass
class Bg:
    img: np.ndarray
    prints: list
    fields: list
    labels: list
    occ: np.ndarray
    rules: list
    pmask: np.ndarray
    checks: list


def is_val(name):
    """Split by the content id of a PDF (file name up to the first `_`), so that its pages are in one split."""
    return int(hashlib.md5(name.split("_")[0].encode()).hexdigest(), 16) % 10 == 0


def render_bg(path, rng, cfg: Config):
    """A page of the PDF with its print boxes in pixels, or None if it is mostly picture or has too little text."""
    with pymupdf.open(path) as doc:
        pg = doc[rng.randrange(min(len(doc), cfg.text_pages_per_file))]
        area = abs(pg.rect)
        if any(abs(pymupdf.Rect(i["bbox"]) & pg.rect) > cfg.max_picture_cover * area for i in pg.get_image_info()):
            return None
        prints, fields, labels = page_segments(pg, cfg)
        lo, hi = cfg.mark_box_pt
        checks = (
            [
                tuple(d["rect"])
                for d in pg.get_drawings()
                if lo <= d["rect"].width <= hi
                and lo <= d["rect"].height <= hi
                and cfg.mark_aspect[0] <= d["rect"].width / d["rect"].height <= cfg.mark_aspect[1]
            ]
            if cfg.ticks
            else []
        )
        k = cfg.long_side / max(pg.rect.width, pg.rect.height)
        px = pg.get_pixmap(matrix=pymupdf.Matrix(k, k), alpha=False)
    img = np.frombuffer(px.samples, np.uint8).reshape(px.h, px.w, px.n)[:, :, :3].copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def seen(b):
        r = gray[max(int(b[1] * k), 0) : int(b[3] * k) + 1, max(int(b[0] * k), 0) : int(b[2] * k) + 1]
        return r.size and np.ptp(r) >= cfg.seen_contrast

    prints = [tuple(v * k for v in b) for b in prints if seen(b)]
    if not cfg.min_prints <= len(prints) <= cfg.max_prints:
        return None
    occ, rules = free_mask(img, prints, cfg)
    pmask = np.zeros(img.shape[:2], np.uint8)
    for x0, y0, x1, y1 in prints:
        pmask[int(y0) : int(y1) + 1, int(x0) : int(x1) + 1] = 1
    sc = lambda bs: [tuple(v * k for v in b) for b in bs]
    overlaps = lambda c: any(c[0] < q[2] and c[2] > q[0] and c[1] < q[3] and c[3] > q[1] for q in prints)
    return Bg(img, prints, sc(fields), sc(labels), occ, rules, pmask, [c for c in sc(checks) if not overlaps(c)])


def scaled_alpha(ink, k):
    """Ink alpha resized by k (<= 1: handwriting is never enlarged, so it is as sharp as the print around it)."""
    assert k <= 1.0 + 1e-9, k
    h, w = int(ink.shape[0] * k), int(ink.shape[1] * k)
    if h < 4 or w < 4:
        return None
    return cv2.resize(ink, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


def paste(img, occ, it, scale, rng, tries, region, cfg: Config):
    """Item on free paper inside region (x0, y0, x1, y1) or the whole page; returns its boxes or None."""
    H, W = img.shape[:2]
    rx0, ry0, rx1, ry1 = map(int, region or (0, 0, W, H))
    if region:
        scale = min(scale, (rx1 - rx0) / it["ink"].shape[1], (ry1 - ry0) / it["ink"].shape[0])
    a = scaled_alpha(it["ink"], scale)
    if a is None or a.shape[0] >= H or a.shape[1] >= W:
        return None
    h, w = a.shape
    for _ in range(tries):
        y = rng.randint(max(ry0, 0), max(min(ry1, H) - h, ry0, 0))
        x = rng.randint(max(rx0, 0), max(min(rx1, W) - w, rx0, 0))
        if y + h <= H and x + w <= W and occ[y : y + h, x : x + w].mean() < cfg.free_cover:
            blend(img, a, it["rgb"], x, y, rng.uniform(*cfg.ink_density))
            occ[y : y + h, x : x + w] = 1
            return [(x + b0 * scale, y + b1 * scale, x + b2 * scale, y + b3 * scale) for b0, b1, b2, b3 in it["boxes"]]
    return None


def line_h(it):
    return it["ink"].shape[0] / len(it["boxes"])


def pick(rng, items, target, used):
    """An item not used yet whose line is at least `target` px tall"""
    free = [it for it in items if id(it) not in used] or items
    ok = [it for it in free if line_h(it) >= target]
    it = rng.choice(ok) if ok else max(free, key=line_h)
    used.add(id(it))
    return it


def fit(it, target):
    """Scale that brings the item's line to `target` px; never above 1 (no enlargement)."""
    return min(target / line_h(it), 1.0)


def print_strip(bg: Bg, rng, cfg: Config):
    """Header or footer rows of a real page with their print boxes, or None."""
    H = bg.img.shape[0]
    top = rng.random() < cfg.p_strip_top
    sel = [b for b in bg.prints if (b[3] < cfg.strip_header * H if top else b[1] > cfg.strip_footer * H)]
    if not sel:
        return None
    if top:
        y0, y1 = 0, int(max(b[3] for b in sel)) + cfg.strip_pad
    else:
        y0, y1 = max(int(min(b[1] for b in sel)) - cfg.strip_pad, 0), H
    return dict(img=bg.img[y0:y1], boxes=[(a, b - y0, c, d - y0) for a, b, c, d in sel], top=top)


def yolo_rows(boxes_by_class, W, H, cfg: Config):
    rows = []
    for c, boxes in boxes_by_class:
        for x0, y0, x1, y1 in boxes:
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
            if x1 - x0 > cfg.min_box_px and y1 - y0 > cfg.min_box_px:
                rows.append(
                    f"{c} {(x0 + x1) / 2 / W:.5f} {(y0 + y1) / 2 / H:.5f} {(x1 - x0) / W:.5f} {(y1 - y0) / H:.5f}"
                )
    return rows


def draw(rng, items, k):
    """k distinct items (fewer if the pool is small)."""
    return rng.sample(items, min(k, len(items)))
