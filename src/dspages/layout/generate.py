"""Part 1: flat page layouts from the manifest (restyled PDF pages and rasterized documents) with paper effects."""
import cv2
import numpy as np

from ..config import Preset
from . import paper
from .content import content_mask, window_stats
from .grid import crossings, grid_page
from .sources import is_pdf, raster_page
import pymupdf

from .template import sheet


def _crop(src, x, y, w, h, pad):
    """src[y:y + h, x:x + w] with the parts outside src filled with pad."""
    H, W = src.shape
    out = np.full((h, w), pad, np.uint8)
    xa, ya, xb, yb = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if xb > xa and yb > ya:
        out[ya - y:yb - y, xa - x:xb - x] = src[ya:yb, xa:xb]
    return out


def _raster_sheet(entry, pw, ph, c):
    """Sheet from a raster page: the manifest's window, PDF pages rendered at the zoom that makes it >= 1:1 (within
    max_zoom)."""
    x, y, w, h = entry["window"]
    pdf = is_pdf(entry)
    k = min(max(pw / w, 1.0), c.max_zoom) if pdf else 1.0
    src = raster_page(entry, k)
    pad = int(np.percentile(src, 95))
    crop = _crop(src, *(round(v * k) for v in (x, y, w, h)), pad)
    gray = cv2.resize(crop, (pw, ph), interpolation=cv2.INTER_AREA if crop.shape[1] >= pw else cv2.INTER_CUBIC)
    return gray, dict(window=(x, y, w, h), zoom=k, fill=entry["fill"], image_share=entry["picture_share"],
                      upscale=float(pw / crop.shape[1]), margin=entry["margin"])


def sheet_fill(gray, grid):
    """Fill of the whole sheet, measured as the raster pages' window: share of grid x grid patches with ink."""
    mask, s = content_mask(gray)
    ink = cv2.integral(mask.astype(np.uint8)).astype(np.int64)
    return float(window_stats(ink, np.zeros_like(ink), s, [(0, 0, gray.shape[1], gray.shape[0])], grid)[0][0])


def make_layout(rng, entry, P: Preset, fonts, spread=False):
    """-> dict(page: uint8 luminance sheet, gray: uint8 clean sheet before effects, meta). Raster pages (window)
    and restyled PDF pages (body size, scale, margin and seed) as the manifest checked them. spread: an open book, the
    restyled page and its neighbour in the PDF side by side in the same fonts."""
    c = P.layout
    pw, ph = c.page_px
    ppm = pw / c.sheet_mm[0]
    meta = dict(source=entry["source_path"], page_index=entry["page_index"], kind=entry["kind"],
                language=entry["language"], category=entry["category"])
    if rng.random() < c.grid_prob:
        m = rng.uniform(*c.margin_frac)
        mx, my = round(m * pw), round(m * ph)
        gray = np.full((ph, pw), 255, np.uint8)
        gray[my:ph - my, mx:pw - mx] = grid_page(pw - 2 * mx, ph - 2 * my, ppm, c.grid)
        meta.update(kind="grid", margin=float(m), fill=1.0,
                    crossings_mm=crossings((pw - 2 * mx) / ppm, (ph - 2 * my) / ppm, c.grid) + [mx / ppm, my / ppm])
    elif entry["kind"] == "raster":
        gray, m = _raster_sheet(entry, pw, ph, c)
        meta.update(m)
    else:
        m, pt, scale = rng.uniform(*c.margin_frac), rng.uniform(*c.font_pt), rng.uniform(*c.role_scale)
        n, seed = entry["page_index"], int(rng.integers(1 << 31))
        pages = [n]
        if spread:
            with pymupdf.open(entry["source_path"]) as doc:
                pages = sorted((n, n + 1 if n + 1 < doc.page_count else max(n - 1, 0)))
        parts = [sheet(entry["source_path"], k, seed, fonts, pt, scale, m, (pw // len(pages), ph)) for k in pages]
        gray, info = np.hstack([g for g, _ in parts]), parts[0][1]
        mx, my = round(m * pw / len(pages)), round(m * ph)
        meta.update(pages=pages, margin=float(m), font_pt=float(pt), scale=float(scale), body_pt=info["body_pt"],
                    fill=sheet_fill(gray[my:ph - my, mx:pw - mx], c.fill_grid))
    page, effects = paper.apply(rng, gray, ppm, c, P.geometry.shallow)
    meta["effects"] = effects
    return dict(page=page, gray=gray, meta=meta)
