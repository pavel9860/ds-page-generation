"""Part 1: flat page layouts from the manifest (book texts and rasterized documents) with paper effects."""
import cv2
import numpy as np

from ..config import Preset, page_px
from . import paper
from .content import bbox, content_mask, place_window
from .pictures import picture_mask
from .grid import crossings, grid_page
from .sources import is_pdf, raster_page
from .text import book_page, render_text


def _crop(src, x, y, w, h, pad):
    """src[y:y + h, x:x + w] with the parts outside src filled with pad."""
    H, W = src.shape
    out = np.full((h, w), pad, np.uint8)
    xa, ya, xb, yb = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if xb > xa and yb > ya:
        out[ya - y:yb - y, xa - x:xb - x] = src[ya:yb, xa:xb]
    return out


def _raster_sheet(rng, entry, pw, ph, c):
    """Sheet from a raster page: the window of place_window, PDF pages re-rendered so the window is >= 1:1."""
    src = raster_page(entry)
    mask, s = content_mask(src)
    b = bbox(mask)
    y0, y1, x0, x1 = (v / s for v in b)
    cap = c.max_zoom if is_pdf(entry) else 1.0
    pics = picture_mask(src, mask.shape, s, c.pictures)
    (x, y, w, h), fill, img = place_window(rng, (x0, y0, x1 - x0 + 1 / s, y1 - y0 + 1 / s), pw / ph,
                                           c.margin_frac[1], pw / cap, mask, s, pics, c.min_fill, c.fill_grid,
                                           c.max_image)
    k = min(max(pw / w, 1.0), cap)
    if k > 1:
        src = raster_page(entry, k)
    pad = int(np.percentile(src, 95))
    crop = _crop(src, *(round(v * k) for v in (x, y, w, h)), pad)
    gray = cv2.resize(crop, (pw, ph), interpolation=cv2.INTER_AREA if crop.shape[1] >= pw else cv2.INTER_CUBIC)
    margin = float(max(x0 - x, x + w - x1, y0 - y, y + h - y1, 0.0) / w)
    return gray, dict(window=(x, y, w, h), zoom=k, fill=fill, image_share=img, upscale=float(pw / crop.shape[1]),
                      margin=margin)


def make_layout(rng, entry, P: Preset, fonts):
    """-> dict(page: uint8 luminance sheet, gray: uint8 clean sheet before effects, meta)."""
    c = P.layout
    pw, ph = page_px(c.sheet_mm, c.canvas_px)
    ppm = pw / c.sheet_mm[0]
    meta = dict(source=entry["source_path"], page_index=entry["page_index"], kind=entry["kind"],
                language=entry["language"], category=entry["category"])
    grid = rng.random() < c.grid_prob
    if entry["kind"] == "raster" and not grid:
        gray, m = _raster_sheet(rng, entry, pw, ph, c)
        meta.update(m)
    else:
        m = rng.uniform(*c.margin_frac)
        mx, my = round(m * pw), round(m * ph)
        iw, ih = pw - 2 * mx, ph - 2 * my
        if not grid:
            pt, ls = rng.uniform(*c.font_pt), rng.uniform(*c.line_spacing)
            text = book_page(entry["source_path"], entry["page_index"], c.book_page_chars)
            inner = render_text(text, rng, iw, ih, ppm, fonts, pt, ls)
            meta.update(font_pt=float(pt), line_spacing=float(ls))
        else:
            inner = grid_page(iw, ih, ppm, c.grid)
            meta.update(kind="grid", crossings_mm=(crossings(iw / ppm, ih / ppm, c.grid) + [mx / ppm, my / ppm]))
        gray = np.full((ph, pw), 255, np.uint8)
        gray[my:my + ih, mx:mx + iw] = inner
        meta.update(margin=float(m), fill=1.0)
    page, effects = paper.apply(rng, gray, ppm, c, P.geometry.shallow)
    meta["effects"] = effects
    return dict(page=page, gray=gray, meta=meta)
