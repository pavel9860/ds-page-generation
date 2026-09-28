"""Part 1: flat page layouts from the manifest (book texts and rasterized documents) with paper effects."""
import cv2
import numpy as np

from ..config import Preset, page_px
from . import paper
from .content import content_mask, select_window
from .grid import crossings, grid_page
from .sources import raster_page
from .text import char_budget, render_text, sample_snippet


def make_layout(rng, entry, P: Preset, fonts):
    """-> dict(page: uint8 luminance sheet, gray: uint8 clean sheet before effects, meta)."""
    c = P.layout
    pw, ph = page_px(c.sheet_mm, c.canvas_px)
    ppm = pw / c.sheet_mm[0]
    m = rng.uniform(*c.margin_frac)
    mx, my = round(m * pw), round(m * ph)
    iw, ih = pw - 2 * mx, ph - 2 * my
    meta = dict(source=entry["source_path"], page_index=entry["page_index"], kind=entry["kind"],
                language=entry["language"], category=entry["category"], margin=float(m))
    if rng.random() < c.grid_prob:
        inner = grid_page(iw, ih, ppm, c.grid)
        meta.update(kind="grid", fill=1.0, crossings_mm=(crossings(iw / ppm, ih / ppm, c.grid) + [mx / ppm, my / ppm]))
    elif entry["kind"] == "book_text":
        pt, ls = rng.uniform(*c.font_pt), rng.uniform(*c.line_spacing)
        text = sample_snippet(entry["source_path"], rng, char_budget(iw, ih, ppm, pt, ls))
        inner = render_text(text, rng, iw, ih, ppm, fonts, pt, ls)
        meta.update(font_pt=float(pt), line_spacing=float(ls), fill=1.0)
    else:
        src = raster_page(entry)
        mask, s = content_mask(src)
        (y0, x0, h, w), fill = select_window(mask, iw / ih, rng, c.min_fill, c.fill_grid)
        y0, x0, h, w = (round(v / s) for v in (y0, x0, h, w))
        crop = src[y0:y0 + h, x0:x0 + w]
        inner = cv2.resize(crop, (iw, ih), interpolation=cv2.INTER_AREA if crop.shape[1] > iw else cv2.INTER_CUBIC)
        meta.update(window=(y0, x0, h, w), fill=fill)
    gray = np.full((ph, pw), 255, np.uint8)
    gray[my:my + ih, mx:mx + iw] = inner
    page, effects = paper.apply(rng, gray, ppm, c, P.geometry.shallow)
    meta["effects"] = effects
    return dict(page=page, gray=gray, meta=meta)
