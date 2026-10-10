"""Part 1: flat page layouts from the manifest (restyled PDF pages and rasterized documents) with paper effects."""
import cv2
import numpy as np

from ..config import Preset
from . import paper
from .content import SCAN_MAX_SIDE, bbox, content_mask
from .grid import crossings, grid_page
from .sources import is_pdf, raster_page
import pymupdf

from .template import draw_sheet, sheet_layout

INK_BORDER = 0.005


def _crop(src, x, y, w, h, pad):
    """src[y:y + h, x:x + w] with the parts outside src filled with pad."""
    H, W = src.shape
    out = np.full((h, w), pad, np.uint8)
    xa, ya, xb, yb = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if xb > xa and yb > ya:
        out[ya - y:yb - y, xa - x:xb - x] = src[ya:yb, xa:xb]
    return out


def _raster_sheet(entry, pw, ph, c):
    """Sheet (pw, ph) px from a raster page: the manifest's window, PDF pages rendered at the zoom that makes it the
    sheet's size (within max_zoom), images resized."""
    x, y, w, h = entry["window"]
    k = min(pw / w, c.max_zoom) if is_pdf(entry) else 1.0
    src = raster_page(entry, k)
    exact = k == pw / w                                     # rendered at the sheet's size: cut, not resized
    crop = _crop(src, round(x * k), round(y * k), *((pw, ph) if exact else (round(w * k), round(h * k))), int(np.percentile(src, 95)))
    if crop.shape != (ph, pw):
        crop = cv2.resize(crop, (pw, ph), interpolation=cv2.INTER_AREA if crop.shape[1] >= pw else cv2.INTER_CUBIC)
    return crop, dict(window=(x, y, w, h), zoom=k, upscale=float(pw / round(w * k)))


def _raster_ink(entry):
    """Ink box of a raster page's window, shares of the window (x0, y0, x1, y1); a band of INK_BORDER along the
    window's edges (scan borders, dark page edges) is left out."""
    x, y, w, h = entry["window"]
    k = min(1.0, SCAN_MAX_SIDE / max(w, h)) if is_pdf(entry) else 1.0
    src = raster_page(entry, k)
    crop = _crop(src, *(round(v * k) for v in (x, y, w, h)), int(np.percentile(src, 95)))
    mask, _ = content_mask(crop)
    by, bx = (max(1, round(INK_BORDER * n)) for n in mask.shape)
    b = bbox(mask[by:-by, bx:-bx])
    if b is not None:
        b = (b[0] + by, b[1] + by, b[2] + bx, b[3] + bx)
    if b is None:
        return (0.0, 0.0, 1.0, 1.0)
    return (b[2] / mask.shape[1], b[0] / mask.shape[0], (b[3] + 1) / mask.shape[1], (b[1] + 1) / mask.shape[0])


def make_layout(rng, entry, P: Preset, fonts, spread=False, font_pt=None):
    """-> dict(desc: what draw_layout draws at any scale, meta). Raster pages (window) and restyled PDF pages (body size,
    scale, margin and seed) as the manifest checked them, grid pages. spread: an open book, the restyled page and its
    neighbour in the PDF side by side in the same fonts; font_pt: a restyled page's body size. desc["ink"]: the ink box
    (x0, y0, x1, y1) as shares of the sheet."""
    c = P.layout
    pw, ph = c.page_px
    ppm = pw / c.sheet_mm[0]
    meta = dict(source=entry["source_path"], page_index=entry["page_index"], kind=entry["kind"],
                language=entry["language"], category=entry["category"])
    desc = dict(size=(pw, ph), ppm=ppm, entry=entry)
    if rng.random() < c.grid_prob:
        m = rng.uniform(*c.margin_frac)
        mx, my = round(m * pw), round(m * ph)
        desc.update(kind="grid", margin=(mx, my), ink=(mx / pw, my / ph, 1 - mx / pw, 1 - my / ph))
        meta.update(kind="grid", margin=float(m), fill=1.0,
                    crossings_mm=crossings((pw - 2 * mx) / ppm, (ph - 2 * my) / ppm, c.grid) + [mx / ppm, my / ppm])
    elif entry["kind"] == "raster":
        x, y, w, h = entry["window"]
        desc.update(kind="raster", ink=_raster_ink(entry))
        meta.update(window=(x, y, w, h), fill=entry["fill"], image_share=entry["picture_share"], margin=entry["margin"])
    else:
        m = rng.uniform(*c.margin_frac)
        n, seed = entry["page_index"], int(rng.integers(1 << 31))
        pages = [n]
        if spread:
            with pymupdf.open(entry["source_path"]) as doc:
                pages = sorted((n, n + 1 if n + 1 < doc.page_count else max(n - 1, 0)))
        hw = pw // len(pages)
        parts = [sheet_layout(entry["source_path"], k, seed, fonts, m, (hw, ph), font_pt and font_pt * ppm * 25.4 / 72)
                 for k in pages]
        ink = np.array([[d["ink"][0] + i * hw, d["ink"][1], d["ink"][2] + i * hw, d["ink"][3]] for i, (d, _) in enumerate(parts)])
        desc.update(kind="refont", parts=[d for d, _ in parts],
                    ink=(ink[:, 0].min() / pw, ink[:, 1].min() / ph, ink[:, 2].max() / pw, ink[:, 3].max() / ph))
        meta.update(pages=pages, margin=float(m), fill=entry["fill"], body_pt=round(parts[0][1]["body_px"] / ppm * 72 / 25.4, 1))
    desc["effects_seed"] = int(rng.integers(1 << 31))
    desc["rotate180"] = meta["rotate180"] = bool(rng.random() < c.rotate180_prob)
    if desc["rotate180"]:
        x0, y0, x1, y1 = desc["ink"]
        desc["ink"] = (1 - x1, 1 - y1, 1 - x0, 1 - y0)
        if "crossings_mm" in meta:
            meta["crossings_mm"] = np.array(c.sheet_mm) - meta["crossings_mm"]
        if "pages" in meta:
            meta["pages"] = meta["pages"][::-1]
    return dict(desc=desc, meta=meta)


def draw_layout(desc, s, P: Preset):
    """A make_layout description drawn s times the sheet's size (round(page_px * s)): -> (uint8 luminance page with the
    paper effects, uint8 clean sheet, names of the effects)."""
    c = P.layout
    pw, ph = (round(v * s) for v in desc["size"])
    ppm = desc["ppm"] * s
    if desc["kind"] == "grid":
        mx, my = (round(v * s) for v in desc["margin"])
        gray = np.full((ph, pw), 255, np.uint8)
        gray[my:ph - my, mx:pw - mx] = grid_page(pw - 2 * mx, ph - 2 * my, ppm, c.grid)
    elif desc["kind"] == "raster":
        gray, _ = _raster_sheet(desc["entry"], pw, ph, c)
    else:
        gray = np.full((ph, pw), 255, np.uint8)
        hw = desc["size"][0] // len(desc["parts"])
        for i, d in enumerate(desc["parts"]):
            part = draw_sheet(d, s)
            x0 = round(i * hw * s)
            gray[:, x0:x0 + part.shape[1]] = part[:ph, :pw - x0]
    page, effects = paper.apply(np.random.default_rng(desc["effects_seed"]), gray, ppm, c, P.geometry.shallow)
    if desc["rotate180"]:
        page, gray = np.ascontiguousarray(page[::-1, ::-1]), np.ascontiguousarray(gray[::-1, ::-1])
    return page, gray, effects
