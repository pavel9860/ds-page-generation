"""Training samples: a digital page rendered clean (labels) and degraded like a scan (features)."""
import cv2
import numpy as np
import pymupdf

from ..layout import refont
from .config import CONFIG
from .detector import jpeg_roundtrip
from .maps import ink, page_maps
from .render import render_page
from .tiles import tile_cells, tile_mean, window
from .util import time_limit


def degrade(g, rng, cfg=CONFIG):
    """A rendered grey page -> the same page as a deskewed scan: resolution loss, blur, paper tone and shadows, noise, dust, JPEG, sometimes binarised."""
    t = cfg.train
    h, w = g.shape
    s = rng.uniform(*t.degrade_scale)
    if s < 0.97:
        g = cv2.resize(cv2.resize(g, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < t.degrade_blur_prob:
        g = cv2.GaussianBlur(g, (0, 0), rng.uniform(*t.degrade_blur_sigma))
    f = g.astype(np.float32) * rng.uniform(*t.degrade_gain) + rng.uniform(*t.degrade_offset)
    xx, yy = np.linspace(-1, 1, w, dtype=np.float32)[None, :], np.linspace(-1, 1, h, dtype=np.float32)[:, None]
    a = t.degrade_shade
    f *= 1 + rng.uniform(-a, a) * xx + rng.uniform(-a, a) * yy + rng.uniform(-t.degrade_shade_xy, t.degrade_shade_xy) * xx * yy
    f += rng.normal(0, rng.uniform(*t.degrade_noise), f.shape).astype(np.float32)
    if rng.random() < t.degrade_dust_prob:
        m = rng.random(f.shape) < rng.uniform(*t.degrade_dust)
        f[m] = rng.choice([0, 255], int(m.sum()))
    g = jpeg_roundtrip(np.clip(f, 0, 255).astype(np.uint8), int(rng.integers(*t.degrade_jpeg)))
    if rng.random() < t.degrade_binarise_prob:
        g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return g


def picture_rects(page, src, number, cfg=CONFIG):
    """(rects (x0, y0, x1, y1) of the pictures of the PDF page as shares of the page, whether one is page-sized)."""
    pw, ph = page.rect.width, page.rect.height
    r = [(d["rect"].x0 / pw, d["rect"].y0 / ph, d["rect"].x1 / pw, d["rect"].y1 / ph) for d in refont.pictures(src, number, ph, pw)]
    return r, any((x1 - x0) * (y1 - y0) > cfg.geometry.page_sized_picture for x0, y0, x1, y1 in r)


def picture_cells(rects, win, shape, cfg=CONFIG):
    """(cells, cells) bool: the digital pic mask on the window `win` (px of an image of `shape`): every cell a picture touches."""
    n = cfg.geometry.picture_cells
    x, y, w, h = win
    m = np.zeros((n, n), bool)
    for x0, y0, x1, y1 in rects:
        a, b = int(np.floor((x0 * shape[1] - x) / w * n)), int(np.ceil((x1 * shape[1] - x) / w * n))
        e, f = int(np.floor((y0 * shape[0] - y) / h * n)), int(np.ceil((y1 * shape[0] - y) / h * n))
        a, b, e, f = max(a, 0), min(b, n), max(e, 0), min(f, n)
        if b > a and f > e:
            m[e:f, a:b] = True
    return m


def picture_tiles(rects, win, shape, cfg=CONFIG):
    """(grid, grid) share of every tile covered by pictures; the mean over the tiles is the digital pic."""
    g, n = cfg.geometry.grid, cfg.geometry.picture_cells
    return picture_cells(rects, win, shape, cfg).reshape(g, n // g, g, n // g).mean((1, 3))


def word_mask(src, page, number, shape):
    wm = np.zeros(shape, np.uint8)
    scale = np.array([shape[1] / page.rect.width, shape[0] / page.rect.height] * 2)
    for x0, y0, x1, y1 in np.array([w[:4] for w in src.words(number)], float).reshape(-1, 4) * scale:
        wm[max(int(y0), 0):int(np.ceil(y1)), max(int(x0), 0):int(np.ceil(x1))] = 1
    return wm


def text_tiles(wm, win, shape, cfg=CONFIG):
    """(grid, grid) bool: tiles that hold words of the PDF. Text on a picture makes its tile a text tile, no picture tile."""
    return tile_mean(cv2.resize(wm, (shape[1], shape[0]), interpolation=cv2.INTER_AREA), win, cfg.geometry.grid) > cfg.ink.eps


def sample(job, cfg=CONFIG):
    """(path, page, seed) -> per window (A: that of the clean page with a random margin, B: that of the scan's own ink) the tile means of
    the maps of the degraded page and the labels of the clean one: ink (grid * grid bool) and pic (grid * grid share). None when it fails."""
    path, number, seed = job[:3]
    rng = np.random.default_rng(seed)
    try:
        with time_limit(cfg.train.sample_seconds), pymupdf.open(path) as doc:
            src = refont.Source(doc)
            page = src[number]
            clean = render_page(page, cfg)
            rects, big = picture_rects(page, src, number, cfg)
            wm = word_mask(src, page, number, clean.shape)
            ink0 = ink(clean, cfg)
            scan = jpeg_roundtrip(degrade(clean, rng, cfg), cfg.geometry.jpeg_quality)
            ink1 = ink(scan, cfg)
            maps, paper, noise = page_maps(scan, ink1, cfg)
            out = dict(path=path, page=number, big=big, scal=np.array([paper, noise], np.float32))
            for v, win in (("A", window(ink0.mask, rng.uniform(0, cfg.geometry.margin), cfg)), ("B", window(ink1.mask, cfg.geometry.infer_margin, cfg))):
                if win is None:
                    return None
                out[f"cells{v}"] = tile_cells(maps, win, scan.shape, cfg)
                out[f"ink{v}"] = (tile_mean(ink0.mask, win, cfg.geometry.grid) > cfg.ink.eps).ravel()
                out[f"pic{v}"] = (picture_tiles(rects, win, ink0.mask.shape, cfg) * ~text_tiles(wm, win, ink0.mask.shape, cfg)).ravel().astype(np.float32)
            return out
    except BaseException:
        return None
