"""The window around the content and the tile means of maps over it."""
import numpy as np

from .config import CONFIG
from .maps import map_names


def window(mask, margin, cfg=CONFIG):
    """(x, y, w, h) in mask px: the ink box (rows and columns with at least min_ink_px ink px) widened to the canvas aspect, top-aligned, with `margin` (share of the
    window) around it; None when there is no ink."""
    g = cfg.geometry
    rows, cols = np.flatnonzero(mask.sum(1) >= g.min_ink_px), np.flatnonzero(mask.sum(0) >= g.min_ink_px)
    if not len(rows) or not len(cols):
        return None
    x0, x1, y0, y1 = cols[0], cols[-1] + 1, rows[0], rows[-1] + 1
    w = max(x1 - x0, (y1 - y0) * g.canvas_w / g.canvas_h) / (1 - 2 * margin)
    return (x0 + x1 - w) / 2, y0 - margin * w * g.canvas_h / g.canvas_w, w, w * g.canvas_h / g.canvas_w


def tile_matrices(win, shape, grid):
    """(Ty (grid, H), Tx (grid, W), tile area): 0/1 matrices summing the rows and columns of every tile of the window (px of an image of `shape`; the part outside the
    image is empty). A map's tile sums are Ty @ map @ Tx.T."""
    x, y, w, h = win
    t = np.linspace(0, 1, grid + 1)
    ys, xs = np.clip(np.round(y + h * t).astype(int), 0, shape[0]), np.clip(np.round(x + w * t).astype(int), 0, shape[1])
    ry, rx = np.arange(shape[0]), np.arange(shape[1])
    return ((ry >= ys[:-1, None]) & (ry < ys[1:, None])).astype(np.float32), ((rx >= xs[:-1, None]) & (rx < xs[1:, None])).astype(np.float32), max(w * h / grid / grid, 1)


def tile_mean(m, win, grid=CONFIG.geometry.grid):
    Ty, Tx, area = tile_matrices(win, m.shape, grid)
    return (Ty @ m.astype(np.float32, copy=False) @ Tx.T / area).astype(np.float32)


def tile_cells(maps, win, shape, cfg=CONFIG):
    """(grid, grid, n maps): tile means of every map over the window `win` (px of the downscaled copy); `shape` is the page's."""
    grid = cfg.geometry.grid
    ref = maps["hat"][0].shape
    rx, ry = shape[1] / ref[1], shape[0] / ref[0]
    mats = {0: tile_matrices(win, ref, grid), 1: tile_matrices((win[0] * rx, win[1] * ry, win[2] * rx, win[3] * ry), shape, grid)}
    out = []
    for n in map_names(cfg):
        m, page_size = maps[n]
        Ty, Tx, area = mats[page_size]
        out.append(Ty @ m.astype(np.float32, copy=False) @ Tx.T / area)
    return np.stack(out, -1).astype(np.float32)
