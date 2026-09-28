"""Printed grid pages: fine grid with a heavy grid every few cells, anti-aliased by line coverage, and the exact
positions of the fine-grid crossings."""
import numpy as np

from ..config import GridCfg


def grid_page(w_px, h_px, px_per_mm, c: GridCfg):
    """uint8 (h_px, w_px), 0 ink .. 255 paper."""
    def coverage(coord_mm, half_px, every_mm):
        d = np.abs(coord_mm - np.round(coord_mm / every_mm) * every_mm) * px_per_mm
        return np.clip(half_px - d + 0.5, 0.0, 1.0)

    fine, heavy = 0.5 * c.fine_width_mm * px_per_mm, 0.5 * c.heavy_width_mm * px_per_mm
    xs = np.arange(w_px, dtype=np.float32) / px_per_mm
    ys = np.arange(h_px, dtype=np.float32) / px_per_mm
    cx = np.maximum(coverage(xs, fine, c.fine_mm), coverage(xs, heavy, c.fine_mm * c.heavy_every))
    cy = np.maximum(coverage(ys, fine, c.fine_mm), coverage(ys, heavy, c.fine_mm * c.heavy_every))
    cov = np.maximum(cx[None, :], cy[:, None])
    return np.round(255 * (1 - cov)).astype(np.uint8)


def crossings(w_mm, h_mm, c: GridCfg):
    """(n, 2) fine-grid crossings [mm] inside the sheet."""
    u, v = np.meshgrid(np.arange(0, w_mm + 1e-9, c.fine_mm), np.arange(0, h_mm + 1e-9, c.fine_mm))
    return np.stack([u.ravel(), v.ravel()], -1)
