"""Per-tile feature columns: the tile and its neighbours, context blurs, window means, page values and place."""

import cv2
import numpy as np

from .config import CONFIG
from .maps import map_names


def derive(cells, scal, cfg=CONFIG):
    """(cells (n, G, G, M), scal (n, 2) paper tone and noise) -> (n * G * G, F) float32, columns as in `columns`."""
    G = cfg.geometry.grid
    n, m = len(cells), cells.shape[-1]
    pad = np.pad(cells, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
    blocks = [pad[:, i : i + G, j : j + G] for i in range(3) for j in range(3)]
    blur = lambda k: np.stack([cv2.blur(c, (k, k), borderType=cv2.BORDER_REPLICATE).reshape(G, G, m) for c in cells])
    blocks += [blur(k) for k in cfg.features.blur_sizes] + [
        np.broadcast_to(cells.mean((1, 2))[:, None, None, :], cells.shape)
    ]
    page = np.broadcast_to(
        np.concatenate([scal / 255, cells[..., :1].mean((1, 2, 3))[:, None]], 1)[:, None, None, :], (n, G, G, 3)
    )
    r, c = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    pos = np.broadcast_to(
        np.stack([r / (G - 1), c / (G - 1), np.minimum.reduce([r, c, G - 1 - r, G - 1 - c]) / ((G - 1) / 2)], -1)[None],
        (n, G, G, 3),
    )
    return np.concatenate(blocks + [page, pos], -1).reshape(n * G * G, -1).astype(np.float32)


def columns(cfg=CONFIG):
    """[(derived group, map name or '')] for the columns of derive."""
    names = map_names(cfg)
    ctx = [f"ctx{k}" for k in cfg.features.blur_sizes]
    return (
        [("n3", n) for _ in range(9) for n in names]
        + [(c, n) for c in ctx for n in names]
        + [("pagemean", n) for n in names]
        + [("page", "")] * 3
        + [("pos", "")] * 3
    )


def chosen(maps, derived, cfg=CONFIG):
    """Boolean column mask of the map names and derived groups."""
    return np.array([(d in derived) and (n == "" or n in maps) for d, n in columns(cfg)])
