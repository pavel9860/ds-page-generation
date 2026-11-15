"""Per-pixel feature maps of a grey page (tile means of these are the features) and the ink mask they share."""

from collections import namedtuple

import cv2
import numpy as np

from .config import CONFIG

Ink = namedtuple("Ink", "mask small hat")


def map_groups(cfg=CONFIG):
    m = cfg.maps
    k, t = m.stroke_ksize, m.stroke_thresh
    return {
        "base": [f"hat{x}" for x in m.hat_thresholds]
        + ["hat", "gray", "spread", "sobel_x", "sobel_y", "laplace", "midgrey", "dark", "paper_diff"],
        "hat_ms": [f"hat{k}", f"hat{k}_{t}"],
        "struct": ["grad_xx", "grad_yy", "coherence"],
        "hist": [f"g<{e}" for e in m.hist_edges] + [f"g>={m.hist_edges[-1]}"],
        "canny": ["canny_hi", "canny_lo"],
        "std_ms": [f"std{x}" for x in m.std_kernels],
        "comps": ["blob", "letters"],
        "blobs": ["dark_open", "mid_open"],
        "runs": ["run_h", "run_v"],
    }


def map_names(cfg=CONFIG):
    return [n for g in map_groups(cfg).values() for n in g]


def _ellipse(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def hat(img, size):
    k = _ellipse(size)
    return cv2.max(cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, k), cv2.morphologyEx(img, cv2.MORPH_TOPHAT, k))


def ink(gray, cfg=CONFIG):
    """Ink mask on a copy downscaled to mask_side; also returns that copy and the line response the mask is cut from."""
    g, i = cfg.geometry, cfg.ink
    s = min(1.0, g.mask_side / max(gray.shape))
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    h = hat(small, i.line_ksize)
    return Ink(h > i.line_thresh, small, h)


def canny(img, lo, hi, min_threads):
    """cv2.Canny on at least min_threads threads: on one it corrupts the heap on pages with a very dense edge map (OpenCV 4.14 and 5.0)."""
    n = cv2.getNumThreads()
    cv2.setNumThreads(max(n, min_threads))
    try:
        return cv2.Canny(img, lo, hi)
    finally:
        cv2.setNumThreads(n)


def page_maps(g, ink_, cfg=CONFIG):
    """({name: (map, at page size)}, paper tone, noise). Strokes (hat, connected strokes, runs) are found at page size, the rest on the downscaled copy."""
    m = cfg.maps
    sm = ink_.small
    f = sm.astype(np.float32)
    paper = float(np.percentile(g[:: m.paper_stride, :: m.paper_stride], m.paper_percentile))
    lap = np.abs(cv2.Laplacian(f, cv2.CV_32F, ksize=3))
    line = ink_.hat.astype(np.float32)
    box = lambda a: cv2.boxFilter(a, -1, (m.box_size, m.box_size))
    mu, sq = box(f), box(f * f)
    ix, iy = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    jxx, jyy, jxy = box(ix * ix), box(iy * iy), box(ix * iy)
    stroke = hat(g, m.stroke_ksize).astype(np.float32)
    strokes = (stroke > m.stroke_thresh).astype(np.uint8)
    _, lab, st, _ = cv2.connectedComponentsWithStats(strokes, connectivity=8)
    area = st[:, cv2.CC_STAT_AREA].astype(np.int64)
    area[0] = 0
    size = area[lab]
    ell = _ellipse(m.open_ksize)
    lo, hi = m.mid_grey
    mid = (sm > lo) & (sm < hi)
    std = lambda k: (
        np.sqrt(np.maximum(cv2.boxFilter(f * f, -1, (k, k)) - cv2.boxFilter(f, -1, (k, k)) ** 2, 0)) / m.spread_scale
    )
    names = map_groups(cfg)
    out = {f"hat{t}": (line > t, 0) for t in m.hat_thresholds}
    out["hat"] = (line / 255, 0)
    out.update(
        {
            "gray": (f / 255, 0),
            "spread": (np.sqrt(np.maximum(sq - mu * mu, 0)) / m.spread_scale, 0),
            "sobel_x": (np.abs(ix) / m.sobel_scale, 0),
            "sobel_y": (np.abs(iy) / m.sobel_scale, 0),
            "laplace": (lap / 255, 0),
            "midgrey": (mid & (np.abs(f - mu) < m.mid_flat), 0),
            "dark": (sm < m.dark_level, 0),
            "paper_diff": (np.clip((paper - f) / 255, 0, 1), 0),
        }
    )
    out[names["hat_ms"][0]], out[names["hat_ms"][1]] = (stroke / 255, 1), (stroke > m.stroke_thresh, 1)
    out.update(
        {
            "grad_xx": (jxx / m.grad_scale, 0),
            "grad_yy": (jyy / m.grad_scale, 0),
            "coherence": (np.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy) / (jxx + jyy + m.coherence_eps), 0),
        }
    )
    edges = m.hist_edges
    bounds = (0,) + tuple(edges) + (256,)
    for name, a, b in zip(names["hist"], bounds[:-1], bounds[1:]):
        out[name] = ((sm >= a) & (sm < b), 0)
    (hi_lo, hi_hi), (lo_lo, lo_hi) = m.canny
    out["canny_hi"], out["canny_lo"] = (
        (canny(sm, hi_lo, hi_hi, m.canny_min_threads) > 0, 0),
        (canny(sm, lo_lo, lo_hi, m.canny_min_threads) > 0, 0),
    )
    for name, k in zip(names["std_ms"], m.std_kernels):
        out[name] = (std(k), 0)
    out["blob"], out["letters"] = (size >= m.blob_area, 1), ((size > 0) & (size < m.blob_area), 1)
    out["dark_open"] = (cv2.morphologyEx((sm < m.dark_level).astype(np.uint8), cv2.MORPH_OPEN, ell), 0)
    out["mid_open"] = (cv2.morphologyEx(mid.astype(np.uint8), cv2.MORPH_OPEN, ell), 0)
    out["run_h"] = (cv2.morphologyEx(strokes, cv2.MORPH_OPEN, np.ones((1, m.run_length), np.uint8)), 1)
    out["run_v"] = (cv2.morphologyEx(strokes, cv2.MORPH_OPEN, np.ones((m.run_length, 1), np.uint8)), 1)
    return out, paper, float(np.median(lap[:: m.noise_stride, :: m.noise_stride]))
