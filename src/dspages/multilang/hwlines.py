"""Boxes around handwritten text lines from the ink"""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from scipy.signal import find_peaks

from .config import HWLINES as C
from .paths import UVTM


def ink_of(rgb):
    """Binary ink of a page without its ruling."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, C.block, C.offset)
    rules = np.zeros_like(ink)
    for k in ((gray.shape[1] // C.rule_div, 1), (1, gray.shape[0] // C.rule_div)):
        rules |= cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k))
    ink = cv2.subtract(ink, rules)
    return cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((C.open_kernel, C.open_kernel), np.uint8))


def _profile(ink):
    return cv2.GaussianBlur((ink > 0).sum(1).astype(np.float32)[:, None], (1, 0), C.sigma)[:, 0]


def _columns(ink, y0, y1, out, cap=None):
    """Splits the row between y0 and y1 into boxes at wide gaps"""
    ys = np.flatnonzero((ink[y0:y1] > 0).sum(1))
    if not len(ys):
        return
    h = ys[-1] + 1 - ys[0]
    h = min(h, cap) if cap else h
    cols = np.flatnonzero((ink[y0:y1] > 0).sum(0))
    cuts = np.flatnonzero(np.diff(cols) > C.gap * h)
    for a, b in zip(np.r_[0, cuts + 1], np.r_[cuts, len(cols) - 1]):
        x0, x1 = cols[a], cols[b] + 1
        rows = np.flatnonzero((ink[y0:y1, x0:x1] > 0).sum(1))
        if len(rows) and x1 - x0 >= C.min_width * h and rows[-1] + 1 - rows[0] >= C.min_height * h:
            out.append((int(x0), int(y0 + rows[0]), int(x1), int(y0 + rows[-1] + 1)))


def _bands_by_level(ink):
    prof = _profile(ink)
    on = prof > max(prof.max() * C.on_share, C.on_min)
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]) if C.min_h <= b - a <= C.max_h]


def _bands_by_valley(ink):
    prof = _profile(ink)
    z = prof - prof.mean()
    ac = np.correlate(z, z, "full")[len(z) - 1:]
    lo, hi = C.lag[0], min(C.lag[1], len(z) // 2)
    if hi <= lo + 2 or ac[0] <= 0 or ac[lo:hi].max() <= C.ac_min * ac[0]:
        return None
    pitch = lo + int(np.argmax(ac[lo:hi]))
    valleys, _ = find_peaks(-prof, distance=max(int(C.valley * pitch), 5), prominence=C.prominence * prof.max())
    cuts = [0] + [int(v) for v in valleys] + [len(prof)]
    return [(a, b, 1.6 * pitch) for a, b in zip(cuts, cuts[1:]) if b - a >= C.min_h and prof[a:b].max() > C.on_share * prof.max()]


def boxes_of(ink):
    """[(x0, y0, x1, y1)] of the text lines of `ink` (see `ink_of`)."""
    out = []
    bands = _bands_by_valley(ink) if ink.any() else None
    if bands:
        for a, b, cap in bands:
            _columns(ink, a, b, out, cap)
    if not out:
        for a, b in _bands_by_level(ink) if ink.any() else []:
            _columns(ink, a, b, out)
    return out


def page_boxes(path, page, size):
    """Boxes of a corpus page rendered at C.dpi, scaled to an image whose width is `size`."""
    pix = pymupdf.open(UVTM / "Layouts" / path)[page].get_pixmap(dpi=C.dpi, colorspace=pymupdf.csRGB)
    rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, 3)
    k = size / pix.w
    return [[round(v * k) for v in b] for b in boxes_of(ink_of(rgb))]


def _one(job):
    name, path, page, jpg = job
    cv2.setNumThreads(1)
    im = cv2.imread(jpg)
    return name, page_boxes(path, page, im.shape[1]), im.shape[:2]


def label_dir(d, workers=10):
    """Labels the pages of a folder made by `hwpick` (meta.json + <name>.jpg)"""
    d = Path(d)
    meta = json.loads((d / "meta.json").read_text())
    (d / "labels").mkdir(exist_ok=True)
    (d / "boxed").mkdir(exist_ok=True)
    jobs = [(k, v["path"], v["page"], str(d / f"{k}.jpg")) for k, v in meta.items() if (d / f"{k}.jpg").exists()]
    with ProcessPoolExecutor(workers) as ex:
        res = {n: (b, hw) for n, b, hw in ex.map(_one, jobs, chunksize=2)}
    for n, (boxes, (H, W)) in res.items():
        (d / "labels" / f"{n}.txt").write_text("\n".join(f"0 {(a + c) / 2 / W:.6f} {(b + e) / 2 / H:.6f} {(c - a) / W:.6f} {(e - b) / H:.6f}" for a, b, c, e in boxes))
        im = cv2.imread(str(d / f"{n}.jpg"))
        for a, b, c, e in boxes:
            cv2.rectangle(im, (a, b), (c, e), (0, 160, 0), 2)
        cv2.imwrite(str(d / "boxed" / f"{n}.jpg"), im)
    (d / "boxes.json").write_text(json.dumps({n: b for n, (b, _) in res.items()}))
    return {n: len(b) for n, (b, _) in res.items()}
