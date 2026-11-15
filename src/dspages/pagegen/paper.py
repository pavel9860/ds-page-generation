"""Pixels of a page: print boxes and form fields of a PDF page, free paper, ink blended on it, the scan look."""
from dataclasses import replace

import cv2
import numpy as np

from hwsynth import scan
from hwsynth.config import ScanCfg
from .config import GenCfg


def page_segments(pg, cfg: GenCfg, textpage=None):
    """Print boxes split at leader runs and wide gaps; leader runs are fields to fill; prints ending in '"""
    prints, fields, labels = [], [], []
    leaders = set(cfg.leader_chars)

    def bbox(cs):
        return (min(c["bbox"][0] for c in cs), min(c["bbox"][1] for c in cs),
                max(c["bbox"][2] for c in cs), max(c["bbox"][3] for c in cs))

    for b in pg.get_text("rawdict", textpage=textpage)["blocks"]:
        for ln in b.get("lines", []):
            chars = [c for sp in ln["spans"] if sp["size"] > cfg.text_min_font for c in sp["chars"]]
            if not chars:
                continue
            h = max(c["bbox"][3] - c["bbox"][1] for c in chars)
            segs, cur, prev = [], None, None
            for c in chars:
                kind = "f" if c["c"] in leaders else ("s" if not c["c"].strip() else "p")
                if kind == "s":
                    if cur:
                        cur[1].append(c)
                    continue
                gap = prev is not None and c["bbox"][0] - prev["bbox"][2] > cfg.gap_factor * h
                if cur is None or cur[0] != kind or gap:
                    cur = [kind, []]
                    segs.append(cur)
                cur[1].append(c)
                prev = c
            for kind, cs in segs:
                solid = [c for c in cs if c["c"].strip()]
                if kind == "f" and len(cs) >= cfg.min_leader_run:
                    fields.append(bbox(cs))
                elif solid:
                    prints.append(bbox(solid))
                    if "".join(c["c"] for c in solid)[-1] in ":：" and kind == "p":
                        labels.append(prints[-1])
    return prints, fields, labels


def free_mask(img, prints, cfg: GenCfg):
    """occ: everything that is not plain paper, thin form rules excepted; rules: long horizontal rules to write on."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    ink = ((hsv[..., 2] < cfg.paper_value) | (hsv[..., 1] > cfg.paper_sat)).astype(np.uint8)
    kernel = lambda w, h: cv2.getStructuringElement(cv2.MORPH_RECT, (w, h))
    solid = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((cfg.solid, cfg.solid), np.uint8))
    hor = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel(cfg.rule_len, 1)) & ~solid
    ver = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel(1, cfg.vrule_len)) & ~solid
    occ = ink & ~(hor | ver)
    for x0, y0, x1, y1 in prints:
        occ[int(y0):int(y1) + 1, int(x0):int(x1) + 1] = 1
    _, _, st, _ = cv2.connectedComponentsWithStats(hor)
    rules = [(x, y, x + w, y + h) for x, y, w, h, _ in st[1:] if w >= cfg.rule_len and h <= cfg.rule_max_h]
    k = np.ones((cfg.occ_dilate, cfg.occ_dilate), np.uint8)
    return cv2.dilate(occ, k), rules


def blend(img, alpha, rgb, x, y, density):
    h, w = alpha.shape
    img[y:y + h, x:x + w] = (img[y:y + h, x:x + w] * (1 - density * alpha[..., None] * (1 - rgb[None, None]))).astype(np.uint8)


def degrade(img, rng, cfg: GenCfg, p_scan):
    """Whole-page look, same chain for print and handwriting"""
    nrng = np.random.default_rng(rng.getrandbits(32))
    if rng.random() < p_scan:
        f = img.astype(np.float32) / 255
        sc = replace(ScanCfg(), p_bitonal=cfg.scan_bitonal, blur=cfg.scan_blur, p_bleed=cfg.scan_bleed)
        out = scan.degrade(f, 1 - (f[..., 0] + f[..., 1] + f[..., 2]) / 3, nrng, cfg.scan_dpi, scan.age(nrng, sc), sc)
        return (np.clip(out, 0, 1) * 255).astype(np.uint8)
    if rng.random() < cfg.p_blur:
        return cv2.GaussianBlur(img, (0, 0), rng.uniform(*cfg.blur_sigma))
    return img
