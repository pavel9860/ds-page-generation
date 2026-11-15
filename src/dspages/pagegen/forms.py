"""Unfilled form pages: fill spots with their labels, and filling them by one writer at one size."""
import random
import re
from collections import Counter

import cv2
import numpy as np
import pymupdf
from dspages.hand import texts as T

from hwsynth.scripts import text_script
from . import paper
from .config import GenCfg
from .hand import FontHand
from .page import is_rtl

TINT_DARK = 24
MIN_WIDGETS = 3
MIN_SPOT_PX = 3.0
RULE_GRAY = 250
_DOTS = re.compile(r"^[.…_·‥\-–—:]+$")


class Form:
    """A form page rendered at the long side"""

    def __init__(self, img, prints, spots, words, leaders=(), sample=""):
        self.img, self.prints, self.spots, self.words, self.leaders = img, prints, spots, words, list(leaders)
        self.sample = sample


def _overlap(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0) * max(h, 0) / max(min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])), 1)


def _grid_spots(img, prints, lh):
    """Empty cells and underlines found in the raster"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ink = (gray < 215).astype(np.uint8)
    text = np.zeros_like(ink)
    for x0, y0, x1, y1 in prints:
        text[max(int(y0) - 1, 0):int(y1) + 2, max(int(x0) - 1, 0):int(x1) + 2] = 1
    ink[text > 0] = 0
    faint = (gray < RULE_GRAY).astype(np.uint8)
    faint[text > 0] = 0
    faint &= 1 - cv2.dilate(cv2.morphologyEx(faint, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)), np.ones((3, 3), np.uint8))
    faint |= ink
    hor = cv2.morphologyEx(faint, cv2.MORPH_OPEN, np.ones((1, max(25, int(2.5 * lh))), np.uint8))
    ver = cv2.morphologyEx(faint, cv2.MORPH_OPEN, np.ones((max(10, int(1.2 * lh)), 1), np.uint8))
    rules = cv2.dilate(hor | ver, np.ones((3, 3), np.uint8))
    free = ((rules == 0) & (text == 0) & (gray > 190)).astype(np.uint8)
    H, W = gray.shape
    n, _, st, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    boxes, cells = [], []
    for x, y, w, h, area in st[1:]:
        if (0.7 * lh <= w <= 3 * lh and 0.7 * lh <= h <= 3 * lh and area >= 0.9 * w * h and x > 1 and y > 1
                and x + w < W - 1 and y + h < H - 1):
            cells.append((x, y, x + w, y + h))
        if (3 * lh <= w <= 0.95 * W and 0.8 * lh <= h <= 4 * lh and area >= 0.8 * w * h and x > 1 and y > 1
                and x + w < W - 1 and y + h < H - 1):
            below = text[y + h:y + h + max(2, int(0.5 * lh)), x:x + w]
            if below.mean() > 0.3 and rules[y + h:y + h + 3, x:x + w].mean() < 0.5:
                continue
            boxes.append((x, y, x + w, y + h))
    m, _, st, _ = cv2.connectedComponentsWithStats(hor, connectivity=8)
    for x, y, w, h, _ in st[1:]:
        crossing = np.diff(np.r_[0, ver[y + 2:y + int(lh), x:x + w].any(axis=0).astype(int)]) > 0
        if w < 3 * lh or h > 5 or y <= 1.4 * lh or crossing.sum() >= 2:
            continue
        free = ~text[int(y - 1.2 * lh):int(y), x:x + w].any(axis=0)
        runs, i = [], 0
        while i < w:
            if free[i]:
                j = i
                while j < w and free[j]:
                    j += 1
                runs.append((i, j))
                i = j
            else:
                i += 1
        if runs:
            a, b = max(runs, key=lambda r: r[1] - r[0])
            if b - a >= 3 * lh:
                boxes.append((x + a, y - 1.4 * lh, x + b, y - 1))
    return boxes, cells


def _tint_spots(img, prints, lh):
    """Fields drawn as a tinted block (light grey, yellow, blue) without a rule round them"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    paper = float(np.percentile(gray, 90))
    tint = ((gray < paper - 6) & (gray > paper - 40)).astype(np.uint8)
    for x0, y0, x1, y1 in prints:
        tint[max(int(y0) - 1, 0):int(y1) + 2, max(int(x0) - 1, 0):int(x1) + 2] = 0
    tint = cv2.morphologyEx(tint, cv2.MORPH_OPEN, np.ones((max(3, int(0.5 * lh)), max(3, int(0.5 * lh))), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(tint, connectivity=4)
    H, W = gray.shape
    out = []
    for x, y, w, h, area in st[1:]:
        if 3 * lh <= w <= 0.8 * W and 0.7 * lh <= h <= 4 * lh and area >= 0.85 * w * h and x > 1 and y > 1 and x + w < W - 1 and y + h < H - 1:
            out.append((x, y, x + w, y + h))
    return [b for b in out if not _holds_print(b, prints)]


def _holds_print(b, prints, share=0.04):
    """True when printed text lies on the box (a tinted label or header cell, not an empty field)."""
    area = max((b[2] - b[0]) * (b[3] - b[1]), 1)
    return sum(max(min(b[2], q[2]) - max(b[0], q[0]), 0) * max(min(b[3], q[3]) - max(b[1], q[1]), 0) for q in prints) / area > share


def _inset_spots(img, prints, lh):
    """Fields drawn as white boxes set into a tinted or dark area (no rule round them)"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    paper = float(np.percentile(gray, 90))
    white = (gray >= paper - 4).astype(np.uint8)
    for x0, y0, x1, y1 in prints:
        white[max(int(y0) - 1, 0):int(y1) + 2, max(int(x0) - 1, 0):int(x1) + 2] = 1
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(white, connectivity=4)
    H, W = gray.shape
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    out = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if i not in edge and 3 * lh <= w <= 0.8 * W and 0.8 * lh <= h <= 10 * lh and area >= 0.9 * w * h:
            out.append((x, y, x + w, y + h))
    return out


def _underlined(gray, b, share=0.5):
    """True when a rule (even a pale one) runs under the area b over at least `share` of its width"""
    x0, x1, y1 = int(b[0]), int(b[2]), int(b[3])
    band = gray[max(y1 - 3, 0):min(y1 + 5, gray.shape[0]), x0:x1]
    return band.size > 0 and float((band.min(axis=0) < 235).mean()) >= share


def _implausible(gray, prints, b):
    """Boxes that read as part of the print, not as a field (measured against the fillable forms of the corpus)"""
    x0, y0, x1, y1 = (int(v) for v in b)
    inner = gray[y0 + 1:y1 - 1, x0 + 1:x1 - 1]
    if inner.size == 0:
        return True
    if _holds_print(b, prints, 0.05) and (x1 - x0) <= 0.14 * gray.shape[1]:
        return True
    return float(inner.mean()) - float(np.percentile(gray, 90)) < -TINT_DARK


def _combs(cells, lh):
    """Rows of at least 4 equal adjacent empty cells (one letter per cell); returns [[cell, ...]] left to right."""
    rows = []
    for c in sorted(cells, key=lambda c: (round(c[1] / (0.3 * lh)), c[0])):
        for r in rows:
            p = r[-1]
            same = abs(c[1] - p[1]) < 0.3 * lh and abs((c[3] - c[1]) - (p[3] - p[1])) < 0.2 * (p[3] - p[1]) + 1
            if same and abs((c[2] - c[0]) - (p[2] - p[0])) < 0.2 * (p[2] - p[0]) + 1 and -2 <= c[0] - p[2] < 0.5 * (p[2] - p[0]):
                r.append(c)
                break
        else:
            rows.append([c])
    return [r for r in rows if len(r) >= 4]


def _occupied(ink, b, share=0.025):
    """True when the inside of box b already holds printed text (a filled cell, a label), not just its frame."""
    x0, y0, x1, y1 = int(b[0]) + 3, int(b[1]) + 3, int(b[2]) - 3, int(b[3]) - 3
    return x1 > x0 and y1 > y0 and ink[y0:y1, x0:x1].mean() > share


def render_form(path, page_no, cfg: GenCfg, use_widgets=False, debug=None, lang=None) -> Form:
    """Words and lines are read by the one page reader (dspages.hand.texts, legacy fonts of lang decoded). Spots are leader runs, drawn underlines and empty cells (from the page content and from the raster) and the area right of a label that ends with a colon; the label of a spot is the words left of it on its row, else the words right above."""
    with pymupdf.open(path) as doc:
        pg = doc[page_no]
        tp = pg.get_textpage(flags=pymupdf.TEXTFLAGS_RAWDICT)     # read once for the three outputs (words and text skip the image blocks it keeps)
        prints, fields, labels = paper.page_segments(pg, cfg, tp)
        read = T.page_lines(pg, lang)
        words = T.page_words(pg, lang, read)
        lines = [t for l in read if 25 <= len(t := l["text"]) <= 70 and _DOTS.sub("", t).strip()]
        sample = max(lines, key=len) if lines else " ".join(w[4] for w in words[:10])
        k = cfg.long_side / max(pg.rect.width, pg.rect.height)
        widgets = [(tuple(w.rect), bool(w.field_value)) for w in (pg.widgets() or []) if w.field_type_string in ("Text", "ComboBox")]
        px = pg.get_pixmap(matrix=pymupdf.Matrix(k, k), alpha=False)
    img = np.frombuffer(px.samples, np.uint8).reshape(px.h, px.w, px.n)[:, :, :3].copy()
    W = img.shape[1]
    sc = lambda b: tuple(v * k for v in b)
    prints, fields, labels = [sc(b) for b in prints], [sc(b) for b in fields], [sc(b) for b in labels]
    words = [sc(w[:4]) + (w[4],) for w in words]
    lh = float(np.median([b[3] - b[1] for b in prints])) if prints else 10.0

    def clean(ws):
        return " ".join(w[4] for w in ws if not _DOTS.match(w[4])).strip(" :")

    def label_of(b):
        mid, h = (b[1] + b[3]) / 2, max(b[3] - b[1], 8)
        left = sorted(w for w in words if w[2] <= b[0] + 2 and abs((w[1] + w[3]) / 2 - mid) < h and b[0] - w[2] < 200 * k)
        above = sorted(w for w in words if w[3] <= b[1] + 2 and b[1] - w[3] < 2.5 * h and w[2] > b[0] and w[0] < b[2])
        if not left and not above:
            above = sorted(w for w in words if w[3] <= b[1] + 2 and b[1] - w[3] < 14 * h and w[2] > b[0] and w[0] < b[2])
            above = [w for w in above if w[3] >= max(x[3] for x in above) - 2.5 * h][-6:] if above else []
        return clean(left[-6:]) or clean(above[-6:])

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    def walled(b, up=0.0):
        x0, y0, x1, y1 = int(b[0]), max(int(b[1] - up), 0), int(b[2]), int(b[3])
        return any(((gray[y0:y1, max(x - a, 0):x + c] < 235).mean(axis=0) > 0.7).any() for x, a, c in ((x0, 14, 3), (x1, 3, 14)))

    boxes, colon = list(fields), []
    for b in labels:
        right = [q[0] for q in prints if q[0] > b[2] and min(q[3], b[3]) - max(q[1], b[1]) > 0.5 * (b[3] - b[1])]
        r = min(right + [W - 20 * k]) - 4 * k
        row = gray[int(b[1] + (b[3] - b[1]) * 0.3):int(b[3] - (b[3] - b[1]) * 0.3) + 1, int(b[2]) + 3:int(r)]
        if row.size:
            hit = np.flatnonzero(np.median(row, axis=0) < 175)
            r = min(r, b[2] + 3 + hit[0] - 3) if hit.size else r
        if r - b[2] > 4 * (b[3] - b[1]) and _underlined(gray, (b[2] + 0.4 * (b[3] - b[1]), b[1], r, b[3])):
            colon.append((b[2] + 0.4 * (b[3] - b[1]), b[1], r, b[3]))
    if use_widgets and len(widgets) >= MIN_WIDGETS:
        own = []
        for r, filled in sorted(widgets, key=lambda w: (w[0][1], w[0][0])):
            b = sc(r)
            if not filled and b[2] - b[0] >= lh and b[3] - b[1] >= 0.6 * lh and not any(_overlap(b, o) > 0.8 for o in own):
                own.append(b)
        spots = [{"label": label_of(b), "box": list(b), "src": "widget"} for b in own]
        return Form(img, prints, spots, len(words), fields, sample)
    grid, small = _grid_spots(img, prints, lh)
    combs = _combs(small, lh)
    tint = _tint_spots(img, prints, lh) + _inset_spots(img, prints, lh)
    cells = [g for g in grid if walled(g, 0.8 * lh)]
    boxes += tint + grid + [c for c in colon if not any(0 <= g[1] - c[3] < 2.5 * lh and min(g[2], c[2]) - max(g[0], c[0]) > 0.3 * (c[2] - c[0])
                                                 for g in cells)]

    def header(b, others):
        h = b[3] - b[1]
        beside = any(min(q[3], b[3]) - max(q[1], b[1]) > 0.5 * h and (-lh <= b[0] - q[2] < 1.5 * lh or -lh <= q[0] - b[2] < 1.5 * lh)
                     for q in prints)
        below = any(0 <= o[1] - b[3] < 2.5 * lh and min(o[2], b[2]) - max(o[0], b[0]) > 0.5 * (b[2] - b[0]) for o in others)
        return h <= 1.8 * lh and beside and below and walled(b)

    sources = [("leader", list(fields)), ("tint", tint), ("colon", colon), ("cell", cells)]
    def keep(bs, ok, why):
        for b in bs:
            if not ok(b) and debug is not None:
                debug.append((tuple(b), why))
        return [b for b in bs if ok(b)]

    boxes = keep(boxes, lambda b: any(_overlap(b, t) > 0.8 for t in tint) or not header(b, boxes), "header")
    ink = (gray < 170).astype(np.uint8)
    ink[cv2.dilate(cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(25, int(2.5 * lh))), np.uint8)) |
                   cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((max(10, int(1.2 * lh)), 1), np.uint8)), np.ones((5, 5), np.uint8)) > 0] = 0
    for x0, y0, x1, y1 in fields:
        ink[max(int(y0) - 2, 0):int(y1) + 3, int(x0):int(x1) + 1] = 0
    boxes = keep(boxes, lambda b: not _occupied(ink, b), "holds text")
    spots, seen = [], []
    for b in sorted(boxes, key=lambda b: (b[1], b[0])):
        if any(_overlap(b, o) > 0.4 or abs(b[0] - o[0]) < 8 and abs(b[1] - o[1]) < 8 or (abs((b[0] + b[2]) / 2 - (o[0] + o[2]) / 2) < 8
               and abs(b[3] - o[3]) < 8) for o in seen):
            if debug is not None:
                debug.append((tuple(b), "duplicate"))
            continue
        seen.append(b)
        if debug is not None and b[2] - b[0] < MIN_SPOT_PX * lh:
            debug.append((tuple(b), "narrow"))
        elif debug is not None and not label_of(b) and not any(_overlap(b, t) > 0.8 for t in tint):
            debug.append((tuple(b), "no label"))
        if _implausible(gray, prints, b):
            if debug is not None:
                debug.append((tuple(b), "implausible"))
            continue
        if b[2] - b[0] >= MIN_SPOT_PX * lh and ((label := label_of(b)) or any(_overlap(b, t) > 0.8 for t in tint)):
            spots.append({"label": label, "box": list(b), "src": next((n for n, bs in sources if any(_overlap(b, o) > 0.6 for o in bs)), "rule")})
    for r in combs:
        b = (min(c[0] for c in r), min(c[1] for c in r), max(c[2] for c in r), max(c[3] for c in r))
        spots.append({"label": label_of(b) or "", "box": list(b), "cells": [list(c) for c in r]})
    spots.sort(key=lambda s: (s["box"][1], s["box"][0]))
    return Form(img, prints, spots, len(words), fields, sample)


def best_fit(hand: FontHand, options, width):
    """The longest of the option texts whose strip is no wider than `width`"""
    ok = lambda s: s is not None and s.alpha.shape[1] <= width
    fits = [s for t in options if ok(s := hand.strip(t))]
    if fits:
        return max(fits, key=lambda s: s.alpha.shape[1])
    tokens = min(options, key=len).split()
    while len(tokens) > 1:
        tokens = tokens[:-1]
        if ok(s := hand.strip(" ".join(tokens))):
            return s
    word = tokens[0] if tokens else ""
    while word:
        if ok(s := hand.strip(word)):
            return s
        word = word[:-1]
    return None


def _baseline(dark, box, lh):
    """y of the rule or dotted line a spot is written on"""
    x0, y0, x1, y1 = (int(v) for v in box)
    ya, yb = int((y0 + y1) / 2), min(int(y1 + 0.35 * lh), dark.shape[0] - 1)
    if x1 <= x0 or yb <= ya:
        return y1
    share = dark[ya:yb + 1, x0:x1].mean(axis=1)
    return ya + int(share.argmax()) if share.max() >= 0.2 else y1


def _comb(hand: FontHand, text, cells, img, rng):
    """One letter per cell, centred, at the hand's size; returns True when a letter was written."""
    placed = False
    for ch, c in zip(re.sub(r"\s+", "", text), cells):
        st = hand.strip(ch)
        if st is None:
            continue
        h, w = st.alpha.shape
        x, y = int((c[0] + c[2] - w) / 2 + rng.gauss(0, 0.04) * w), int((c[1] + c[3] - h) / 2 + rng.gauss(0, 0.04) * h)
        if x < 0 or y < 0 or x + w >= img.shape[1] or y + h >= img.shape[0]:
            continue
        paper.blend(img, st.alpha, hand.rgb, x, y, hand.density)
        placed = True
    return placed


def capacity(form: Form, height, sample, cfg: GenCfg, rng: random.Random, writers=3):
    """Characters of `sample`'s language and script that fit each spot when the font is `height` px high"""
    char_px = float(np.mean([FontHand(cfg, rng, form.img.shape[0], 10**5, height).char_px(sample) for _ in range(writers)]))
    return [len(s["cells"]) if "cells" in s else max(1, int((s["box"][2] - s["box"][0]) / char_px)) for s in form.spots]


MIN_LABEL_CHARS = 2


def fill(form: Form, values, height, cfg: GenCfg, rng: random.Random):
    """Writes values[i] (a text or a list of alternatives, None to skip) into spot i with one writer at one size"""
    img = form.img.copy()
    lh = float(np.median([b[3] - b[1] for b in form.prints])) if form.prints else 10.0
    hand = FontHand(cfg, rng, img.shape[0], img.shape[1], height)
    texts = [t for v in values if v for t in ([v] if isinstance(v, str) else v)]
    hand.script = Counter(text_script(t) for t in texts[:40]).most_common(1)[0][0] if texts else "Latin"
    hand.rtl = is_rtl(texts)
    dark = (cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) < 170).astype(np.uint8)
    boxes, n, words = [], 0, 0
    for spot, options in zip(form.spots, values):
        if not options:
            continue
        options = [options] if isinstance(options, str) else options
        if "cells" in spot:
            text = max(options, key=len)[:len(spot["cells"])]
            letters = len(text.replace(" ", ""))
            if _comb(hand, text, spot["cells"], img, rng):
                n += 1
                words += 1
                if letters >= MIN_LABEL_CHARS:
                    used = spot["cells"][:letters]
                    boxes.append((min(c[0] for c in used), min(c[1] for c in used), max(c[2] for c in used), max(c[3] for c in used)))
            continue
        fx0, fy0, fx1, fy1 = spot["box"]
        strip = best_fit(hand, options, fx1 - fx0)
        if strip is None:
            continue
        h, w = strip.alpha.shape
        pad = min(0.25 * height, max(fx1 - fx0 - w, 0))
        x = int(fx1 - w - pad if hand.rtl else fx0 + pad)
        y = int(_baseline(dark, spot["box"], lh) - 0.8 * h + rng.gauss(0, 0.04) * h)
        x, y = min(max(x, 0), img.shape[1] - w - 1), min(max(y, 0), img.shape[0] - h - 1)
        paper.blend(img, strip.alpha, hand.rgb, x, y, hand.density)
        if sum(c.isalnum() for c in options[0]) >= MIN_LABEL_CHARS:
            boxes.append((x, y, x + w, y + h))
        n += 1
        words += strip.words
    return img, boxes, n, words
