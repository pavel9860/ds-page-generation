"""A hand-drawn table: wobbly grid lines and the text of the cells."""

import numpy as np
from hwsynth.scripts import text_script
from hwsynth.writer import Stroke

from .util import CHAR_W, MAX_XH, MIN_XH, clean, trim


def wobbly(rng, x0, y0, x1, y1, step=3.5, noise=0.07):
    """A ruled line drawn by hand: nearly straight, small wobble, a slight overshoot."""
    d = np.hypot(x1 - x0, y1 - y0)
    n = max(int(d / step), 2)
    t = np.linspace(0, 1, n + 1)
    over = rng.uniform(-0.5, 1.5)
    ux, uy = (x1 - x0) / max(d, 1e-6), (y1 - y0) / max(d, 1e-6)
    x = x0 + (x1 - x0) * t + ux * over * t + rng.normal(0, noise, n + 1)
    y = (
        y0
        + (y1 - y0) * t
        + uy * over * t
        + rng.normal(0, noise, n + 1)
        + 0.15 * np.sin(t * rng.uniform(1, 4)) * rng.normal(0, 0.3)
    )
    s = Stroke(np.stack([x, y], 1))
    s.word = -1
    return s


def _try(rows, w, xh, colw, xs, room, rng, base_word, y0, pad=1.4, padv=1.1):
    w.p["size"] = xh
    w.pen.width = min(w.pen.width, xh / 5.2)
    w.pen.edge = min(w.pen.edge, xh / 4.2)
    strokes, words, ytop, tops, base = [], [], y0, [], base_word
    border = rng.random() < 0.8
    for row in rows:
        lays = [
            w.lines(trim(t, int(4 * max(colw[c] - 2 * pad, 4) / (xh * 1.6))), max(colw[c] - 2 * pad, 4), None)
            if t
            else None
            for c, t in enumerate(row)
        ]
        ink = [np.vstack([q.p for q in ly.strokes]) if ly and ly.strokes else None for ly in lays]
        top = min([a[:, 1].min() for a in ink if a is not None] or [0.0])
        rh = (
            max([a[:, 1].max() - top for a in ink if a is not None] or [2.3 * xh]) + 2 * padv
        )  # the row as tall as its ink
        if ytop + rh - y0 > room:
            break
        for c, ly in enumerate(lays):
            if not ly:
                continue
            for s in ly.strokes:
                s.p = s.p + (xs[c] + pad, ytop + padv - top)
                s.word += base
                strokes.append(s)
            words += ly.words
            base += len(ly.words)
        tops.append(ytop)
        ytop += rh
    if not tops:
        return [], [], 0.0, False
    ok = len(tops) == len(rows)
    tops.append(ytop)
    tilt = rng.normal(0, 0.003)
    for k, y in enumerate(tops):  # horizontal rules (the outer ones only when the table has a border)
        if border or 0 < k < len(tops) - 1:
            strokes.append(wobbly(rng, xs[0] - 0.8, y, xs[-1] + 0.8, y + tilt * xs[-1]))
    for c in range(len(colw) + 1):  # vertical rules
        if border or 0 < c < len(colw):
            strokes.append(wobbly(rng, xs[c], tops[0] - 0.8, xs[c] + tilt * (tops[-1] - tops[0]), tops[-1] + 0.8))
    return strokes, words, ytop - y0, ok


def table_block(cells, w, xh, full, room, rng, base_word, y0):
    """Strokes, words and height [mm] of a table; the writing is as large as the columns and the room allow, rows that still do not fit
    are left out."""
    rows = [[clean(c) for c in r] for r in cells]
    rows = [r for r in rows if any(r)]
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    weight = np.array([min(max(max(len(r[c]) for r in rows), 4), 26) for c in range(ncol)], float)
    colw = full * weight / weight.sum()
    xs = np.concatenate([[0.0], np.cumsum(colw)])
    k = CHAR_W.get(text_script(" ".join(c for r in rows for c in r) or "a"), 2.2)
    word = [max((len(x) for r in rows for x in r[c].split()), default=1) for c in range(ncol)]
    cap = min(
        (colw[c] - 2.8) / (max(word[c], 2) * k) for c in range(ncol)
    )  # the longest word of a column fits its width
    x, best = min(xh, max(cap, MIN_XH)), None
    for _ in range(7):
        res = _try(rows, w, x, colw, xs, room, rng, base_word, y0)
        if best is None or res[3]:  # the larger the better while all rows fit
            best = res
        if not res[3] or x * 1.15 > min(cap, 3.2, MAX_XH):
            break
        x *= 1.15
    return best[:3]
