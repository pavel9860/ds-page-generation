"""Handwritten page layouts."""
import re

import cv2
import numpy as np

from .canvas import Canvas, Hand

MARGIN = (0.09, 0.07)
LETTERS = {"Latin": "abcdefghijklmnop", "Cyrillic": "абвгдежзиклмнопр", "Greek": "αβγδεζηθικλμνξο",
           "Hebrew": "אבגדהוזחטיכלמנ", "Arabic": "أبتثجحخدذرزسشصض"}
STYLES = {"Latin": ("1.", "1)", "a)", "-", "•"), "Cyrillic": ("1.", "1)", "а)", "-"), "Greek": ("1.", "α)", "-", "•"),
          "Hebrew": ("1.", "א.", "-"), "Arabic": ("1-", "-", "1)", "أ)"), "Han": ("1.", "・", "-", "一、"),
          "Hangul": ("1.", "가.", "-", "•"), "Hiragana": ("1.", "・", "-"), "Katakana": ("1.", "・", "-")}


def bullet(script, style, n):
    """The n-th mark of a list in the styles used in the script: numbers, its own letters, a dash or a dot."""
    if style in ("-", "•", "・"):
        return style
    if style[0].isdigit():
        return f"{n}{style[1:]}"
    if style == "一、":
        return "一二三四五六七八九十"[(n - 1) % 10] + "、"
    letters = LETTERS.get(script, LETTERS["Latin"])
    return letters[(n - 1) % len(letters)] + style[1:]


def _sentences(pool):
    out = []
    for p in pool.get("para", []):
        out += [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+", p) if 12 <= len(s.strip()) <= 140]
    return out


def paras(pool):
    return pool.get("para") or _sentences(pool)


def items(pool):
    """List entries: sentences of the paragraphs (real list items of the documents are mostly a few words long)"""
    return _sentences(pool) or pool.get("item", [])


def heads(pool):
    return pool.get("heading") or [" ".join(s.split()[:4]) for s in _sentences(pool)]


def shorts(pool, n=1):
    """Short phrases (2-5 words, or 4-12 characters without spaces) cut from sentences."""
    out = []
    for s in _sentences(pool):
        w = s.split()
        out.append(" ".join(w[:5]) if len(w) > 2 else s[:12])
    return out or heads(pool)


def _baseline(hand: Hand, w, row_px):
    """Vertical offsets (px) along a row of width w: a slow aperiodic trend."""
    k = hand.rng.randint(3, 5)
    xs = np.linspace(0, w - 1, k)
    ys = [hand.rng.gauss(0, 0.07 * row_px) for _ in range(k)]
    return np.interp(np.arange(w), xs, ys) + np.arange(w) * hand.tilt + hand.rng.gauss(0, 0.004) * np.arange(w)


def _curve(hand, strip):
    """The strip with its baseline bent along `_baseline`."""
    h, w = strip.alpha.shape
    off = _baseline(hand, w, h)
    pad = int(np.ceil(np.abs(off).max())) + 1
    mx = np.tile(np.arange(w, dtype=np.float32), (h + 2 * pad, 1))
    my = (np.arange(h + 2 * pad, dtype=np.float32)[:, None] - pad - off[None, :]).astype(np.float32)
    return strip.with_alpha(cv2.remap(strip.alpha, mx, my, cv2.INTER_LINEAR, borderValue=0))


def _drift(hand):
    """Start of the next row, relative to the margin (px)"""
    hand.drift = 0.6 * getattr(hand, "drift", 0.0) + hand.rng.gauss(0, 0.5 * hand.row_mm * hand.px_mm)
    return float(np.clip(hand.drift, -1.4 * hand.row_mm * hand.px_mm, 1.4 * hand.row_mm * hand.px_mm))


def _pitch(hand):
    return 1.55 * hand.row_mm * hand.px_mm


def _bounds(c: Canvas):
    return c.W * MARGIN[0], c.W * (1 - MARGIN[0]), c.H * MARGIN[1]


def row(hand, text, x0, x1, y, align="start", justify=False, indent=0.0):
    """One written row of `text` (it fits the width: callers wrap)."""
    c = hand.canvas
    s = hand.strip(text)
    if s is None:
        return None
    width = x1 - x0 - indent
    w = s.alpha.shape[1]
    if w > width:
        s = s.with_alpha(cv2.resize(s.alpha, (int(width), s.alpha.shape[0]), interpolation=cv2.INTER_AREA))
        w = int(width)
    elif justify and 0.88 * width <= w:
        s = s.with_alpha(cv2.resize(s.alpha, (int(width), s.alpha.shape[0]), interpolation=cv2.INTER_LINEAR))
        w = int(width)
    s = _curve(hand, s)
    off = _drift(hand)
    if align == "center":
        x = x0 + indent + (width - w) / 2
    elif (align == "start") != hand.rtl:
        x = x0 + indent + max(off, 0.0)
    else:
        x = x1 - w - max(-off, 0.0)
    c.put(s, x, y, hand.rgb)
    return s


def paragraph(hand, text, x0, x1, y, indent=0.0, align=None, step=1.0):
    """The text wrapped into rows (left or RTL-start aligned, or justified; the last row stays ragged)."""
    align = align or hand.align
    c = hand.canvas
    pieces = hand.wrap(text, x1 - x0 - indent)
    pitch = _pitch(hand) * step
    for i, piece in enumerate(pieces):
        if y + pitch > c.y_max:
            break
        s = row(hand, piece, x0, x1, y, "start", justify=(align == "justify" and i < len(pieces) - 1),
                indent=indent if i == 0 else 0.0)
        y += max(pitch * hand.rng.uniform(0.95, 1.07), 0.9 * s.alpha.shape[0] if s is not None else 0.0)
    return y


def underline(hand, strip_y, x_from, x_to):
    c = hand.canvas
    c.stroke(x_from, strip_y, x_to, strip_y + hand.rng.uniform(-2, 2), hand.rgb, hand.pen, hand.density)


def heading(hand, text, x0, x1, y, center=False, underlined=True):
    """A heading row (one row, shortened to fit), underlined by hand."""
    c = hand.canvas
    text = hand.wrap(text, 0.8 * (x1 - x0))[0]
    s = hand.strip(text)
    if s is None:
        return y
    s = _curve(hand, s)
    w = s.alpha.shape[1]
    x = x0 + (x1 - x0 - w) / 2 if center else (x1 - w if hand.rtl else x0)
    c.put(s, x, y, hand.rgb)
    if underlined:
        underline(hand, y + s.alpha.shape[0] + 2, x, x + w)
    return y + 1.7 * _pitch(hand)


def cell(hand, text, x, y, width, lines=1):
    """Up to `lines` rows of text in a cell `width` px wide, each squeezed to the width when it is wider."""
    c = hand.canvas
    for i, piece in enumerate(hand.wrap(text, width)[:lines]):
        s = hand.strip(piece)
        if s is None:
            continue
        w = s.alpha.shape[1]
        if w > width:
            s = s.with_alpha(cv2.resize(s.alpha, (int(width), s.alpha.shape[0]), interpolation=cv2.INTER_AREA))
        c.put(s, x + width - s.alpha.shape[1] if hand.rtl else x, y + i * 1.15 * s.alpha.shape[0], hand.rgb)


def _more(hand, target):
    return hand.canvas.words < target


def plain(hand, pool, target, rng):
    x0, x1, y = _bounds(hand.canvas)
    for p in paras(pool):
        if not _more(hand, target) or y > hand.canvas.y_max:
            break
        y = paragraph(hand, p, x0, x1, y, indent=2 * hand.row_mm * hand.px_mm if rng.random() < 0.4 else 0.0)
        y += 0.3 * _pitch(hand)


def blocks(hand, pool, target, rng):
    x0, x1, y = _bounds(hand.canvas)
    h = heads(pool)
    for i, p in enumerate(paras(pool)):
        if not _more(hand, target) or y > hand.canvas.y_max - 3 * _pitch(hand):
            break
        y = heading(hand, h[i % len(h)], x0, x1, y)
        y = paragraph(hand, p, x0, x1, y) + 0.7 * _pitch(hand)


def notes(hand, pool, target, rng):
    c = hand.canvas
    x0, x1, y = _bounds(c)
    style = rng.choice(STYLES.get(hand.script, ("1.", "1)", "-")))
    y = heading(hand, heads(pool)[0], x0, x1, y)
    off = 2.4 * hand.row_mm * hand.px_mm
    for n, t in enumerate(items(pool), 1):
        if not _more(hand, target) or y > c.y_max - _pitch(hand):
            break
        m = hand.strip(bullet(hand.script, style, n))
        if m:
            c.put(m, x1 - m.alpha.shape[1] if hand.rtl else x0, y, hand.rgb)
        y = paragraph(hand, t, x0 + (0 if hand.rtl else off), x1 - (off if hand.rtl else 0), y, align="start")


def letter(hand, pool, target, rng):
    c = hand.canvas
    x0, x1, y = _bounds(c)
    date = hand.strip(f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.{rng.randint(2005, 2026)}")
    if date:
        c.put(date, x0 if hand.rtl else x1 - date.alpha.shape[1], y, hand.rgb)
    y += 2.2 * _pitch(hand)
    sh = shorts(pool)
    y = paragraph(hand, sh[rng.randrange(len(sh))], x0, x1, y, align="start") + 0.5 * _pitch(hand)
    for p in paras(pool):
        if not _more(hand, target - 6) or y > c.y_max - 3 * _pitch(hand):
            break
        y = paragraph(hand, p, x0, x1, y, indent=3 * hand.row_mm * hand.px_mm) + 0.2 * _pitch(hand)
    far = (x0, x0 + 0.45 * (x1 - x0)) if hand.rtl else (x1 - 0.45 * (x1 - x0), x1)
    paragraph(hand, shorts(pool)[0], far[0], far[1], min(y + _pitch(hand), c.y_max * 0.97), align="start")


def math(hand, pool, target, rng):
    c = hand.canvas
    x0, x1, y = _bounds(c)
    f = pool.get("formula") or []
    for i in range(200):
        if not _more(hand, target) or y > c.y_max - _pitch(hand):
            break
        if f:
            line = rng.choice(f)
        else:
            a, b, k = rng.choice("xyabn"), rng.choice("xyabn"), rng.randint(2, 9)
            line = rng.choice([f"{a} + {k} = {rng.randint(10, 99)}", f"{a}^2 + {b}^2 = {k}^2",
                               f"({a} + {b})({a} - {b}) = {a}^2 - {b}^2", f"{a} / {b} = {rng.randint(2, 9)} / {rng.randint(2, 9)}"])
        if i % 4 == 0 and shorts(pool):
            y = paragraph(hand, rng.choice(shorts(pool)), x0, x1, y, align="start")
        y = paragraph(hand, f"{i + 1})  {line}", x0 + 2 * hand.row_mm * hand.px_mm, x1, y, align="start", step=1.25)


def table(hand, pool, target, rng):
    c = hand.canvas
    x0, x1, y = _bounds(c)
    cols = rng.randint(4, 6)
    cw = (x1 - x0) / cols
    ch = 1.7 * hand.row_mm * hand.px_mm
    rows = int((c.y_max - y) / ch)
    words = [w for t in shorts(pool) + heads(pool) for w in t.split()] or ["-"]
    for r in range(rows + 1):
        c.stroke(x0, y + r * ch, x1, y + r * ch, hand.rgb, hand.pen, hand.density)
    for k in range(cols + 1):
        c.stroke(x0 + k * cw, y, x0 + k * cw, y + rows * ch, hand.rgb, hand.pen, hand.density)
    for r in range(rows):
        for k in range(cols):
            if c.words >= target * 1.4:
                return
            text = str(rng.randint(1, 999)) if k == cols - 1 and r else " ".join(rng.sample(words, min(len(words), rng.randint(1, 3))))
            cell(hand, text, x0 + k * cw + 4, y + r * ch + 0.15 * ch, cw - 8)


def diary(hand, pool, target, rng):
    x0, x1, y = _bounds(hand.canvas)
    for p in paras(pool):
        if not _more(hand, target) or y > hand.canvas.y_max - 3 * _pitch(hand):
            break
        d = f"{rng.randint(1, 28)}.{rng.randint(1, 12)}.{rng.randint(2010, 2026)}"
        y = heading(hand, d, x0, x1, y, underlined=rng.random() < 0.5)
        y = paragraph(hand, p, x0, x1, y) + 0.8 * _pitch(hand)


def recipe(hand, pool, target, rng):
    c = hand.canvas
    x0, x1, y = _bounds(c)
    y = heading(hand, heads(pool)[0], x0, x1, y, center=True)
    for t in items(pool)[:7]:
        q = f"{rng.choice([100, 200, 250, 50, 2, 3, 1])} {rng.choice(['g', 'ml', 'x', 'g'])}"
        s = hand.strip(q)
        if s:
            c.put(s, x1 - s.alpha.shape[1] if hand.rtl else x0, y, hand.rgb)
        y = paragraph(hand, " ".join(t.split()[:5]) if " " in t else t[:14], x0 + 4 * hand.row_mm * hand.px_mm, x1, y, align="start")
    y += _pitch(hand)
    for n, p in enumerate(paras(pool), 1):
        if not _more(hand, target) or y > c.y_max - _pitch(hand):
            break
        y = paragraph(hand, f"{n}. {p}", x0, x1, y)


def checklist(hand, pool, target, rng):
    c = hand.canvas
    x0, x1, y = _bounds(c)
    y = heading(hand, heads(pool)[0], x0, x1, y)
    box = 1.0 * hand.row_mm * hand.px_mm
    for t in items(pool):
        if not _more(hand, target) or y > c.y_max - _pitch(hand):
            break
        bx = x1 - box if hand.rtl else x0
        for a, b in (((bx, y), (bx + box, y)), ((bx + box, y), (bx + box, y + box)), ((bx + box, y + box), (bx, y + box)),
                     ((bx, y + box), (bx, y))):
            c.stroke(a[0], a[1], b[0], b[1], hand.rgb, hand.pen, hand.density)
        if rng.random() < 0.5:
            c.stroke(bx + 0.1 * box, y + 0.5 * box, bx + 0.4 * box, y + 0.9 * box, hand.rgb, hand.pen * 1.3, hand.density)
            c.stroke(bx + 0.4 * box, y + 0.9 * box, bx + 1.2 * box, y - 0.3 * box, hand.rgb, hand.pen * 1.3, hand.density)
        y = paragraph(hand, " ".join(t.split()[:9]), x0 + (0 if hand.rtl else 1.8 * box), x1 - (1.8 * box if hand.rtl else 0), y,
                      align="start", step=1.1)


def cornell(hand, pool, target, rng):
    c = hand.canvas
    x0, x1, y = _bounds(c)
    split = x0 + 0.3 * (x1 - x0)
    c.stroke(split, y, split, c.y_max * 0.85, hand.rgb, hand.pen, hand.density)
    c.stroke(x0, c.y_max * 0.85, x1, c.y_max * 0.85, hand.rgb, hand.pen, hand.density)
    yr = y
    hs = heads(pool)
    for i, p in enumerate(paras(pool)):
        if not _more(hand, target) or yr > c.y_max * 0.85 - 3 * _pitch(hand):
            break
        cue = hs[i % len(hs)]
        if hand.rtl:
            paragraph(hand, cue, split + 6, x1, yr, align="start")
            yr = paragraph(hand, p, x0, split - 6, yr, align="start") + 0.5 * _pitch(hand)
        else:
            paragraph(hand, cue, x0, split - 6, yr, align="start")
            yr = paragraph(hand, p, split + 6, x1, yr, align="start") + 0.5 * _pitch(hand)
    paragraph(hand, shorts(pool)[0], x0, x1, c.y_max * 0.87, align="start")


def twocol(hand, pool, target, rng):
    c = hand.canvas
    x0, x1, y = _bounds(c)
    mid, gap = (x0 + x1) / 2, 0.03 * c.W
    ys = [y, y]
    for i, p in enumerate(paras(pool)):
        k = ys.index(min(ys))
        if not _more(hand, target) or ys[k] > c.y_max - 3 * _pitch(hand):
            break
        xa, xb = (x0, mid - gap) if k == 0 else (mid + gap, x1)
        ys[k] = paragraph(hand, p, xa, xb, ys[k]) + 0.4 * _pitch(hand)


def minutes(hand, pool, target, rng):
    c = hand.canvas
    x0, x1, y = _bounds(c)
    y = heading(hand, heads(pool)[0], x0, x1, y)
    t = rng.randint(8, 11) * 60
    for it in items(pool):
        if not _more(hand, target) or y > c.y_max - _pitch(hand):
            break
        s = hand.strip(f"{t // 60:02d}:{t % 60:02d}")
        if s:
            c.put(s, x1 - s.alpha.shape[1] if hand.rtl else x0, y, hand.rgb)
        y = paragraph(hand, it, x0 + (0 if hand.rtl else 3.2 * hand.row_mm * hand.px_mm), x1 - (3.2 * hand.row_mm * hand.px_mm if hand.rtl else 0),
                      y, align="start")
        t += rng.choice([15, 30, 45, 60])


def poem(hand, pool, target, rng):
    x0, x1, y = _bounds(c := hand.canvas)
    y = heading(hand, heads(pool)[0], x0, x1, y, center=True)
    align = rng.choice(["start", "center"])
    for p in paras(pool):
        for line in hand.wrap(p, 0.62 * (x1 - x0)):
            if not _more(hand, target) or y > c.y_max - _pitch(hand):
                return
            row(hand, line, x0, x1, y, "center" if align == "center" else "start")
            y += _pitch(hand) * rng.uniform(0.95, 1.1)
        y += 0.8 * _pitch(hand)


def qa(hand, pool, target, rng):
    c = hand.canvas
    x0, x1, y = _bounds(c)
    ind = 2.5 * hand.row_mm * hand.px_mm
    for n, p in enumerate(paras(pool), 1):
        if not _more(hand, target) or y > c.y_max - 3 * _pitch(hand):
            break
        q = " ".join(p.split()[:9]) if " " in p else p[:20]
        y = paragraph(hand, f"{n}) {q}?", x0, x1, y, align="start")
        y = paragraph(hand, p, x0, x1, y, indent=ind) + 0.5 * _pitch(hand)


def schedule(hand, pool, target, rng):
    c = hand.canvas
    x0, x1, y = _bounds(c)
    cols = rng.randint(3, 4)
    cw, ch = (x1 - x0) / (cols + 1), 2.7 * hand.row_mm * hand.px_mm
    rows = int((c.y_max - y) / ch)
    for r in range(rows + 1):
        c.stroke(x0, y + r * ch, x1, y + r * ch, hand.rgb, hand.pen, hand.density)
    for k in range(cols + 2):
        c.stroke(x0 + k * cw, y, x0 + k * cw, y + rows * ch, hand.rgb, hand.pen, hand.density)
    sentences = items(pool)
    for r in range(rows):
        s = hand.strip(f"{8 + r}:00")
        if s:
            c.put(s, x0 + 4, y + r * ch + 4, hand.rgb)
        for k in range(1, cols + 1):
            if c.words < target * 1.5 and rng.random() < 0.75:
                cell(hand, rng.choice(sentences), x0 + k * cw + 5, y + r * ch + 4, cw - 10, lines=2)


def application(hand, pool, target, rng):
    c = hand.canvas
    x0, x1, y = _bounds(c)
    sh = shorts(pool)
    half = x0 + 0.46 * (x1 - x0)
    a, b = (x1 - 0.46 * (x1 - x0), x1), (x0, half)
    left = (a if hand.rtl else b)
    right = (b if hand.rtl else a)
    ya = yb = y
    for t in sh[:3]:
        ya = paragraph(hand, t, left[0], left[1], ya, align="start")
    for t in sh[3:6]:
        yb = paragraph(hand, t, right[0], right[1], yb, align="start")
    y = max(ya, yb) + 1.5 * _pitch(hand)
    y = heading(hand, heads(pool)[0], x0, x1, y, center=True)
    for p in paras(pool):
        if not _more(hand, target - 4) or y > c.y_max - 3 * _pitch(hand):
            break
        y = paragraph(hand, p, x0, x1, y, indent=3 * hand.row_mm * hand.px_mm)
    d = hand.strip(f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.{rng.randint(2005, 2026)}")
    yy = min(y + _pitch(hand), c.y_max * 0.97)
    if d:
        c.put(d, x1 - d.alpha.shape[1] if hand.rtl else x0, yy, hand.rgb)
    sig = hand.strip(sh[-1].split()[0] if sh else "-")
    if sig:
        big = cv2.resize(sig.alpha, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_LINEAR)
        c.put(sig.with_alpha(big), x0 if hand.rtl else x1 - big.shape[1], yy, hand.rgb)


def outline(hand, pool, target, rng):
    c = hand.canvas
    x0, x1, y = _bounds(c)
    step = 2.4 * hand.row_mm * hand.px_mm
    it = items(pool)
    a = b = 0
    for t in it:
        if not _more(hand, target) or y > c.y_max - _pitch(hand):
            break
        sub = rng.random() < 0.45 and a > 0
        if sub:
            b += 1
            mark = f"{a}.{b}"
        else:
            a, b = a + 1, 0
            mark = f"{a}."
        ind = step * (1.6 if sub else 0.0)
        m = hand.strip(mark)
        if m:
            c.put(m, x1 - ind - m.alpha.shape[1] if hand.rtl else x0 + ind, y, hand.rgb)
        y = paragraph(hand, t, x0 + (0 if hand.rtl else ind + step), x1 - (ind + step if hand.rtl else 0), y, align="start")


def quote(hand, pool, target, rng):
    x0, x1, y = _bounds(c := hand.canvas)
    y = c.H * 0.12
    for p in paras(pool):
        if not _more(hand, target - 3):
            break
        for line in hand.wrap(f"“{p}”", 0.7 * (x1 - x0)):
            if y > c.y_max - 3 * _pitch(hand):
                return
            row(hand, line, x0, x1, y, "center")
            y += 1.2 * _pitch(hand)
        y += _pitch(hand)
    s = hand.strip(shorts(pool)[0])
    if s:
        c.put(s, x1 - s.alpha.shape[1] if hand.rtl else x0 + 0.3 * (x1 - x0), min(y, c.y_max), hand.rgb)


def contacts(hand, pool, target, rng):
    c = hand.canvas
    x0, x1, y = _bounds(c)
    y = heading(hand, heads(pool)[0], x0, x1, y)
    for t in shorts(pool):
        if not _more(hand, target) or y > c.y_max - _pitch(hand):
            break
        name = hand.strip(hand.wrap(t, 0.55 * (x1 - x0))[0])
        num = hand.strip(f"+{rng.randint(1, 99)} {rng.randint(100, 999)} {rng.randint(100, 999)} {rng.randint(10, 99)} {rng.randint(10, 99)}")
        if not name or not num:
            continue
        nx = x1 - name.alpha.shape[1] if hand.rtl else x0
        ex = x0 if hand.rtl else x1 - num.alpha.shape[1]
        c.put(name, nx, y, hand.rgb)
        c.put(num, ex, y, hand.rgb)
        lo, hi = sorted((nx + name.alpha.shape[1] if not hand.rtl else ex + num.alpha.shape[1] + 6,
                         ex - 6 if not hand.rtl else nx - 6))
        for x in np.arange(lo + 4, hi, 7):
            c.stroke(x, y + name.alpha.shape[0] * 0.8, x + 1.5, y + name.alpha.shape[0] * 0.8, hand.rgb, hand.pen, hand.density)
        y += _pitch(hand) * 1.1


def corrections(hand, pool, target, rng):
    """A paragraph with some words struck through and a replacement written above them."""
    c = hand.canvas
    x0, x1, y = _bounds(c)
    for p in paras(pool):
        if not _more(hand, target) or y > c.y_max - 3 * _pitch(hand):
            break
        for piece in hand.wrap(p, x1 - x0):
            if y + _pitch(hand) > c.y_max or not _more(hand, target):
                break
            s = row(hand, piece, x0, x1, y, "start")
            pitch_now = max(_pitch(hand) * rng.uniform(0.98, 1.1), 0.9 * s.alpha.shape[0] if s is not None else 0.0)
            if s is not None and rng.random() < 0.35:
                w, h = s.alpha.shape[1], s.alpha.shape[0]
                a = rng.uniform(0.2, 0.7) * w
                span = min(0.18 * w, 14 * hand.char_px(piece))
                yy = y + h * 0.55
                c.stroke(x0 + a, yy, x0 + a + span, yy + rng.uniform(-2, 2), hand.rgb, hand.pen, hand.density)
                ins = hand.strip(piece.split()[0] if " " in piece else piece[:3])
                if ins:
                    c.put(ins.with_alpha(cv2.resize(ins.alpha, None, fx=0.7, fy=0.7), counted=False), x0 + a, y - 0.4 * h, hand.rgb)
            y += pitch_now
        y += 0.4 * _pitch(hand)


LAYOUTS = {f.__name__: f for f in (plain, blocks, notes, letter, math, table, diary, recipe, checklist, cornell, twocol,
                                  minutes, poem, qa, schedule, application, outline, quote, contacts, corrections)}
