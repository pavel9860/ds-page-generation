"""One handwritten page: a writer (type, style, pen, size) writes structured items (headings, paragraphs, list items, verse, tables, columns)
on a paper of a sheet format (A4 or a notebook page); ink is composed on the sheet.

make(plan, items) -> (rgb uint8 (H, W, 3), meta with the word boxes in px)
"""
import re

import numpy as np

import hwsynth as H
from hwsynth.pen import Pen, render
from hwsynth.page import smudge
from hwsynth.scripts import text_script
from hwsynth.writer import Stroke

from .paper import H as PH, W as PW, paper
from .tables import table_block
from .util import (CHAR_W, MAX_GLYPH_MM, MAX_XH, MIN_XH, clean, flat_items, map_text, pick_fonts, strip_missing, trim, uncovered,
                   usable, walk_text)

BULLETS = ("-", "-", "-", "*", "–")
DISTORT = dict(theta0=0.008, bow0=0.5, rho_t=0.35, sd_t=0.0065, rho_b=0.3, sd_b=0.55, sd_drift=0.03, sd_off=0.07, rho_x=0.5, sd_x=0.7)
SCRIPT_SCALE = {"Latin": 1.0, "Cyrillic": 1.0, "Greek": 1.0, "Han": 0.85, "Hangul": 0.85, "Hiragana": 0.85}     # other scripts: 0.7


def style_of_page(rng):
    """Habits of the page: a blank line between paragraphs or a first-line indent (never both), headings, list marker."""
    block = rng.random() < 0.5
    return dict(block=block, gap_len=float(rng.uniform(0.7, 1.0)), indent=0.0 if block else float(rng.uniform(4, 11)) * (rng.random() < 0.85),
                center=rng.random() < 0.3, underline=rng.random() < 0.35, big=float(rng.uniform(1.0, 1.3)), bullet=str(rng.choice(BULLETS)),
                list_indent=float(rng.uniform(4, 9)), after_head=int(rng.random() < 0.4), caps=rng.random() < 0.1)


def gap_lines(prev, k, st, rng):
    """Blank lines between two items when the source does not say: at most one, never doubled."""
    if prev is None:
        return 0
    if k in ("h", "table") or prev == "table":
        return 1 if rng.random() < 0.9 else 0
    if prev == "h":
        return st["after_head"]
    if k == "p" and prev in ("p", "li"):
        return 1 if st["block"] else 0
    if k == "li" and prev == "p":
        return 1 if st["block"] and rng.random() < 0.5 else 0
    return 0


def est_lines(items, cpl):
    """Text lines the items need when a line holds cpl characters (columns side by side: the longest one)."""
    n = 0.0
    for i in items:
        if i["k"] == "cols":
            n += max(est_lines(c["items"], cpl * c["w"]) + c.get("start", 0) for c in i["cols"])
        elif i["k"] == "table":
            n += 8
        else:
            n += max(1, -(-len(i["t"]) // max(int(cpl), 1))) + 0.45
    return n


def choose_size(plan, pp, w, items, scr):
    """x-height and rows per text line such that the text fills the page as asked (one estimate instead of trial layouts)."""
    full = pp.size[0] - pp.left - pp.right
    avail = pp.size[1] - pp.bottom - pp.first
    want = lambda lh: max(1, round((int(avail / lh) + 1) * plan["fill"]))
    cw = CHAR_W.get(scr, 2.4)
    if pp.pitch:
        for mult in (1, 2, 3):                                  # a line of text takes one, two or three rows of the ruling
            xh = float(np.clip(pp.pitch * mult * plan["xh_ratio"], MIN_XH, MAX_XH))
            if mult == 3 or est_lines(items, full / (xh * cw)) >= 0.92 * want(pp.pitch * mult):
                return xh, mult
    xh = max(min(plan["xh"], MAX_XH) * SCRIPT_SCALE.get(scr, 0.7), MIN_XH)
    lh = max(w.p["pitch"], w.min_pitch()) * xh
    est = est_lines(items, full / (xh * cw))
    if est < 0.92 * want(lh):                                   # little text: larger writing fills more of a free page
        xh = min(xh * min(np.sqrt(want(lh) / est), 1.4), MAX_XH)
    return xh, 1


def layout(plan, items, w, pp, xh, st, rng, mult):
    """Strokes, words, line count of the items written at x-height xh. An item may carry `gap` (blank lines before it, as in the source);
    a cols item is blocks side by side. Returns (strokes, words, lines, meta, n_want, xh, line_of, line height)."""
    w.p["size"] = xh
    w.pen.width = min(w.pen.width, xh / 5.2)                   # small writing is made with a fine tip
    w.pen.edge = min(w.pen.edge, xh / 4.2)
    full = pp.size[0] - pp.left - pp.right
    avail = pp.size[1] - pp.bottom - pp.first
    flat = list(flat_items(items))
    scr = text_script(" ".join(i["t"] for i in flat if "t" in i) or "a")
    body = next((i for i in flat if i["k"] in ("p", "li", "v")), None)
    probe = w.lines(clean(body["t"])[:60] if body else "abc def ghi", full, None)        # the line pitch the script needs at this size
    row = pp.pitch * mult if pp.pitch else 0                    # rows of the ruling one line of text takes
    lh = row or probe.pitch
    if pp.pitch and probe.pitch > row * 1.05:                   # script taller than its rows: write smaller to fit them, else on more rows
        fit = xh * row / probe.pitch * 0.97
        if fit >= MIN_XH:
            xh = fit
            w.p["size"] = xh
        else:
            lh = row * int(np.ceil(probe.pitch / row - 0.15))
    n_want = max(1, round((int(avail / lh) + 1) * plan["fill"]))
    strokes, words, meta, line_of = [], [], {}, {}
    snap = (lambda g: float(round(g))) if pp.pitch else (lambda g: g)     # on ruled paper a gap is whole lines

    def run(seq, x_off, wfull, cur):
        nonlocal meta
        prev = None
        cpl = wfull / (xh * CHAR_W.get(scr, 2.4))
        for it in seq:
            k = it["k"]
            if cur >= n_want:
                break
            if k == "cols":
                cur += snap(float(it.get("gap", 0)))
                ends = [run(c["items"], x_off + c["x"] * wfull, max(c["w"] * wfull, 30.0), cur + c.get("start", 0)) for c in it["cols"]]
                cur, prev = max(ends + [cur]), "cols"
                continue
            explicit = "gap" in it
            gap = float(it["gap"]) if explicit else gap_lines(prev, k, st, rng)
            gap *= 1.0 if (pp.pitch or explicit) else st["gap_len"]
            if prev is None and not explicit:
                gap = 0.0
            if k == "table":
                room = (n_want - cur - gap) * lh
                if room < 3 * lh:
                    break
                cur += snap(gap)
                ts, tw, th = table_block(it["rows"], w, xh, wfull, room, rng, len(words), cur * lh)
                if not ts:
                    break
                for q in ts:
                    q.p = q.p + (x_off, 0)
                strokes.extend(ts)
                words.extend(tw)
                cur, prev = cur + th / lh, "table"
                continue
            t = clean(it["t"])
            if not t:
                continue
            if k == "li":
                t = t[1:].lstrip() if t[:1] in "-–—*•·" else t
                if not re.match(r"^\(?\d{1,2}[.)]", t) and (st["bullet"] not in ("-", "–") or rng.random() < 0.7):
                    t = f"{st['bullet']} {t}"
            wid, dx, size = wfull, x_off, xh
            if k == "li":
                wid, dx = wfull - st["list_indent"], x_off + st["list_indent"]
            if k == "h":
                size = xh * st["big"] * float(it.get("scale", 1.0))
                t = t.upper() if st["caps"] else t
            cur += snap(gap)
            left = n_want - cur
            if left < 1:
                break
            t = trim(t, int(1.25 * left * cpl))
            w.p["size"] = size
            lay = w.lines(t, wid, lh if pp.pitch else None)
            while k == "h" and lay.lines > max(2, len(t.split()) // 2) and size > xh * 1.05:     # a heading that does not fit is written smaller
                size = max(size * 0.8, xh)
                w.p["size"] = size
                lay = w.lines(t, wid, lh if pp.pitch else None)
            w.p["size"] = xh
            n_keep = int(min(lay.lines, np.floor(left + 1e-6)))
            if n_keep < 1:
                break
            keep = {i for i, wd in enumerate(lay.words) if wd.line < n_keep}
            if k == "h":
                xs = np.vstack([q.p for q in lay.strokes])[:, 0]
                if st["center"]:
                    dx += max((wid - (xs.max() - xs.min())) / 2 - xs.min(), 0)
                elif rng.random() < 0.3:
                    dx += float(rng.uniform(0, 8))
            first_dx = st["indent"] if k == "p" and (prev not in (None, "h") or (st["indent"] and rng.random() < 0.5)) else 0.0
            base, mine = len(words), []
            for q in lay.strokes:
                if q.word in keep:
                    ln = lay.words[q.word].line
                    q.p = q.p + (dx + (first_dx if ln == 0 else 0), cur * lh)
                    q.word += base
                    line_of[id(q)] = int(np.floor(cur)) + ln
                    strokes.append(q)
                    mine.append(q)
            for i, wd in enumerate(lay.words):
                wd.line += int(np.floor(cur))
                words.append(wd if i in keep else None)
            if k == "h" and st["underline"] and n_keep == lay.lines == 1 and mine:      # a line under the letters, not through them
                a = np.vstack([q.p for q in mine])
                y = a[:, 1].max() + max(0.28 * size, 0.7) + rng.uniform(0, 0.5)
                xs = np.linspace(a[:, 0].min() - 0.5, a[:, 0].max() + 0.5, 14)
                ul = Stroke(np.stack([xs, y + rng.normal(0, 0.06, 14) + 0.012 * (xs - xs[0]) * rng.normal(0, 1.0)], 1))
                ul.word = -1
                strokes.append(ul)
            cur += snap(n_keep) if pp.pitch else n_keep
            prev, meta = k, lay.meta
        return cur

    cur = run(items, 0.0, full, 0.0)
    return strokes, words, cur, meta, n_want, xh, line_of, lh


def distort_lines(strokes, line_of, lh, rng, strength, cfg=DISTORT):
    """Lines on paper without rules: a slight tilt and curve of the page that most lines share, but every line has its own angle, bow
    and start (each only partly repeats the line above), and the line spacing wanders."""
    lines = {}
    for s in strokes:
        if id(s) in line_of:
            lines.setdefault(line_of[id(s)], []).append(s)
    theta0 = rng.normal(0, cfg["theta0"]) * strength
    bow0 = rng.normal(0, cfg["bow0"]) * strength
    e_t = e_b = drift = sx = 0.0
    for ln in sorted(lines):
        ss = lines[ln]
        a = np.vstack([s.p for s in ss])
        xc, wl = (a[:, 0].min() + a[:, 0].max()) / 2, max(np.ptp(a[:, 0]), 1.0)
        e_t = cfg["rho_t"] * e_t + rng.normal(0, cfg["sd_t"]) * strength
        e_b = cfg["rho_b"] * e_b + rng.normal(0, cfg["sd_b"]) * strength
        drift += rng.normal(0, cfg["sd_drift"] * lh) * strength
        sx = cfg["rho_x"] * sx + rng.normal(0, cfg["sd_x"]) * strength
        off = drift + rng.normal(0, cfg["sd_off"] * lh) * strength
        theta, bow = theta0 + e_t, (bow0 + e_b) * min(wl / 100.0, 1.5)
        for s in ss:
            q = s.p.copy()
            u = (q[:, 0] - xc) / (wl / 2)
            q[:, 1] += off + theta * (q[:, 0] - xc) + bow * (u ** 2 - 1 / 3)
            q[:, 0] += sx
            s.p = q


def word_strokes(strokes):
    by = {}
    for q in strokes:
        if q.word >= 0:
            by.setdefault(q.word, []).append(q.p)
    return by


def tallest(by, words):
    """Height [mm] of the tallest word: its ink from the top of an ascender to the bottom of a descender."""
    return max([float(np.ptp(np.concatenate([p[:, 1] for p in pts]))) for i, pts in by.items() if i < len(words) and words[i] is not None] or [0.0])


def contrast(ink, pen):
    """Ink of a minimum darkness: a hard pencil or a light push does not disappear into the paper."""
    m = ink > 0.08
    if m.sum() < 50:
        return ink
    p = float(np.percentile(ink[m], 97))
    return np.clip(ink * float(np.clip((0.62 if "pencil" in pen.tool else 0.9) / max(p, 1e-3), 1.0, 2.6)), 0, 1)


def prepare(plan, w, items):
    """Fonts that cover the page; the text cleaned (strict: nothing dropped); no jumping letters, no absurd widths."""
    for g, v in (("caps", 0.0), ("allo", 0.0), ("size_sd", 0.05), ("base_sd", 0.05), ("rot_sd", 2.0), ("xs", 1.15), ("ws", 1.25)):
        w.p[g] = min(w.p[g], v)
    text = clean(" ".join(walk_text(items)))
    mapping = pick_fonts(w, text, plan.get("prefer_font"))
    own = w.font
    w.font = lambda scr, _m=mapping, _o=own: _m.get(scr) or _o(scr)
    items = map_text(items, clean if plan.get("strict") else lambda t: strip_missing(w, clean(t)))
    return [i for i in items if i["k"] in ("table", "cols") or usable(i["t"])], mapping, uncovered(w, text)


def make(plan, items):
    rng = np.random.default_rng(plan["seed"])
    pp = paper(plan["paper"], rng, plan.get("fmt", "a4"), plan.get("dense", False))
    w = H.sample_writer(plan["persona"], plan["style"], seed=plan["wseed"])
    w.pen = Pen.sample(np.random.default_rng(plan["seed"] + 7), {plan["pen"]: 1})
    items, mapping, missing = prepare(plan, w, items)
    scr = text_script(" ".join(walk_text(items)) or "a")
    st = style_of_page(rng)
    xh, mult = choose_size(plan, pp, w, items, scr)
    state = rng.bit_generator.state
    for _ in range(3):                                          # a word taller than the cap: write smaller (rarely more than once)
        rng.bit_generator.state = state
        strokes, words, cur, lmeta, n_want, xh, line_of, lh = layout(plan, items, w, pp, xh, st, rng, mult)
        by = word_strokes(strokes)
        tall = tallest(by, words)
        if tall <= MAX_GLYPH_MM or xh <= 1.6:
            break
        xh *= 0.85 * MAX_GLYPH_MM / tall if tall > 1.3 * MAX_GLYPH_MM else 0.85
    if not pp.pitch or pp.kind == "dotted":                     # no rules to follow: wandering lines
        distort_lines(strokes, line_of, lh, rng, 1.0 if not pp.pitch else 0.5)
        by = word_strokes(strokes)
    dy = plan["dy"] if pp.pitch else 0.0
    top = pp.first - dy + min(q.p[:, 1].min() for q in strokes)
    shift = 0.0 if pp.pitch else max(0.0, 6.0 - top)            # nothing above the sheet
    origin = np.array((-pp.left, -(pp.first - dy + shift)))
    ink, _ = render(strokes, w.pen, w.p, pp.k * 25.4, w.rng, (PH, PW), origin)
    ink = contrast(ink, w.pen)
    if w.left:
        ink = smudge(ink, w.pen, pp.k * 25.4, w.rng)
    rgb = pp.img * (1 - ink[..., None] * (1 - np.asarray(w.pen.ink)[None, None]))
    out = []
    for i, wd in enumerate(words):
        if wd is not None and by.get(i):
            a = np.vstack(by[i])
            out.append((wd.text, [int(v) for v in (np.r_[a.min(0), a.max(0)] - np.r_[origin, origin]) * pp.k], wd.struck))
    meta = dict(lmeta, paper=pp.kind, fmt=pp.fmt, paper_meta=pp.meta, pitch=pp.pitch, xh_mm=round(xh, 2), lines=round(cur, 1), lines_want=n_want,
                rows=mult, tallest_mm=round(tall, 1), script=scr, chars=sum(len(x[0]) for x in out),
                tables=sum(i["k"] == "table" for i in items), uncovered_chars=missing,
                fonts={k: str(v).split("/")[-1] for k, v in mapping.items()}, struct=dict(st), words=out)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8), meta
