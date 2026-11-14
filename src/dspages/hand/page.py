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
from .util import (CHAR_W, MAX_GLYPH_MM, MAX_XH, MIN_XH, MIN_XH_SCRIPT, clean, flat_items, map_text, pick_fonts, strip_missing, trim, uncovered,
                   usable, walk_text)

SENT_END = re.compile(r"[.!?。！？…؟।॥։]['\"”»)]*$")
BULLETS = ("-", "-", "-", "*", "–")
DISTORT = dict(theta0=0.008, bow0=0.5, rho_t=0.35, sd_t=0.0065, rho_b=0.3, sd_b=0.55, sd_drift=0.03, sd_off=0.07, rho_x=0.5, sd_x=0.7)
SCRIPT_SCALE = {"Latin": 1.0, "Cyrillic": 1.0, "Greek": 1.0, "Han": 0.85, "Hangul": 0.85, "Hiragana": 0.85}     # other scripts: 0.7


def style_of_page(rng):
    """Habits of the page: a blank line between paragraphs or a first-line indent (never both), headings, list marker."""
    block = rng.random() < 0.5
    return dict(block=block, gap_len=float(rng.uniform(0.7, 1.0)), indent=0.0 if block else float(rng.uniform(4, 11)) * (rng.random() < 0.85),
                center=rng.random() < 0.3, underline=rng.random() < 0.35, big=float(rng.uniform(1.0, 1.3)), bullet=str(rng.choice(BULLETS)),
                list_indent=float(rng.uniform(4, 9)), after_head=int(rng.random() < 0.4), caps=False)


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


def src_gap(g):
    """Blank lines for a gap of the source page: one line for a paragraph or section break, two for a wide one (a typeset page has far more
    lines than a handwritten one; its gaps written one to one would leave holes)."""
    return 0.0 if g <= 0 else 2.0 if g >= 3 else 1.0


def line_shifts(lay, wid, align):
    """{line: x shift} that centres or right-aligns every line of a laid out item inside the width wid."""
    if not align:
        return {}
    ext = {}
    for q in lay.strokes:
        ln = lay.words[q.word].line
        lo, hi = ext.get(ln, (np.inf, -np.inf))
        ext[ln] = (min(lo, q.p[:, 0].min()), max(hi, q.p[:, 0].max()))
    if align == "center":
        return {ln: max((wid - (hi - lo)) / 2 - lo, 0.0) for ln, (lo, hi) in ext.items()}
    return {ln: max(wid - hi, 0.0) for ln, (lo, hi) in ext.items()}


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
        n += src_gap(i.get("gap", 0))
    return n


def choose_size(plan, pp, w, items, scr):
    """x-height and rows per text line such that the text fills the page as asked (one estimate instead of trial layouts)."""
    full = pp.size[0] - pp.left - pp.right
    avail = pp.size[1] - pp.bottom - pp.first
    want = lambda lh: max(1, round((int(avail / lh) + 1) * plan["fill"]))
    cw = CHAR_W.get(scr, 2.4)
    lo = max(MIN_XH, MIN_XH_SCRIPT.get(scr, 0))
    if pp.pitch:
        for mult in (1, 2, 3):                                  # a line of text takes one, two or three rows of the ruling
            xh = float(np.clip(pp.pitch * mult * plan["xh_ratio"], lo, MAX_XH))
            if mult == 3 or est_lines(items, full / (xh * cw)) >= 0.92 * want(pp.pitch * mult):
                return xh, mult
    xh = max(min(plan["xh"], MAX_XH) * SCRIPT_SCALE.get(scr, 0.7), lo)
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
    body = next((i for i in flat if i["k"] in ("p", "li", "v", "l")), None)
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
            gap = src_gap(it["gap"]) if explicit else gap_lines(prev, k, st, rng)
            gap *= 1.0 if (pp.pitch or explicit) else st["gap_len"]
            if prev is None and not explicit:
                gap = 0.0
            if k == "table":
                room = (n_want - cur - gap) * lh
                if room < 3 * lh:                                          # a table that does not fit is left out: the items after it still fill the page
                    continue
                ts, tw, th = table_block(it["rows"], w, xh, wfull, room, rng, len(words), (cur + snap(gap)) * lh)
                if not ts:
                    continue
                cur += snap(gap)
                for q in ts:
                    q.p = q.p + (x_off, 0)
                strokes.extend(ts)
                words.extend(tw)
                cur, prev = cur + (np.ceil(th / lh) if pp.pitch else th / lh), "table"      # on ruled paper the table takes whole rows
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
                ind = st["list_indent"] * (1 + 0.8 * it.get("lvl", 0))
                wid, dx = wfull - ind, x_off + ind
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
            if n_keep < lay.lines and k in ("p", "li", "v"):            # a cut paragraph ends at its last sentence end, not in mid-sentence
                ends = [i for i in sorted(keep) if SENT_END.search(lay.words[i].text)]
                if not ends or ends[-1] < 0.4 * len(keep):
                    if words:
                        break                                                 # the page ends after the item before; on an empty page the lines that fit stay
                else:
                    keep = {i for i in keep if i <= ends[-1]}
                    n_keep = max(lay.words[i].line for i in keep) + 1
            align = it.get("align") or ("center" if k == "h" and st["center"] else None)
            shift = line_shifts(lay, wid, align)
            if k == "h" and not align and rng.random() < 0.3:
                dx += float(rng.uniform(0, 8))
            first_dx = st["indent"] if k == "p" and (prev not in (None, "h") or (st["indent"] and rng.random() < 0.5)) else 0.0
            base, mine = len(words), []
            for q in lay.strokes:
                if q.word in keep:
                    ln = lay.words[q.word].line
                    q.p = q.p + (dx + shift.get(ln, 0.0) + (first_dx if ln == 0 and not align else 0), cur * lh)
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
            cur += snap(n_keep) if pp.pitch else n_keep * lay.pitch / lh       # unruled: the item's own lines (a heading's are taller than the body's)
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


def word_jitter(strokes, wj, xh, rng):
    """Every word is rotated, shifted, scaled and sheared a little about its own centre and baseline: real hands vary from word to word far more
    than the writer model's per-letter jitter does. wj = (rotation sd [deg], shift sd [x-heights], scale sd, shear sd [deg])."""
    th, dy, sc, sh = wj
    by = {}
    for q in strokes:
        if q.word >= 0:
            by.setdefault(q.word, []).append(q)
    for ws in by.values():
        a = np.vstack([q.p for q in ws])
        cx, base = (a[:, 0].min() + a[:, 0].max()) / 2, np.median(a[:, 1])
        r, k, sh_ = np.radians(rng.normal(0, th)), 1 + rng.normal(0, sc), np.tan(np.radians(rng.normal(0, sh)))
        c, s_ = np.cos(r), np.sin(r)
        shift = rng.normal(0, dy) * xh
        for q in ws:
            x, y = (q.p[:, 0] - cx) * k, (q.p[:, 1] - base) * k
            x = x + sh_ * y
            q.p = np.stack([cx + c * x - s_ * y, base + shift + s_ * x + c * y], 1)


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


def _setup(plan, items):
    """(rng, paper, writer, items mapped to its fonts, mapping, missing, script) of a page plan: what make() writes with."""
    rng = np.random.default_rng(plan["seed"])
    pp = paper(plan["paper"], rng, plan.get("fmt", "a4"), plan.get("dense", False))
    w = H.sample_writer(plan["persona"], plan["style"], seed=plan["wseed"])
    from hwdet.pagegen.handfonts import Restricted
    w.fonts = Restricted(w.fonts)                                 # the realistic fonts of each script (one policy with the filled forms)
    w.pen = Pen.sample(np.random.default_rng(plan["seed"] + 7), {plan["pen"]: 1})
    w.pen.width *= plan.get("pen_scale", 1.0)
    items, mapping, missing = prepare(plan, w, items)
    w.p.update(plan.get("tune", {}))                          # per-page overrides of the writer genes (after the caps of prepare)
    return rng, pp, w, items, mapping, missing, text_script(" ".join(walk_text(items)) or "a")


def fills(plan, items, part=0.8):
    """Whether the items fill at least `part` of the lines the page wants, by the estimate choose_size sizes the writing with (est_lines at the
    x-height it picks): nothing written (rare.write's test of a page cut short, made before the writing)."""
    _, pp, w, items, _, _, scr = _setup(plan, items)
    xh, mult = choose_size(plan, pp, w, items, scr)
    full, avail = pp.size[0] - pp.left - pp.right, pp.size[1] - pp.bottom - pp.first
    lh = pp.pitch * mult if pp.pitch else max(w.p["pitch"], w.min_pitch()) * xh
    return est_lines(items, full / (xh * CHAR_W.get(scr, 2.4))) >= part * max(1, round((int(avail / lh) + 1) * plan["fill"]))


def make(plan, items):
    rng, pp, w, items, mapping, missing, scr = _setup(plan, items)
    st = style_of_page(rng)
    xh, mult = choose_size(plan, pp, w, items, scr)
    state = rng.bit_generator.state
    for _ in range(3):                                          # a word taller than the cap: write smaller (rarely more than once)
        rng.bit_generator.state = state
        strokes, words, cur, lmeta, n_want, xh, line_of, lh = layout(plan, items, w, pp, xh, st, rng, mult)
        by = word_strokes(strokes)
        tall = tallest(by, words)
        lo = max(MIN_XH, MIN_XH_SCRIPT.get(scr, 0))
        if tall <= MAX_GLYPH_MM or xh <= lo:
            break
        xh = max(xh * (0.85 * MAX_GLYPH_MM / tall if tall > 1.3 * MAX_GLYPH_MM else 0.85), lo)
    if not pp.pitch or pp.kind == "dotted":                     # no rules to follow: wandering lines
        distort_lines(strokes, line_of, lh, rng, 1.0 if not pp.pitch else 0.5)
        by = word_strokes(strokes)
    if any(plan.get("wj", ())):
        word_jitter(strokes, plan["wj"], xh, rng)
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
