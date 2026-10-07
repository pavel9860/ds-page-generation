"""A PDF page restyled on its own template: every box keeps its place, only the text in it is set again.

Body paragraphs of a column (a stretch of them between headings, pictures and gaps) are text frames. Their text, and the
text of the following pages, is one stream poured through the frames in reading order until they are full. Every other
block (heading, table cell, label, header, footer) is a fixed box set at its place; a page-wide shrink keeps its text
inside it. Rules, frames and pictures stay where the source has them, so nothing can overlap and a column has no gap
while the stream lasts.
"""
import bisect
import itertools
import math
import unicodedata
from collections import Counter, defaultdict, deque

import numpy as np
import pymupdf
from PIL import Image

from . import refont as R
from .refont import C, wrap


def _size_class(size):
    return round(math.log(size) / math.log(C.head_ratio))


def _columns(cand):
    """id -> (x0, x1) of the column of each candidate: candidates whose x ranges overlap by half the narrower one share
    a column (a short line or a block without a detected column joins the column it lies in)."""
    parent = {id(b): id(b) for b in cand}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(cand):
        for b in cand[i + 1:]:
            A, B = a["bbox"], b["bbox"]
            if min(A.x1, B.x1) - max(A.x0, B.x0) > 0.5 * min(A.width, B.width):
                parent[root(id(a))] = root(id(b))
    span = {}
    for b in cand:
        r = root(id(b))
        lo, hi = span.get(r, (b["bbox"].x0, b["bbox"].x1))
        span[r] = (min(lo, b["bbox"].x0), max(hi, b["bbox"].x1))
    return {id(b): span[root(id(b))] for b in cand}


def _cells(cand, blocks, rules, g):
    """Candidates that sit side by side with another block between the same two rules: the cells of a table row."""
    ys = [r[2] for r in rules]
    bands = defaultdict(list)
    for b in blocks:
        bb = b["bbox"]
        p = dict(x0=bb.x0 * g, x1=bb.x1 * g, y0=bb.y0 * g, y1=bb.y1 * g, tol=C.row_tol_em * b["size"] * g)
        band = R._band(rules, ys, p)
        if band:
            bands[(round(band[0] / C.table_step), round(band[1] / C.table_step))].append(b)
    def beside(a, b):
        A, B = a["bbox"], b["bbox"]
        return (A.x1 <= B.x0 or B.x1 <= A.x0) and min(A.y1, B.y1) - max(A.y0, B.y0) > 0.5 * min(A.height, B.height)
    return {id(a) for row in bands.values() for a in row if any(b is not a and beside(a, b) for b in row)}


def _mates(cand, col):
    """Candidates of a column that stand side by side on one line (the fields of a row), not paragraphs."""
    out = set()
    ordered = sorted(cand, key=lambda b: b["bbox"].y0)
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if b["bbox"].y0 >= a["bbox"].y1:
                break
            overlap = min(a["bbox"].y1, b["bbox"].y1) - max(a["bbox"].y0, b["bbox"].y0)
            if (col[id(a)] == col[id(b)] and overlap > C.row_overlap * min(a["bbox"].height, b["bbox"].height)
                    and (b["bbox"].x0 >= a["bbox"].x1 - 1 or a["bbox"].x0 >= b["bbox"].x1 - 1)):
                out |= {id(a), id(b)}
    return out


def _runs(cand, others, col):
    """Candidates of a column in runs: a run ends at a gap of frame_gap lines or at a block between two paragraphs."""
    runs = []
    for b in sorted(cand, key=lambda b: (col[id(b)], b["bbox"].y0)):
        prev = runs[-1][-1] if runs else None
        pitch = b["src_pitch"]
        split = (prev is None or col[id(prev)] != col[id(b)] or b["bbox"].y0 - prev["bbox"].y1 > C.frame_gap * pitch
                 or any(min(o["bbox"].x1, b["bbox"].x1) > max(o["bbox"].x0, b["bbox"].x0) and prev["bbox"].y1 - pitch / 2 <= o["bbox"].y0
                        and o["bbox"].y1 <= b["bbox"].y0 + pitch / 2 for o in others))
        if split:
            runs.append([])
        runs[-1].append(b)
    return runs


def _walled(cand, blocks, walls, g):
    """Candidates between two vertical rules with a block beyond one of them on their level: cells of a table ruled by
    its columns only."""
    out = set()
    for b in cand:
        x0, x1, mid = b["bbox"].x0 * g, b["bbox"].x1 * g, (b["bbox"].y0 + b["bbox"].y1) / 2 * g
        xs = [x for x, a, z in walls if a <= mid <= z]
        left, right = max((x for x in xs if x <= x0 + 2), default=None), min((x for x in xs if x >= x1 - 2), default=None)
        if left is not None and right is not None and any(
                o is not b and min(o["bbox"].y1, b["bbox"].y1) > max(o["bbox"].y0, b["bbox"].y0)
                and (o["bbox"].x1 * g <= left + 2 or o["bbox"].x0 * g >= right - 2) for o in blocks):
            out.add(id(b))
    return out


def split_page(blocks, rules, g, boxes=(), walls=()):
    """-> (sections, fixed): the frames (runs of body paragraphs of a column) grouped in sections, the frames between two
    blocks that span the columns, each in reading order; and the other blocks. A paragraph in a box (a cell, a panel) or on
    a tabbed line is not a frame's."""
    classes = {_size_class(b["size"]) for b in blocks if not b["pinned"] and not b.get("image") and len(b["lines"]) >= C.paragraph_lines}
    cand = [b for b in blocks if not b["pinned"] and not b.get("image") and _size_class(b["size"]) in classes
            and not any(l.get("tabbed") for l in b["lines"]) and not any(x.contains(b["bbox"] + (2, 2, -2, -2)) for x in boxes)]
    texts = [b for b in blocks if not b["pinned"] and not b.get("image")]
    cells = _cells(cand, texts, rules, g) | _mates(cand, _columns(cand)) | _walled(cand, texts, walls, g)
    cand = [b for b in cand if id(b) not in cells]
    col = _columns(cand)
    taken = {id(b) for b in cand}
    others = [b for b in blocks if id(b) not in taken and not b["pinned"]]
    runs = [r for r in _runs(cand, others, col) if sum(len(b["lines"]) for b in r) >= C.frame_lines and max(len(b["lines"]) for b in r) >= C.paragraph_lines]
    framed = {id(b) for r in runs for b in r}
    cols = set(col.values())
    breaks = sorted(b["bbox"].y1 for b in others if sum(min(b["bbox"].x1, c[1]) - max(b["bbox"].x0, c[0]) > 0 for c in cols) > min(1, len(cols) - 1))
    rtl = sum(b["rtl"] for b in cand) > len(cand) / 2
    region = lambda r: bisect.bisect_right(breaks, r[0]["bbox"].y0 + 1)
    runs.sort(key=lambda r: (region(r), -col[id(r[0])][0] if rtl else col[id(r[0])][0], r[0]["bbox"].y0))
    return [list(sec) for _, sec in itertools.groupby(runs, key=region)], [b for b in blocks if id(b) not in framed]


def _letters(elems, blocks):
    """The drawings that are letters (small filled shapes inside a text line) repeat the line's text, a bold made of
    outlines: they are left out of the graphics, and a span they cover is bold. -> the other drawings."""
    lines = [l for b in blocks if not b.get("image") for l in b["lines"]]
    covered, keep = defaultdict(float), []
    for e in elems:
        r = e[0]["rect"]
        host = next((l for l in lines if e[0].get("fill") is not None and R._gray_f(e[0]["fill"]) < C.letter_gray and r.height <= C.letter_em * l["bbox"].height
                     and r.width <= C.letter_em * l["bbox"].height and R._overlaps(l["bbox"], r, C.contain_share)), None) if r.get_area() > 0 else None
        span = host and max(host["spans"], key=lambda sp: (r & sp["bbox"]).get_area())
        if span is not None and (r & span["bbox"]).get_area() > 0:
            covered[id(span)] += r.width
        else:
            keep.append(e)
    for l in lines:
        for sp in l["spans"]:
            sp["bold"] = sp["bold"] or covered[id(sp)] >= C.letter_cover * sp["bbox"].width
    return keep


def _metrics(b, toks, px_of, g):
    """Line metrics of a paragraph set at the page's sizes: ascent, descent, pitch, indents, alignment."""
    bpx = px_of(b["size"])
    lettered = next((t for t in toks if t["path"] is not None), None)
    asc, desc = R.font(lettered["path"], bpx, lettered["wght"]).getmetrics() if lettered else (C.ascent_share * bpx, (1 - C.ascent_share) * bpx)
    ind = R.indents(b, g, bpx) if len(b["lines"]) > 1 else (0.0, 0.0)
    return dict(asc=asc, desc=desc, pitch=R.line_pitch(b, bpx, asc, desc), ind=ind, align=b["align"], rtl=b["rtl"])


def _set_lines(lines, x0, W, top, m, final, out, kind, pinned=False):
    """Lines set from `top` in a box at x0, W wide, as the paragraph is aligned; the last line of a paragraph is not stretched."""
    rtl, pitch, asc, desc = m["rtl"], m["pitch"], m["asc"], m["desc"]
    y = top + pitch / 2 + (asc - desc) / 2
    for li, ln in enumerate(lines):
        lw = sum(w for _, _, w, _ in ln) + sum(sw for _, _, _, sw in ln[1:])
        ind_l = m["ind"][0] if li == 0 else m["ind"][1]
        a, extra = m["align"], 0.0
        if a == "justify" and ((final and li == len(lines) - 1) or len(ln) < 2):
            a = "right" if rtl else "left"
        elif a == "justify":
            extra = (W - ind_l - lw) / (len(ln) - 1)
            space = max((sw for _, _, _, sw in ln[1:]), default=0.0)
            if space and extra > C.max_stretch * space:
                a, extra = ("right" if rtl else "left"), 0.0
        if a == "center":
            x = x0 + (W - lw) / 2
        elif a == "right":
            x = x0 + W - lw - (ind_l if rtl else 0)
        else:
            x = x0 + (0 if rtl else ind_l)
        if rtl and m["align"] == "justify" and a == "justify":
            x = x0
        start = x
        for t, f, w, sw in (R.visual(ln) if rtl else ln):
            x += 0.0 if rtl else sw
            if t["path"] is None:
                out["glyphs"].append((x, top + C.ascent_share * t["glyph"][1].height * t["px"] / t["size"] if t["tall"] else y, t))
            else:
                out["items"].append((x, y, f, t["text"], t["gray"], "rtl" if t["rtl"] else "ltr"))
            x += w + (sw if rtl else 0.0) + extra
        out["rects"].append(dict(x0=start, x1=x - extra, y0=y - asc, y1=y + desc, block=len(out["rects"]), src=(x0, x0 + W),
                                 pinned=pinned, kind=kind, align=a, line=li, lines=len(lines), tokens=len(ln)))
        y += pitch
    return top + len(lines) * pitch


def _units(frames, g, k, cap):
    """The paragraphs of frames in reading order, each with the gap before it (the source gap, scaled and capped)."""
    units = []
    for fr in frames:
        for i, b in enumerate(fr):
            gap = max(b["bbox"].y0 - fr[i - 1]["bbox"].y1, 0.0) * g * k if i else 0.0
            units.append(dict(b=b, toks=None, cont=False, gap=min(gap, cap)))
    return units


def _available(units, more):
    """Whether a paragraph is left, taking the following pages for it."""
    while not units:
        if not more():
            return False
    return True


def _free(obstacles, x0, x1, top, bottom):
    """The widest stretch (x, width) of x0..x1 that no obstacle covers between top and bottom (an obstacle that only grazes
    the band, by less than frame_slack of it, does not count)."""
    free = [(x0, x1)]
    tol = C.frame_slack * (bottom - top)
    for ox0, oy0, ox1, oy1 in obstacles:
        if oy0 < bottom - tol and oy1 > top + tol:
            free = [part for a, b in free for part in ((a, min(b, ox0)), (max(a, ox1), b)) if part[1] > part[0]]
    a, b = max(free, key=lambda f: f[1] - f[0], default=(x0, x0))
    return a, b - a


def _stretch(out, start, top, k):
    """The lines a frame got since `start` spread down by k from its top (the leading grows) so a short text fills it."""
    out["items"][start["items"]:] = [(x, top + (y - top) * k, *rest) for x, y, *rest in out["items"][start["items"]:]]
    out["glyphs"][start["glyphs"]:] = [(x, top + (y - top) * k, t) for x, y, t in out["glyphs"][start["glyphs"]:]]
    for r in out["rects"][start["rects"]:]:
        r["y0"], r["y1"] = top + (r["y0"] - top) * k, top + (r["y1"] - top) * k


def _pour(frames, units, more, obstacles, typesetter, role_of, px_of, g, width, out, up, last, cut=0.0):
    """The stream of paragraphs poured through the frames in turn, line by line, around the obstacles (a box or a picture
    that lies in a frame); a paragraph that does not end in a frame goes on in the next. `more` adds the paragraphs of the
    next page (False when there are none). The frames are moved up by `up` px; those of the last section keep their bottom
    (they take the room the sections above left) and a short one spreads its lines; the others end `cut` px higher. ->
    (whether every frame is full, the lowest line's bottom)."""
    full, low = True, 0.0
    for fr in frames:
        x0, x1 = max(min(b["bbox"].x0 for b in fr) * g, 0.0), min(max(b["bbox"].x1 for b in fr) * g, width)
        top, bottom = fr[0]["bbox"].y0 * g - up, fr[-1]["bbox"].y1 * g - (0.0 if last else up + cut)
        near = [o for o in obstacles if o[0] < x1 and o[2] > x0 and o[1] < bottom and o[3] > top]
        y, ended, start = top, False, {k: len(out[k]) for k in ("items", "glyphs", "rects")}
        while not ended and _available(units, more):
            u = units[0]
            if u["toks"] is None:
                u["toks"] = R.tokens(u["b"], typesetter, role_of, px_of)
                u["m"] = _metrics(u["b"], u["toks"], px_of, g) if u["toks"] else None
            if not u["toks"]:
                units.popleft()
                continue
            m = u["m"]
            y += u["gap"] if y > top else 0.0
            while u["toks"]:
                ended = y + m["pitch"] > bottom + C.frame_slack * m["pitch"]
                if ended:
                    break
                ax, aw = _free(near, x0, x1, y, y + m["pitch"])
                if aw < C.min_line_em * m["pitch"]:
                    y += m["pitch"]
                    continue
                ind = m["ind"][1] if u["cont"] else m["ind"][0]
                lines = wrap(u["toks"], aw, ind, m["ind"][1])
                y = _set_lines(lines[:1], ax, aw, y, dict(m, ind=(ind, ind)), len(lines) == 1, out, "frame")
                out["placed"] += sum(len(t["text"].replace(" ", "")) for t, *_ in lines[0])
                u["toks"], u["cont"] = [t for ln in lines[1:] for t, *_ in ln], True
            if not u["toks"]:
                units.popleft()
        if last and not ended and not near and y > top:
            _stretch(out, start, top, min((bottom - top) / (y - top), C.leading_max))
            y = top + (y - top) * min((bottom - top) / (y - top), C.leading_max)
        out["fill"].append((_size_class(fr[0]["size"]), y - top, bottom - top))
        full = full and ended
        low = max(low, bottom if ended else y)
    return full, low


def _balance(frames, units, obstacles, typesetter, role_of, px_of, g, width, up, height):
    """How much higher the frames of a section can end with all its text still in them: the columns balanced, the section
    as short as its text (bisection over trial pours)."""
    for u in units:
        if u["toks"] is None:
            u["toks"] = R.tokens(u["b"], typesetter, role_of, px_of)
            u["m"] = _metrics(u["b"], u["toks"], px_of, g) if u["toks"] else None

    def fits(cut):
        trial = deque(dict(u) for u in units)
        _pour(frames, trial, lambda: False, obstacles, typesetter, role_of, px_of, g, width,
              dict(items=[], glyphs=[], rects=[], placed=0, fill=[]), up, False, cut)
        return not trial

    if not fits(0.0):
        return 0.0
    lo, hi = 0.0, height
    while hi - lo > C.balance_px:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fits(mid) else (lo, mid)
    return lo


def _fill_frames(build, mult, res):
    """Frames the stream left short (a page with no text after it) are filled by growing the letters of their size class:
    each class by what its frames lack. The grown page is taken when it still sets all the text the first did."""
    short = defaultdict(lambda: [0.0, 0.0])
    for cls, used, room in res["out"]["fill"]:
        short[cls][0] += used
        short[cls][1] += room
    grow = {c: min((room / max(used, 1.0)) ** 0.5 * C.shrink, C.grow_max) for c, (used, room) in short.items() if used < C.fill_share * room}
    if not grow:
        return res
    grown = build(mult, True, grow)
    return grown if grown["out"]["placed"] >= res["out"]["placed"] else res


def layout(src, number, rng, fonts, body_pt, scale, width_px):
    """Layout of page `number` of a refont.Source with its body text at body_pt on the sheet (210 mm wide, width_px wide), the
    other sizes keeping their ratio to the body's times `scale`."""
    pw, ph = src[number].rect.width, src[number].rect.height
    g = width_px / pw
    first = R.page_blocks(src, number, True)
    bottom = max((b["bbox"].y1 for b in first if not b["pinned"]), default=None)
    elems = _letters(R.graphics(src, number, True, ph, pw, below=bottom), first)
    spans = [s for b in first for l in b["lines"] for s in l["spans"]]
    body = Counter({k: sum(len(s["text"]) for s in spans if round(s["size"] * 2) / 2 == k) for k in {round(s["size"] * 2) / 2 for s in spans}}).most_common(1)[0][0]
    sc_font = body_pt * (width_px / C.sheet_pt_w) / body
    chars = {c for s in spans for c in s["text"] if not c.isspace()}
    need = {(s["bold"], s["italic"]) for s in spans if not s["mono"]}
    cols_px = [(c[0] * g, c[1] * g) for c in R._columns(first)]
    first += R.pictures(src, number, ph, pw, below=bottom)
    sections, fixed = split_page(first, R._hrules(elems, g), g, R.cell_boxes(elems, pw * ph), R._vrules(elems, g))
    frames = [r for sec in sections for r in sec]
    role_of = lambda sz: "head" if sz > body * C.head_ratio else "body"
    px_of = lambda sz: sz * sc_font * R.ratio(sz, body, scale)
    texts = [b for b in fixed if not b.get("image")]
    walls = R._vrules(elems, g)

    def prepare_fixed(typesetter, px):
        return [p for p in (R.prepare(b, first, pw, g, typesetter, role_of, px, cols_px, walls) for b in texts) if p]

    def need_shrink(p):
        return p["fit"] if not p["multi"] else (len(p["lines"]) * p["pitch"] / max(p["y1"] - p["y0"], p["pitch"])) ** 0.5

    best = None
    for _ in range(C.font_tries):
        typesetter = R.Typesetter(fonts, rng, chars, need)
        fit = max((need_shrink(p) for p in prepare_fixed(typesetter, px_of)), default=0.0)
        best = best if best and best[0] <= fit else (fit, typesetter)
        if fit <= 1.0:
            break
    fit, typesetter = best
    mult = max(min(1.0, 1.0 / fit) if fit else 1.0, C.fit_floor)
    running = R.running_bands(first)

    fixed_cache, later = {}, {}

    def build(mult, pages, grow=None):
        """The page set at this shrink; `pages`: whether the stream may take the following pages."""
        px = lambda sz: px_of(sz) * mult * (grow or {}).get(_size_class(sz), 1.0)
        out = dict(items=[], glyphs=[], rects=[], placed=0, fill=[])
        key = (round(mult, 9), tuple(sorted((grow or {}).items())))
        if key not in fixed_cache:
            fixed_cache[key] = prepare_fixed(typesetter, px)
        fixed_ps = fixed_cache[key]
        pics = [(b["bbox"].x0 * g, b["bbox"].y0 * g, dict(b, scale=1.0)) for b in fixed if b.get("image")]
        st, k = dict(used=1), px(body) / (body * g)
        gap = C.gap_body * px(body)
        units = [deque(_units(sec, g, k, gap)) for sec in sections]

        def more():
            if not pages or st["used"] >= C.max_pages or number + st["used"] >= len(src):
                return False
            nxt = number + st["used"]
            st["used"] += 1
            if nxt not in later:
                blocks = R.page_blocks(src, nxt, False, running) if R.text_page(src, nxt) else []
                gfx = R.graphics(src, nxt, False, ph, pw, number)
                later[nxt] = split_page(blocks, R._hrules(gfx, g), g, R.cell_boxes(gfx, pw * ph), R._vrules(gfx, g))[0]
            units[-1].extend(_units([r for sec in later[nxt] for r in sec], g, k, gap))
            return True

        obstacles = [(p["bx0"], p["y0"], p["bx0"] + p["W"], p["y1"]) for p in fixed_ps if not p["pinned"]]
        obstacles += [(x, t, x + b["rect"].width * g, t + b["rect"].height * g) for x, t, b in pics if not b["pinned"]]
        full, done = True, []

        def up(y):
            """How far what starts at y moved up: the room left by the sections above it."""
            return sum(s for bottom, s in done if bottom <= y + 1)

        for n, sec in enumerate(sections):
            last = n == len(sections) - 1
            top, bottom = min(fr[0]["bbox"].y0 for fr in sec) * g, max(fr[-1]["bbox"].y1 for fr in sec) * g
            d = up(top)
            moved = [(a, b - up(b), c, e - up(b)) for a, b, c, e in obstacles]
            cut = 0.0 if last or len(sec) < 2 else _balance(sec, units[n], moved, typesetter, role_of, px, g, width_px, d, bottom - top)
            ok, low = _pour(sec, units[n], more if last else (lambda: False), moved, typesetter, role_of, px, g, width_px, out, d, last, cut)
            full = full and ok
            done.append((bottom, 0.0 if last else max(bottom - d - low, 0.0)))
        for p in fixed_ps:
            d = 0.0 if p["pinned"] else up(p["y0"])
            _set_lines(p["lines"], p["bx0"], p["W"], p["y0"] - d + (p["y1"] - p["y0"] - len(p["lines"]) * p["pitch"]) / 2, p, True, out, "fixed", p["pinned"])
        pics = [(x, t if b["pinned"] else t - up(t), b) for x, t, b in pics]
        drawings = [(dr, fx, 0.0 if fx else up(dr["rect"].y0 * g)) for dr, fx in elems]
        return dict(out=out, pics=pics, fixed_ps=fixed_ps, full=full, pages=st["used"], drawings=drawings)

    own = sum(len(sp["text"].strip()) for r in frames for b in r for l in b["lines"] for sp in l["spans"])
    dry = build(mult, False)
    if own and dry["out"]["placed"] < own:
        mult = max(mult * (dry["out"]["placed"] / own) ** 0.5 * C.shrink, C.fit_floor)
    res = build(mult, True)
    if not res["full"]:
        res = _fill_frames(build, mult, res)
    out, pics, fixed_ps, full, elems = res["out"], res["pics"], res["fixed_ps"], res["full"], res["drawings"]
    st = dict(used=res["pages"])
    info = dict(head=typesetter.role["head"], body=typesetter.role["body"], mono=typesetter.role["mono"],
                body_pt=round(body_pt * mult, 1), body_src_pt=body, pages=st["used"], dropped=typesetter.dropped, full=full,
                wrapped=sum(1 for p in fixed_ps if not p["multi"] and len(p["lines"]) > 1),
                unresolved={f"U+{ord(c):04X} {unicodedata.name(c, '?')}": n for c, n in typesetter.unresolved.items()})
    return dict(items=out["items"], glyphs=out["glyphs"], pics=pics, rects=out["rects"], elems=elems, cols=cols_px, g=g,
                size=(width_px, round(ph * g)), info=info)


def sheet(path, number, seed, fonts, body_pt, scale, margin, size):
    """The content of a PDF page restyled on a sheet of size (width, height) px with a blank margin (share of the sheet's
    sides) all around: -> (uint8 sheet, layout info). The same seed gives the same sheet."""
    sw, sh = size
    mx, my = round(margin * sw), round(margin * sh)
    with pymupdf.open(path) as doc:
        box = R.content_box(doc[number])
        for k in range(number, min(number + C.max_pages, len(doc))):
            o = doc[k].cropbox_position
            doc[k].set_cropbox((box + (o.x, o.y, o.x, o.y)) & doc[k].mediabox)
        w = min(sw - 2 * mx, round((sh - 2 * my) * box.width / box.height))
        src = R.Source(doc)
        lay = layout(src, number, np.random.default_rng(seed), fonts, body_pt, scale, w)
        img = R.render(src, lay)
    out = Image.new("L", size, 255)
    out.paste(img, (mx + (sw - 2 * mx - img.width) // 2, my))
    return np.asarray(out), lay["info"]
