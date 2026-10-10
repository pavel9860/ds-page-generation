"""A PDF page restyled on its own template: every box keeps its place, only the text in it is set again.

Body paragraphs of a column (a stretch of them between headings, pictures and gaps) are text frames. Their text, and the
text of the following pages, is one stream poured through the frames in reading order until they are full. Every other
block (heading, table cell, label, header, footer) is a fixed box set at its place. All sizes keep their source ratios to
the body; text a box cannot hold is cut at the box. Rules, frames and pictures stay where the source has them, so nothing can overlap and a column has no gap
while the stream lasts.
"""
import bisect
import itertools
import math
import unicodedata
from collections import defaultdict, deque

import numpy as np
import pymupdf
from PIL import Image

from . import refont as R
from .refont import C, wrap
from .turn import Turn


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
    """Candidates that sit side by side with another block between the same two rules, with no other block above or below
    them there: the cells of a table row (stacked blocks between two rules are columns of text)."""
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

    def stacked(a, b):
        return min(a["bbox"].x1, b["bbox"].x1) > max(a["bbox"].x0, b["bbox"].x0)
    return {id(a) for row in bands.values() for a in row
            if any(b is not a and beside(a, b) for b in row) and not any(b is not a and stacked(a, b) for b in row)}


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


def _walled(cand, blocks, walls, rules, g):
    """Candidates between two vertical rules with a block beyond one of them on their level, or closed by rules above
    and below within two lines of them: cells of a ruled table."""
    out = set()
    ys = [r[2] for r in rules]
    for b in cand:
        x0, x1, mid = b["bbox"].x0 * g, b["bbox"].x1 * g, (b["bbox"].y0 + b["bbox"].y1) / 2 * g
        xs = [x for x, a, z in walls if a <= mid <= z]
        left, right = max((x for x in xs if x <= x0 + 2), default=None), min((x for x in xs if x >= x1 - 2), default=None)
        if left is None or right is None:
            continue
        band = R._band(rules, ys, dict(x0=x0, x1=x1, y0=b["bbox"].y0 * g, y1=b["bbox"].y1 * g, tol=C.row_tol_em * b["size"] * g))
        if (band and band[1] - band[0] <= (b["bbox"].height + 2 * b["src_pitch"]) * g) or any(
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
    cells = _cells(cand, texts, rules, g) | _mates(cand, _columns(cand)) | _walled(cand, texts, walls, rules, g)
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
    box = np.array([tuple(l["bbox"]) for l in lines], float).reshape(-1, 4)
    lh = box[:, 3] - box[:, 1]
    covered, keep = defaultdict(float), []
    for e in elems:
        r, fill = e[0]["rect"], e[0].get("fill")
        host = None
        if fill is not None and R._gray_f(fill) < C.letter_gray and r.get_area() > 0:
            inside = ((np.minimum(box[:, 2], r.x1) - np.maximum(box[:, 0], r.x0)).clip(0) * (np.minimum(box[:, 3], r.y1) - np.maximum(box[:, 1], r.y0)).clip(0)
                      >= C.contain_share * r.width * r.height)
            k = np.flatnonzero((r.height <= C.letter_em * lh) & (r.width <= C.letter_em * lh) & inside)
            host = lines[k[0]] if len(k) else None
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


def _longer(item, room):
    """A filler token (token, font, width, space) made longer by as many of its characters as fit in room."""
    t, f, w, sw = item
    adv = R._advance(f.key, t["text"][0])
    n = int(room // adv) if adv > 0 else 0
    return dict(t, text=t["text"] + t["text"][0] * n), f, w + n * adv, sw


def _set_lines(lines, x0, W, top, m, final, out, kind, fill=None, pinned=False, clip=None):
    """Lines set from `top` in a box at x0, W wide, as the paragraph is aligned; the last line of a paragraph is not stretched.
    clip: (x0, y0, x1, y1) the text is cut to, its box and the room for the letters' tails below it (None: the page). fill: the
    width a filler (a run of dots or underscores) at the end of a line is made to reach (None: it is not)."""
    rtl, pitch, asc, desc = m["rtl"], m["pitch"], m["asc"], m["desc"]
    box = None if clip is None else (clip[0], clip[1], clip[2], clip[3] - C.descent_room * desc)
    y = top + pitch / 2 + (asc - desc) / 2
    for li, ln in enumerate(lines):
        shown = box is None or (y - C.core_share * asc >= box[1] - 0.5 and y <= box[3] + 0.5)
        if li and not shown:
            break
        ind_l = m["ind"][0] if li == 0 else m["ind"][1]
        lw = sum(w for _, _, w, _ in ln) + sum(sw for _, _, _, sw in ln[1:])
        if fill and ln[-1][0].get("leader") and fill - ind_l - lw > 0:
            ln = ln[:-1] + [_longer(ln[-1], fill - ind_l - lw)]
            lw = sum(w for _, _, w, _ in ln) + sum(sw for _, _, _, sw in ln[1:])
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
                out["items"].append((x, y, f, t["text"], t["gray"], "rtl" if t["rtl"] else "ltr", clip))
            x += w + (sw if rtl else 0.0) + extra
        y0, y1 = (y - asc, y + desc) if box is None else (max(y - asc, box[1]), min(y + desc, box[3]))
        if y1 > y0:
            out["rects"].append(dict(x0=start, x1=x - extra, y0=y0, y1=y1, block=len(out["rects"]), src=(x0, x0 + W),
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


def _pour(frames, units, more, obstacles, typesetter, role_of, px_of, g, width, height, out, up, last, cut=0.0):
    """The stream of paragraphs poured through the frames in turn, line by line, around the obstacles (a box or a picture
    that lies in a frame); a paragraph that does not end in a frame goes on in the next. `more` adds the paragraphs of the
    next page (False when there are none). The frames are moved up by `up` px; those of the last section keep their bottom (they take the room the sections above left); the others end
    `cut` px higher; no line goes below `height`. ->
    (whether every frame is full, the lowest line's bottom)."""
    full, low = True, 0.0
    for fr in frames:
        x0, x1 = max(min(b["bbox"].x0 for b in fr) * g, 0.0), min(max(b["bbox"].x1 for b in fr) * g, width)
        y0, y1 = fr[0]["bbox"].y0 * g, fr[-1]["bbox"].y1 * g
        top, bottom = y0 - up, y1 - (0.0 if last else up + cut)
        near = [o for o in obstacles if o[0] < x1 and o[2] > x0 and o[1] < bottom and o[3] > top]
        y, ended = top, False
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
                ended = y + m["pitch"] > min(bottom + C.frame_slack * m["pitch"], height)
                if ended:
                    break
                ax, aw = _free(near, x0, x1, y, y + m["pitch"])
                if aw < C.min_line_em * m["pitch"]:
                    y += m["pitch"]
                    continue
                ind = m["ind"][1] if u["cont"] else m["ind"][0]
                lines = wrap(u["toks"], aw, ind, m["ind"][1])
                y = _set_lines(lines[:1], ax, aw, y, dict(m, ind=(ind, ind)), len(lines) == 1, out, "frame")
                u["toks"], u["cont"] = [t for ln in lines[1:] for t, *_ in ln], True
            if not u["toks"]:
                units.popleft()
        full = full and ended
        low = max(low, bottom if ended else y)
    return full, low


def _balance(frames, units, obstacles, typesetter, role_of, px_of, g, width, page_h, up, height):
    """How much higher the frames of a section can end with all its text still in them: the columns balanced, the section
    as short as its text (bisection over trial pours)."""
    for u in units:
        if u["toks"] is None:
            u["toks"] = R.tokens(u["b"], typesetter, role_of, px_of)
            u["m"] = _metrics(u["b"], u["toks"], px_of, g) if u["toks"] else None

    def fits(cut):
        trial = deque(dict(u) for u in units)
        _pour(frames, trial, lambda: False, obstacles, typesetter, role_of, px_of, g, width, page_h,
              dict(items=[], glyphs=[], rects=[]), up, False, cut)
        return not trial

    if not fits(0.0):
        return 0.0
    lo, hi = 0.0, height
    while hi - lo > C.balance_px:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fits(mid) else (lo, mid)
    return lo


def layout(src, number, rng, fonts, width_px, k=1.0, turn=False):
    """Layout of page `number` of a refont.Source width_px wide, every text at k x its source size (scaled with the page).
    turn: the page turned a quarter (see turn.py): width_px is the width of the turned page, and the text, which keeps its
    size as far as the fixed rows of the page allow, is set again in the turned boxes."""
    sw, sh = src[number].rect.width, src[number].rect.height
    T = Turn(sw, sh) if turn else None
    pw, ph = (sh, sw) if T else (sw, sh)
    g = width_px / pw
    page_h = round(ph * g)
    first = R.page_blocks(src, number, True)
    bottom = max((b["bbox"].y1 for b in first if not b["pinned"]), default=None)
    gfx, found = R.graphics(src, number, True, sh, sw, below=bottom), R.pictures(src, number, sh, sw, below=bottom)
    k_units = k
    if T:
        frames = {id(b) for sec in split_page(first + found, R._hrules(gfx, 1.0), 1.0, R.cell_boxes(gfx, sw * sh), R._vrules(gfx, 1.0))[0]
                  for fr in sec for b in fr}
        first, gfx, found, squeeze = T.apply(first, gfx, found, frames)
        k *= squeeze
    elems = _letters(gfx, first)
    spans = [s for b in first for l in b["lines"] for s in l["spans"]]
    body = R.body_size(first)
    chars = {c for s in spans for c in s["text"] if not c.isspace()}
    need = {(s["bold"], s["italic"]) for s in spans if not s["mono"]}
    cols_px = [(c[0] * g, c[1] * g) for c in R._columns(first)]
    first += found
    sections, fixed = split_page(first, R._hrules(elems, g), g, R.cell_boxes(elems, pw * ph), R._vrules(elems, g))
    role_of = lambda sz: "head" if sz > body * C.head_ratio else "body"
    px_of = lambda sz: sz * g * k
    texts = [b for b in fixed if not b.get("image")]
    walls = R._vrules(elems, g)

    text_blocks = [o for o in first if not o.get("image")]
    boxes = np.array([tuple(o["bbox"]) for o in text_blocks], float).reshape(-1, 4)
    ids = np.array([id(o) for o in text_blocks])

    typesetter = R.Typesetter(fonts, rng, chars, need)
    running = R.running_bands(first)
    later = {}

    out = dict(items=[], glyphs=[], rects=[])
    fixed_ps = [p for p in (R.prepare(b, boxes[ids != id(b)], pw, g, typesetter, role_of, px_of, cols_px, walls) for b in texts) if p]
    pics = [(b["bbox"].x0 * g, b["bbox"].y0 * g, dict(b, scale=1.0)) for b in fixed if b.get("image")]
    st = dict(used=1)
    gap = C.gap_body * px_of(body)
    units = [deque(_units(sec, g, k_units, gap)) for sec in sections]

    def more():
        if st["used"] >= C.max_pages or number + st["used"] >= len(src):
            return False
        nxt = number + st["used"]
        st["used"] += 1
        if nxt not in later:
            blocks = R.page_blocks(src, nxt, False, running) if R.text_page(src, nxt) else []
            gfx = R.graphics(src, nxt, False, sh, sw, number)
            if T:
                blocks, gfx = T.apply(blocks, gfx, [])[:2]
            later[nxt] = split_page(blocks, R._hrules(gfx, g), g, R.cell_boxes(gfx, pw * ph), R._vrules(gfx, g))[0]
        units[-1].extend(_units([r for sec in later[nxt] for r in sec], g, k_units, gap))
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
        cut = 0.0 if last or len(sec) < 2 else _balance(sec, units[n], moved, typesetter, role_of, px_of, g, width_px, page_h, d, bottom - top)
        ok, low = _pour(sec, units[n], more if last else (lambda: False), moved, typesetter, role_of, px_of, g, width_px, page_h, out, d, last, cut)
        full = full and ok
        done.append((bottom, 0.0 if last else max(bottom - d - low, 0.0)))
    for p in fixed_ps:
        d = 0.0 if p["pinned"] else up(p["y0"])
        top, h = p["y0"] - d, p["y1"] - p["y0"]
        _set_lines(p["lines"], p["bx0"], p["W"], top + max(h - len(p["lines"]) * p["pitch"], 0.0) / 2, p, not p["b"].get("flush"), out, "fixed", p["x1"] - p["x0"],
                   p["pinned"], (p["bx0"] - 1, top, p["bx0"] + p["W"] + 1, top + h + C.descent_room * p["desc"]))
    pics = [(x, t if b["pinned"] else t - up(t), b) for x, t, b in pics]
    elems = [(dr, fx, 0.0 if fx else up(dr["rect"].y0 * g)) for dr, fx in elems]
    info = dict(head=typesetter.role["head"], body=typesetter.role["body"], mono=typesetter.role["mono"],
                body_src_pt=body, k=k, pages=st["used"], dropped=typesetter.dropped, full=full,
                wrapped=sum(1 for p in fixed_ps if not p["multi"] and len(p["lines"]) > 1),
                unresolved={f"U+{ord(c):04X} {unicodedata.name(c, '?')}": n for c, n in typesetter.unresolved.items()})
    return dict(items=out["items"], glyphs=out["glyphs"], pics=pics, rects=out["rects"], elems=elems, cols=cols_px, g=g,
                size=(width_px, page_h), info=info)


def _open(path, number, box=None):
    """The PDF with its pages from number on cropped to box (None: the content box of page number) -> doc, box."""
    doc = pymupdf.open(path)
    box = R.content_box(doc[number]) if box is None else box
    for k in range(number, min(number + C.max_pages, len(doc))):
        cb, mb = doc[k].cropbox, doc[k].mediabox      # set_cropbox: x in PDF units, y down from the MediaBox top;
        inside = pymupdf.Rect(mb.x0, 0, mb.x1, mb.height) + (0.01, 0.01, -0.01, -0.01)   # its check is exact
        doc[k].set_cropbox((box + (cb.x0, cb.y0, cb.x0, cb.y0)) & inside)
    return doc, box


def _ink_box(lay):
    """(x0, y0, x1, y1) px of what a layout draws: text lines and items (font size above the baseline, 0.3 of it
    below), pictures, glyph runs and graphics."""
    g = lay["g"]
    boxes = [(r["x0"], r["y0"], r["x1"], r["y1"]) for r in lay["rects"]]
    boxes += [(x, y - f.key[1], x + R.text_width(*f.key, text, direction), y + 0.3 * f.key[1])
              for x, y, f, text, _, direction, _ in lay["items"] if text.strip()]
    boxes += [(x, top, x + b["rect"].width * g, top + b["rect"].height * g) for x, top, b in lay["pics"]]
    boxes += [(x, y - t["px"], x + t["px"], y) for x, y, t in lay["glyphs"]]
    boxes += [(d["rect"].x0 * g, d["rect"].y0 * g - dy, d["rect"].x1 * g, d["rect"].y1 * g - dy) for d, _, dy in lay["elems"]]
    if not boxes:
        return (0.0, 0.0) + tuple(lay["size"])
    b = np.array(boxes, float)
    return (*np.maximum(b[:, :2].min(0), 0), *np.minimum(b[:, 2:].max(0), lay["size"]))


def sheet_layout(path, number, seed, fonts, margin, size, body_px=None, turn=False):
    """The content of a PDF page restyled on a sheet of size (width, height) px with a blank margin (share of the sheet's
    sides) all around, its body at body_px (None: the source size) and its other sizes in the source's ratio to it: ->
    a sheet description (draw_sheet draws it at any scale) with the layout info (body_px) and the ink box (x0, y0, x1, y1)
    px on the sheet. The same seed gives the same sheet. turn: the page turned a quarter (see turn.py): its content is
    set on the sheet turned, its text at the size the source has (less only where the fixed rows of the page leave no room)."""
    sw, sh = size
    mx, my = round(margin * sw), round(margin * sh)
    doc, box = _open(path, number)
    with doc:
        w = min(sw - 2 * mx, round((sh - 2 * my) * box.width / box.height))
        src = R.Source(doc)
        g = w / box.width
        body = R.body_size(R.page_blocks(src, number, True))
        k = body_px / (body * g) if body_px and body else 1.0
        lay = layout(src, number, np.random.default_rng(seed), fonts, round(w * box.height / box.width) if turn else w, k, turn)
    lw, lh = lay["size"]
    x0, y0, x1, y1 = _ink_box(lay)
    if turn:
        lw, lh, (x0, y0, x1, y1) = lh, lw, (lh - y1, x0, lh - y0, x1)
    pos = (mx + (sw - 2 * mx - lw) // 2, my)
    ink = (x0 + pos[0], y0 + pos[1], x1 + pos[0], y1 + pos[1])
    info = dict(lay["info"], body_px=lay["info"]["body_src_pt"] * lay["g"] * lay["info"]["k"])
    return dict(path=path, number=number, box=box, lay=lay, size=size, pos=pos, turn=turn, ink=ink), info


def draw_sheet(desc, s=1.0):
    """uint8 sheet of a sheet_layout description drawn s times its size."""
    sw, sh = desc["size"]
    with _open(desc["path"], desc["number"], desc["box"])[0] as doc:
        img = R.render(R.Source(doc), desc["lay"], s)
    if desc["turn"]:
        img = img.rotate(-90, expand=True)
    out = Image.new("L", (round(sw * s), round(sh * s)), 255)
    out.paste(img, tuple(round(v * s) for v in desc["pos"]))
    return np.asarray(out)


def sheet(path, number, seed, fonts, margin, size, body_px=None, turn=False):
    """sheet_layout drawn at its size: -> (uint8 sheet, layout info with body_px)."""
    desc, info = sheet_layout(path, number, seed, fonts, margin, size, body_px, turn)
    return draw_sheet(desc), info
