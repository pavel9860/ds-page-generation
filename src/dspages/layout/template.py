"""A PDF page restyled on its own template: every box keeps its place, only the text in it is set again.

Body paragraphs of a column (a stretch of them between headings, pictures and gaps) are text frames. Their text, and the
text of the following pages, is one stream poured through the frames in reading order until they are full. Every other
block (heading, table cell, label, header, footer) is a fixed box set at its place; a page-wide shrink keeps its text
inside it. Rules, frames and pictures stay where the source has them, so nothing can overlap and a column has no gap
while the stream lasts.
"""
import bisect
import unicodedata
from collections import Counter, defaultdict, deque

import numpy as np
import pymupdf
from PIL import Image

from . import refont as R
from .refont import C, wrap


def _is_body(b, body):
    return body / C.head_ratio < b["size"] <= body * C.head_ratio


def _cells(cand, rules, g):
    """Candidates that sit side by side between the same two rules: the cells of a table row."""
    ys = [r[2] for r in rules]
    bands = defaultdict(list)
    for b in cand:
        bb = b["bbox"]
        p = dict(x0=bb.x0 * g, x1=bb.x1 * g, y0=bb.y0 * g, y1=bb.y1 * g, tol=C.row_tol_em * b["size"] * g)
        band = R._band(rules, ys, p)
        if band:
            bands[(round(band[0] / C.table_step), round(band[1] / C.table_step))].append(b)
    return {id(b) for row in bands.values() if len(row) > 1 for b in row}


def _runs(cand, others):
    """Candidates of a column in runs: a run ends at a gap of frame_gap lines or at a block between two paragraphs."""
    runs = []
    for b in sorted(cand, key=lambda b: (b["colspan"], b["bbox"].y0)):
        prev = runs[-1][-1] if runs else None
        pitch = b["src_pitch"]
        split = (prev is None or prev["colspan"] != b["colspan"] or b["bbox"].y0 - prev["bbox"].y1 > C.frame_gap * pitch
                 or any(min(o["bbox"].x1, b["bbox"].x1) > max(o["bbox"].x0, b["bbox"].x0) and prev["bbox"].y1 - 1 <= o["bbox"].y0
                        and o["bbox"].y1 <= b["bbox"].y0 + 1 for o in others))
        if split:
            runs.append([])
        runs[-1].append(b)
    return runs


def split_page(blocks, body, rules, g):
    """-> (frames, fixed): the frames (blocks of body paragraphs of a column, in reading order) and the other blocks."""
    cand = [b for b in blocks if not b["pinned"] and not b.get("image") and b["colspan"] is not None and _is_body(b, body)]
    cells = _cells(cand, rules, g)
    cand = [b for b in cand if id(b) not in cells]
    taken = {id(b) for b in cand}
    others = [b for b in blocks if id(b) not in taken and not b["pinned"]]
    runs = [r for r in _runs(cand, others) if sum(len(b["lines"]) for b in r) >= C.frame_lines and max(len(b["lines"]) for b in r) >= C.paragraph_lines]
    framed = {id(b) for r in runs for b in r}
    cols = {b["colspan"] for b in cand}
    breaks = sorted(b["bbox"].y1 for b in others if sum(min(b["bbox"].x1, c[1]) - max(b["bbox"].x0, c[0]) > 0 for c in cols) > 1)
    rtl = sum(b["rtl"] for b in cand) > len(cand) / 2
    runs.sort(key=lambda r: (bisect.bisect_right(breaks, r[0]["bbox"].y0 + 1), -r[0]["bbox"].x0 if rtl else r[0]["bbox"].x0))
    return runs, [b for b in blocks if id(b) not in framed]


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
                out["used"].append((t["sig"], t["path"], t["wght"], round(t["px"], 1)))
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
    """The widest stretch (x, width) of x0..x1 that no obstacle covers between top and bottom."""
    free = [(x0, x1)]
    for ox0, oy0, ox1, oy1 in obstacles:
        if oy0 < bottom and oy1 > top:
            free = [part for a, b in free for part in ((a, min(b, ox0)), (max(a, ox1), b)) if part[1] > part[0]]
    a, b = max(free, key=lambda f: f[1] - f[0], default=(x0, x0))
    return a, b - a


def _pour(frames, units, more, obstacles, typesetter, role_of, px_of, g, width, out):
    """The stream of paragraphs poured through the frames in turn, line by line, around the obstacles (a box or a picture
    that lies in a frame); a paragraph that does not end in a frame goes on in the next. `more` adds the paragraphs of the
    next page (False when there are none). -> whether every frame is full."""
    full = True
    for fr in frames:
        x0, x1 = max(min(b["bbox"].x0 for b in fr) * g, 0.0), min(max(b["bbox"].x1 for b in fr) * g, width)
        top, bottom = fr[0]["bbox"].y0 * g, fr[-1]["bbox"].y1 * g
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
                u["toks"], u["cont"] = [t for ln in lines[1:] for t, *_ in ln], True
            if not u["toks"]:
                units.popleft()
        full = full and ended
    return full


def layout(doc, number, rng, fonts, body_pt, scale, width_px):
    """Layout of a page with its body text at body_pt on the sheet (210 mm wide, width_px wide), the other sizes keeping
    their ratio to the body's times `scale`."""
    page = doc[number]
    pw, ph = page.rect.width, page.rect.height
    g = width_px / pw
    first = R.page_blocks(page, 0.0, True)
    spans = [s for b in first for l in b["lines"] for s in l["spans"]]
    body = Counter({k: sum(len(s["text"]) for s in spans if round(s["size"] * 2) / 2 == k) for k in {round(s["size"] * 2) / 2 for s in spans}}).most_common(1)[0][0]
    sc_font = body_pt * (width_px / C.sheet_pt_w) / body
    chars = {c for s in spans for c in s["text"] if not c.isspace()}
    need = {(s["bold"], s["italic"]) for s in spans if not s["mono"]}
    cols_px = [(c[0] * g, c[1] * g) for c in R._columns(first)]
    elems = R.graphics(doc, number, 0.0, True, ph, pw, below=max((b["bbox"].y1 for b in first if not b["pinned"]), default=None))
    first += R.pictures(doc, number, 0.0, True, ph, pw, below=max((b["bbox"].y1 for b in first if not b["pinned"]), default=None))
    frames, fixed = split_page(first, body, R._hrules(elems, g), g)
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
    px = lambda sz: px_of(sz) * mult
    out = dict(items=[], glyphs=[], rects=[], used=[])
    fixed_ps = prepare_fixed(typesetter, px)
    for p in fixed_ps:
        _set_lines(p["lines"], p["bx0"], p["W"], p["y0"] + (p["y1"] - p["y0"] - len(p["lines"]) * p["pitch"]) / 2, p, True, out, "fixed", p["pinned"])
    pics = [(b["bbox"].x0 * g, b["bbox"].y0 * g, dict(b, scale=1.0)) for b in fixed if b.get("image")]

    st = dict(used=1)
    k = px(body) / (body * g)
    units = deque(_units(frames, g, k, C.gap_body * px(body)))

    def more():
        if st["used"] >= C.max_pages or number + st["used"] >= len(doc):
            return False
        nxt = doc[number + st["used"]]
        st["used"] += 1
        blocks = R.page_blocks(nxt, 0.0, False, running) if R.text_page(nxt) else []
        found, _ = split_page(blocks, body, R._hrules(R.graphics(doc, nxt.number, 0.0, False, ph, pw, number), g), g)
        units.extend(_units(found, g, k, C.gap_body * px(body)))
        return True

    running = R.running_bands(first)
    obstacles = [(p["bx0"], p["y0"], p["bx0"] + p["W"], p["y1"]) for p in fixed_ps if not p["pinned"]]
    obstacles += [(x, t, x + b["rect"].width * g, t + b["rect"].height * g) for x, t, b in pics]
    full = _pour(frames, units, more, obstacles, typesetter, role_of, px, g, width_px, out)
    info = dict(head=typesetter.role["head"], body=typesetter.role["body"], mono=typesetter.role["mono"],
                body_pt=round(body_pt * mult, 1), body_src_pt=body, pages=st["used"], dropped=typesetter.dropped, full=full,
                wrapped=sum(1 for p in fixed_ps if not p["multi"] and len(p["lines"]) > 1),
                unresolved={f"U+{ord(c):04X} {unicodedata.name(c, '?')}": n for c, n in typesetter.unresolved.items()})
    spans_, roots = R.touching_ranges(elems, g)
    return dict(items=out["items"], glyphs=out["glyphs"], pics=pics, tokens=out["used"], anchors=[], rects=out["rects"], elems=elems,
                spans=spans_, roots=roots, cols=cols_px, limit=ph * g, sc=1.0, g=g, cutoff=float("inf"), size=(width_px, round(ph * g)),
                info=info, slack=0.0,
                role_paths={f[0] for k_ in typesetter.role.values() for f in (typesetter.prose.get(k_) or typesetter.mono[k_]).values()})


def sheet(path, number, seed, fonts, body_pt, scale, margin, size):
    """The content of a PDF page restyled on a sheet of size (width, height) px with a blank margin (share of the sheet's
    sides) all around: -> (uint8 sheet, layout info). The same seed gives the same sheet."""
    sw, sh = size
    mx, my = round(margin * sw), round(margin * sh)
    with pymupdf.open(path) as doc:
        box = R.content_box(doc[number])
        for k in range(number, min(number + C.max_pages, len(doc))):
            doc[k].set_cropbox(box & doc[k].mediabox)
        w = min(sw - 2 * mx, round((sh - 2 * my) * box.width / box.height))
        lay = layout(doc, number, np.random.default_rng(seed), fonts, body_pt, scale, w)
        img = R.render(doc, lay)
    out = Image.new("L", size, 255)
    out.paste(img, (mx + (sw - 2 * mx - img.width) // 2, my))
    return np.asarray(out), lay["info"]
