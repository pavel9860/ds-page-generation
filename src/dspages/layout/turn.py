"""A page turned a quarter: its content set on the page of the turned shape (the width of the page becomes its height).

The turned page has the area of the source page, so the same text takes the same room; only its shape changes. Every element
keeps its kind of scale:

* text keeps its size (as far as the page allows) and is set again in its boxes: a paragraph frame takes any height, as its
  text only wraps in wider lines; a fixed box (table cell, label, heading) keeps its height, as its text does not wrap;
* pictures and figures keep their shape and size, and sit at the corner of their place;
* tables, rules and boxes are stretched with the page.

The rows of the page are of four kinds: figures, rows of fixed text, rows of frames, and gaps. The turned page is as high
as the source is wide, so the frames and the gaps give the room that figures and fixed rows keep; only when they cannot
(a page of tables) do the fixed rows shrink with their text.
"""
import bisect

import numpy as np
import pymupdf
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .refont import C, _columns, _text, single_align


def _curved(d):
    """Whether a drawing has a curve or a slanted edge: part of a figure (a logo, an icon, a plot line), not of a table or a rule."""
    for it in d["items"]:
        if it[0] == "c":
            return True
        if it[0] == "l" and abs(it[1].x - it[2].x) > 0.5 and abs(it[1].y - it[2].y) > 0.5:
            return True
        if it[0] == "qu" and (abs(it[1].ul.x - it[1].ll.x) > 0.5 or abs(it[1].ul.y - it[1].ur.y) > 0.5):
            return True
    return False


def _touching(rects, t):
    """Boxes of the groups of rectangles that touch each other (within t)."""
    if not rects:
        return []
    r = np.array([tuple(c) for c in rects], float)
    near = ((r[:, None, 0] <= r[None, :, 2] + t) & (r[None, :, 0] <= r[:, None, 2] + t)
            & (r[:, None, 1] <= r[None, :, 3] + t) & (r[None, :, 1] <= r[:, None, 3] + t))
    _, label = connected_components(csr_matrix(near), directed=False)
    return [pymupdf.Rect(*np.r_[r[label == k][:, :2].min(0), r[label == k][:, 2:].max(0)].tolist()) for k in range(label.max() + 1)]


def _inside(box, rect, margin=0.0):
    return rect in box + (-margin, -margin, margin, margin)


class Turn:
    """Maps the geometry of a page of w x h (points) to the turned page of h x w; see the module doc."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.sx = h / w
        self.plan = None
        self.home_of = {}

    def figures(self, elems, pics, blocks):
        """Boxes of the figures of a page and the figure of each drawing (None: none). A figure is a picture or a group of
        drawings with curves or slanted edges, with the rules and drawings that lie in it. Curves under body text (a
        watermark) and boxes that hold text (a callout) are no figures: they are stretched like tables."""
        page, t = self.w * self.h, C.touch_pt
        texts = [b["bbox"] for b in blocks if not b.get("image")]
        body = [b["bbox"] for b in blocks if len(b["lines"]) >= C.paragraph_lines and not b.get("image")]

        def holds_text(c):
            """An outline (nothing filled in it) that text covers: a callout."""
            filled = any(d.get("fill") is not None and _inside(c, d["rect"], t) for d, _ in elems)
            return not filled and sum((c & b).get_area() for b in texts) >= C.container_cover * c.get_area()

        curves = [d["rect"] for d, _ in elems if d["rect"].get_area() < C.background_share * page and _curved(d)]
        curves = [c for c in _touching(curves, t) if not any((c & b).get_area() > 0 for b in body) and not holds_text(c)]
        boxes = _touching(curves + [pymupdf.Rect(p["rect"]) for p in pics], t)
        for d, _ in sorted(elems, key=lambda e: e[0]["rect"].get_area()):
            rect = d["rect"]
            inside = [k for k, b in enumerate(boxes) if _inside(rect, b, t)]
            if len(inside) >= 2 and rect.get_area() < C.background_share * page and not _curved(d):
                boxes = [b for k, b in enumerate(boxes) if k not in inside] + [pymupdf.Rect(rect)]
        merged = True
        while merged and len(boxes) > 1:
            merged = False
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    if (boxes[i] & boxes[j]).get_area() > 0:
                        boxes[i] |= boxes.pop(j)
                        merged = True
                        break
                if merged:
                    break
        home = [None] * len(elems)
        for _ in range(C.figure_rounds):
            for k, box in enumerate(boxes):
                reach = box + (-C.figure_reach_pt, -C.figure_reach_pt, C.figure_reach_pt, C.figure_reach_pt)
                for i, (d, _) in enumerate(elems):
                    rect = d["rect"]
                    if home[i] is not None or rect.get_area() >= C.background_share * page:
                        continue
                    ox, oy = min(rect.x1, reach.x1) - max(rect.x0, reach.x0), min(rect.y1, reach.y1) - max(rect.y0, reach.y0)
                    along = ox / max(rect.width, 1e-3) if rect.width >= rect.height else oy / max(rect.height, 1e-3)
                    if ox >= 0 and oy >= 0 and (rect in reach or (along >= C.figure_share and rect.get_area() < C.cell_share * box.get_area())):
                        home[i] = k
                        boxes[k] |= rect
        return boxes, home

    def make_plan(self, boxes, fixed, frames):
        """The vertical map and the horizontal knots of the page. fixed, frames: rectangles of the fixed text blocks and of
        the frames. -> dict(r: scale of the fixed rows (and their text), bands: the figures' rows with their knots,
        rows: (source start, source end, new start, scale))."""
        bands = []
        for b in sorted(boxes, key=lambda b: b.y0):
            if bands and b.y0 <= bands[-1]["z"]:
                bands[-1]["z"] = max(bands[-1]["z"], b.y1)
                bands[-1]["boxes"].append(b)
            else:
                bands.append(dict(a=b.y0, z=b.y1, boxes=[b]))
        for band in bands:
            width = max(b.width for b in band["boxes"])
            band["scale"] = min(1.0, self.h / max(width, 1e-3))
        edges = sorted({0.0, self.h} | {min(max(v, 0.0), self.h) for r in list(fixed) + list(frames) for v in (r.y0, r.y1)}
                       | {v for band in bands for v in (band["a"], band["z"])})
        rows = []
        for a, z in zip(edges, edges[1:]):
            mid = (a + z) / 2
            band = next((k for k, g in enumerate(bands) if g["a"] <= mid <= g["z"]), None)
            if band is not None:
                kind = "figure"
            elif any(r.y0 <= mid <= r.y1 for r in fixed):
                kind = "fixed"
            elif any(r.y0 <= mid <= r.y1 for r in frames):
                kind = "frame"
            else:
                kind = "gap"
            rows.append([a, z, kind, band])
        size = {kind: sum(z - a for a, z, k, _ in rows if k == kind) for kind in ("fixed", "frame", "gap")}
        figure = sum(bands[band]["scale"] * (z - a) for a, z, k, band in rows if k == "figure")
        free, avail = size["frame"] + size["gap"], self.w
        fit = C.frame_give / self.sx  # the scale of a frame that keeps (nearly) its text: the lines are as much wider as they are fewer
        r, soft = 1.0, (avail - figure - size["fixed"]) / free if free > 1e-6 else fit
        if soft < fit:
            soft = fit
            r = (avail - figure - soft * free) / size["fixed"] if size["fixed"] > 1e-6 else 1.0
            if r < C.turn_floor:
                r = C.turn_floor
                soft = max((avail - figure - r * size["fixed"]) / free, C.turn_gap_min) if free > 1e-6 else fit
        soft = min(soft, C.turn_gap_max)
        over = (r * size["fixed"] + soft * free + figure) - avail
        if over > 1e-6 and figure > 1e-6:      # the minimum scales of the rest still leave the page too tall: the figures shrink, evenly
            f = max((avail - r * size["fixed"] - soft * free) / figure, C.figure_floor)
            for band in bands:
                band["scale"] *= f
            figure *= f
            over = (r * size["fixed"] + soft * free + figure) - avail
        if over > 1e-6:        # still too tall: give back evenly what the floors kept (never beyond the page)
            f = (avail - figure) / (r * size["fixed"] + soft * free)
            r, soft = r * f, soft * f
        out, new = [], 0.0
        for a, z, kind, band in rows:
            scale = {"figure": bands[band]["scale"] if band is not None else 1.0, "fixed": r}.get(kind, soft)
            out.append((a, z, new, scale))
            new += scale * (z - a)
        return dict(r=r, bands=bands, boxes=list(boxes), rows=out, starts=[a for a, *_ in out])

    def y_of(self, y):
        rows, starts = self.plan["rows"], self.plan["starts"]
        k = min(max(bisect.bisect_right(starts, y) - 1, 0), len(rows) - 1)
        a, z, new, scale = rows[k]
        return new + scale * (y - a)

    def band_at(self, y):
        return next((b for b in self.plan["bands"] if b["a"] <= y <= b["z"]), None)

    def apply(self, blocks, elems, pics, frames=None):
        """The text blocks, drawings (drawing, fixed) and picture blocks of the first page in their turned places (frames:
        the ids of the blocks that are frames); with frames None, those of a later page in the places the first page gave.
        -> blocks, drawings, pictures and the scale of the fixed rows (the scale of the text)."""
        if frames is not None:
            boxes, home = self.figures(elems, pics, blocks)
            fixed = [b["bbox"] for b in blocks + pics if id(b) not in frames]
            self.plan = self.make_plan(boxes, fixed, [b["bbox"] for b in blocks if id(b) in frames])
            self.home_of = {id(d): boxes[k] for (d, _), k in zip(elems, home) if k is not None}
        home_of = self.home_of if frames is not None else {}

        def mapper(r, home=None):
            """x and y maps of an element: a figure and what lies in it keep their scale and sit in the stretched footprint of
            their box (left stays left, right stays right); everything else is stretched with the page."""
            c = ((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)
            box = home if home is not None else next((b for b in self.plan["boxes"] if b.x0 <= c[0] <= b.x1 and b.y0 <= c[1] <= b.y1), None)
            if box is None:
                return (lambda x: x * self.sx), self.y_of
            band = self.band_at(c[1])
            scale = band["scale"] if band else 1.0
            spare = self.w - box.width
            frac = box.x0 / spare if spare > 1e-3 else 0.0
            x0 = box.x0 * self.sx + frac * box.width * (self.sx - scale)
            return (lambda x: x0 + scale * (x - box.x0)), self.y_of

        rect = lambda r, m: pymupdf.Rect(m[0](r.x0), m[1](r.y0), m[0](r.x1), m[1](r.y1))
        pt = lambda p, m: pymupdf.Point(m[0](p.x), m[1](p.y))

        def drawing(d):
            m = mapper(d["rect"], home_of.get(id(d)))
            items = []
            for it in d["items"]:
                if it[0] == "re":
                    items.append(("re", rect(it[1], m), *it[2:]))
                elif it[0] == "qu":
                    q = it[1]
                    items.append(("qu", pymupdf.Quad(*(pt(p, m) for p in (q.ul, q.ur, q.ll, q.lr)))))
                else:
                    items.append((it[0], *(pt(p, m) for p in it[1:])))
            return dict(d, items=items, rect=rect(d["rect"], m))

        cols = _columns(blocks)
        for b in blocks:
            c = b["bbox"]
            m = mapper(c)
            cs = b.get("colspan")
            span = cs or next(((a, z) for a, z in cols if a - 2 <= c.x0 and c.x1 <= z + 2), (min(a for a, _ in cols), max(z for _, z in cols)))
            if len(b["lines"]) == 1 and not b.get("image"):
                c0, c1 = span
                b["src_align"] = single_align(c.x0, c.x1, c0, c1, b["rtl"])
            b["flush"] = (len(b["lines"]) == 1 and len(_text(b["lines"][0]).split()) >= C.flush_words
                          and c.width >= C.full_line * (span[1] - span[0]) and b["align"] == "left")
            b["bbox"] = rect(c, m)
            if cs:
                b["colspan"] = (m[0](cs[0]), m[0](cs[1]))
            for l in b["lines"]:
                l["bbox"] = rect(l["bbox"], m)
                for sp in l["spans"]:
                    sp["bbox"] = rect(sp["bbox"], m)
        placed = [dict(p, rect=rect(p["rect"], mapper(p["rect"])), bbox=rect(p["rect"], mapper(p["rect"]))) for p in pics]
        return blocks, [(drawing(d), fixed) for d, fixed in elems], placed, self.plan["r"]
