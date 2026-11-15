"""Page structure -> items for page.make: the blocks of a page (reading order, positions, sizes) become headings, paragraphs and list items
with the blank lines of the original, and blocks that stand side by side become columns (recursive XY-cut: a column may hold columns and full-width rows of its own).

to_items(blocks) -> [item]; block = (kind, text, (x0, y0, x1, y1), size[, {align, lvl}]) in any units, kind h | p | li | l | v.
item = {k: h|p|li|l|v, t, gap, scale, align, lvl} | {k: cols, cols: [{x, w, start, items}], gap}   (x, w: shares of the text width;
align center | right when the block is set so; lvl: indent level of a list item)
"""

from .config import MAX_COLS  # noqa: F401


def _gap(b, prev_y1, line):
    if prev_y1 is None:
        return 0
    g = (b[2][1] - prev_y1) / line
    return 0 if g < 0.7 else 1 if g < 2.5 else 2 if g < 5 else 3


def _align(box, x0, span):
    """center / right for a block set so inside the text width [x0, x0 + span], None for a left-aligned or full-width one."""
    w = box[2] - box[0]
    if w > 0.75 * span or box[0] - x0 < 0.08 * span:
        return None
    if abs((box[0] + box[2]) / 2 - (x0 + span / 2)) < 0.05 * span:
        return "center"
    if abs(box[2] - (x0 + span)) < 0.04 * span and box[0] - x0 > 0.3 * span:
        return "right"
    return None


def _item(b, prev_y1, body, line, x0=None, span=None):
    k, t, box, size = b[:4]
    if k == "table":
        return dict(k="table", rows=b[4]["rows"], gap=_gap(b, prev_y1, line))
    d = dict(k=k, t=t, gap=_gap(b, prev_y1, line))
    if k == "h":
        d["scale"] = float(min(max(size / body, 1.0), 1.25)) if len(t) <= 70 else 1.0
    if len(b) > 4:  # the reader's own alignment and list level
        if b[4].get("lvl"):
            d["lvl"] = int(b[4]["lvl"])
        if b[4].get("align"):
            d["align"] = b[4]["align"]
    elif span and k in ("h", "l", "v", "p") and (k != "p" or len(t) <= 160):
        a = _align(box, x0, span)  # OCR blocks: alignment from the position
        if a:
            d["align"] = a
    return d


def _flow(blocks, body, line, x0=None, span=None):
    out, prev = [], None
    for b in sorted(blocks, key=lambda b: (b[2][1], b[2][0])):
        out.append(_item(b, prev, body, line, x0, span))
        prev = b[2][3]
    return out


def _cut(blocks, axis):
    """Blocks split at the whitespace gaps of their projection on axis (0: x, 1: y): groups in order, no two overlapping on it."""
    out, end = [], None
    for b in sorted(blocks, key=lambda b: b[2][axis]):
        if out and b[2][axis] < end:
            out[-1].append(b)
            end = max(end, b[2][axis + 2])
        else:
            out.append([b])
            end = b[2][axis + 2]
    return out


def _columns(blocks):
    """The vertical-gap groups of the blocks when they are columns: 2..MAX_COLS of them and side by side (two overlapping in height), else None."""
    g = _cut(blocks, 0)
    ys = [(min(b[2][1] for b in c), max(b[2][3] for b in c)) for c in g]
    side = any(min(a[1], b[1]) > max(a[0], b[0]) for i, a in enumerate(ys) for b in ys[i + 1 :])
    return g if 2 <= len(g) <= MAX_COLS and side else None


def _xy(blocks, x0, span, body, line, prev):
    """Recursive XY-cut (Nagy & Seth 1984): blocks side by side (a vertical gap through all of them) become columns, each cut again inside its
    own width; else blocks one above the other (a horizontal gap) are cut in turn, the rows a vertical gap divides together kept as one band; a
    group no gap divides is one flow. Columns never overlap."""
    cols = _columns(blocks)
    if cols:
        top = min(b[2][1] for b in blocks)
        lefts = [min(b[2][0] for b in c) for c in cols] + [x0 + span]
        res = []
        for i, c in enumerate(cols):
            items = _xy(c, lefts[i], max(b[2][2] for b in c) - lefts[i], body, line, None)
            items[0]["gap"] = 0
            ext, room = (max(b[2][2] for b in c) - lefts[i]) / span, (lefts[i + 1] - lefts[i]) / span - 0.02
            res.append(
                dict(
                    x=(lefts[i] - x0) / span,
                    w=min(max(ext, 0.15), room),
                    start=int(min(max((min(b[2][1] for b in c) - top) / line * 0.5, 0), 6)),
                    items=items,
                )
            )
        return [dict(k="cols", cols=res, gap=_gap(min(blocks, key=lambda b: b[2][1]), prev, line))]
    rows = []
    for r in _cut(blocks, 1):  # rows one above the other that a vertical gap still divides together: one band
        if rows and _columns(rows[-1] + r):
            rows[-1] = rows[-1] + r
        else:
            rows.append(r)
    if len(rows) < 2:
        items = _flow(blocks, body, line, x0, span)
        items[0]["gap"] = _gap(min(blocks, key=lambda b: b[2][1]), prev, line)
        return items
    out = []
    for r in rows:
        out += _xy(r, x0, span, body, line, prev)
        prev = max(b[2][3] for b in r)  # the next gap is measured from the bottom of the group above
    return out


def to_items(blocks):
    if not blocks:
        return []
    paras = sorted(b[3] for b in blocks if b[0] == "p") or sorted(b[3] for b in blocks)
    body = max(paras[len(paras) // 2], 1e-6)
    x0, x1 = min(b[2][0] for b in blocks), max(b[2][2] for b in blocks)
    return _xy(blocks, x0, max(x1 - x0, 1e-6), body, body * 1.25, None)
