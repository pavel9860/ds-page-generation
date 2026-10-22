"""Page structure -> items for page.make: the blocks of a page (reading order, positions, sizes) become headings, paragraphs and list items
with the blank lines of the original, and blocks that stand side by side become columns.

to_items(blocks) -> [item]; block = (kind, text, (x0, y0, x1, y1), size) in any units, kind h | p | li | v.
item = {k: h|p|li|v, t, gap, scale} | {k: cols, cols: [{x, w, start, items}], gap}   (x, w: shares of the text width)
"""

WIDE = 0.62            # a block wider than this share of the page is a full-width block
COL_GAP = 0.10         # blocks whose left edges differ by more than this share of the page are in different columns
MAX_COLS = 4


def _gap(b, prev_y1, line):
    if prev_y1 is None:
        return 0
    g = (b[2][1] - prev_y1) / line
    return 0 if g < 0.7 else 1 if g < 2.5 else 2 if g < 5 else 3


def _item(b, prev_y1, body, line):
    k, t, _, size = b
    d = dict(k=k, t=t, gap=_gap(b, prev_y1, line))
    if k == "h":
        d["scale"] = float(min(max(size / body, 1.0), 1.25)) if len(t) <= 70 else 1.0
    return d


def _flow(blocks, body, line):
    out, prev = [], None
    for b in sorted(blocks, key=lambda b: (b[2][1], b[2][0])):
        out.append(_item(b, prev, body, line))
        prev = b[2][3]
    return out


def _columns(blocks, span, x0, body, line):
    """Blocks of one band grouped by left edge; None when they do not form at least two columns."""
    cols = []
    for b in sorted(blocks, key=lambda b: b[2][0]):
        if cols and b[2][0] - cols[-1]["x0"] <= COL_GAP * span:
            cols[-1]["blocks"].append(b)
        else:
            cols.append(dict(x0=b[2][0], blocks=[b]))
    cols = [c for c in cols if sum(len(b[1]) for b in c["blocks"]) >= 60]
    if len(cols) < 2:
        return None
    cols = sorted(cols, key=lambda c: -sum(len(b[1]) for b in c["blocks"]))[:MAX_COLS]
    cols.sort(key=lambda c: c["x0"])
    top = min(b[2][1] for c in cols for b in c["blocks"])
    res = []
    for i, c in enumerate(cols):
        bs = sorted(c["blocks"], key=lambda b: b[2][1])
        left = min(b[2][0] for b in bs)
        right = cols[i + 1]["x0"] if i + 1 < len(cols) else x0 + span
        items = _flow(bs, body, line)
        items[0]["gap"] = 0
        res.append(dict(x=(left - x0) / span, w=max((min(right, max(b[2][2] for b in bs) + 0.02 * span) - left) / span - 0.02, 0.15),
                        start=int(min(max((bs[0][2][1] - top) / line * 0.5, 0), 6)), items=items))
    return res


def to_items(blocks):
    if not blocks:
        return []
    paras = sorted(b[3] for b in blocks if b[0] == "p") or sorted(b[3] for b in blocks)
    body = max(paras[len(paras) // 2], 1e-6)
    line = body * 1.25
    x0, x1 = min(b[2][0] for b in blocks), max(b[2][2] for b in blocks)
    span = max(x1 - x0, 1e-6)
    out, narrow, prev = [], [], None
    for b in sorted(blocks, key=lambda b: (b[2][1], b[2][0])):
        if (b[2][2] - b[2][0]) > WIDE * span:                      # a full-width block closes the band of columns above it
            out += _band(narrow, span, x0, body, line, prev)
            narrow = []
            out.append(_item(b, prev, body, line))
            prev = b[2][3]
        else:
            narrow.append(b)
    out += _band(narrow, span, x0, body, line, prev)
    return out


def _band(narrow, span, x0, body, line, prev):
    if not narrow:
        return []
    cols = _columns(narrow, span, x0, body, line)
    gap = _gap(narrow[0], prev, line)
    if cols is None:
        items = _flow(narrow, body, line)
        items[0]["gap"] = gap
        return items
    return [dict(k="cols", cols=cols, gap=gap)]
