"""Page kind from the text layer of a digital PDF page: plain text, poem or formatted (headings, lists, columns, tables, mixed fonts), from
the geometry and typography of its lines only (no rendering, no visual model). About 2-5 ms a page.

features(page) -> np.ndarray (len(NAMES));  kind(page) -> ("plain" | "poems" | "formatted", probabilities)
The model is a logistic regression on the standardised features (coefficients below, fitted by tools/pagekind_eval.py on pagetype_labelled).
"""
import re
from collections import Counter

import numpy as np
import pymupdf

KINDS = ("formatted", "plain", "poems", "tables", "forms_blank", "forms_filled", "bureaucracy")
BULLET = re.compile(r"^\s*(?:[•▪●◦‣⁃■□✓✔\-–—*·➢►]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)]|[IVX]{1,4}\.)(?:\s|$)")
END = re.compile(r"[.!?。…।۔؟։\"”»)]\s*$")
NAMES = ("lines", "full", "short", "very_short", "right_flush", "indent_levels", "main_left", "centered", "size_mix", "sizes", "big", "small",
         "bold", "italic", "fonts", "bullets", "gap_med", "gap_big", "gap_cv", "stanza", "end_punct", "hyphen", "upper_start", "lower_start",
         "columns", "drawings", "images", "digits", "colon_end", "len_cv", "width_share", "hrules", "vrules", "rects", "grid", "segments",
         "multi_seg", "aligned_cols", "leaders", "boxes", "numeric_cells", "text_cover", "no_text", "num_end", "bullet_glyph",
         "w10", "w25", "w50", "w75", "g25", "g75", "g90", "cap_after_open", "caps_lines", "bold_lines", "big_lines", "word_len", "chars_line",
         "ragged", "sentences_line", "punct_inside")
READER = ("r_h", "r_p", "r_li", "r_l", "r_table", "r_blocks", "r_lines_p", "r_center", "r_right", "r_justify", "r_cols", "r_tables",
          "r_one_line", "r_max_lines")
LEADER = re.compile(r"[_.\u2026]{4,}|_{2,}")
BOX = re.compile(r"[\u2610\u2611\u2612\u25a1\u25a0\u274f\uf06f\uf0a8\uf0fe]|\[\s?\]")
NUM = re.compile(r"^[\s$€£¥%+\-–(),.:/0-9]+$")


def _lines(page):
    """Text lines of the page: (x0, y0, x1, y1, text, [(size, flags, font, chars)]), page furniture (running heads, page numbers) left out."""
    d = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_PRESERVE_IMAGES)
    h = page.rect.height or 1.0
    out = []
    for b in d["blocks"]:
        for ln in b.get("lines", ()):
            if ln["dir"][0] < 0.9:                                  # rotated lines: margins, stamps
                continue
            spans = [s for s in ln["spans"] if s["text"].strip()]
            if not spans:
                continue
            t = "".join(s["text"] for s in ln["spans"]).strip()
            x0, y0, x1, y1 = ln["bbox"]
            if (y1 < 0.07 * h or y0 > 0.93 * h) and len(t) < 60:     # running head / foot, page number
                continue
            out.append((x0, y0, x1, y1, t, [(s["size"], s["flags"], s["font"].split("+")[-1].split("-")[0].lower(), len(s["text"].strip()))
                                             for s in spans]))
    out.sort(key=lambda r: (r[1] + r[3]) / 2)
    rows, band = [], []                                             # pieces of one text line that the PDF split (bullet and text, cells)
    for r in out + [None]:
        if band and (r is None or (r[1] + r[3]) / 2 - (band[0][1] + band[0][3]) / 2 > 0.5 * (band[0][3] - band[0][1])):
            band.sort(key=lambda q: q[0])
            cur = band[0]
            for q in band[1:]:
                if q[0] - cur[2] < 3 * (q[3] - q[1]):
                    cur = (cur[0], min(cur[1], q[1]), max(cur[2], q[2]), max(cur[3], q[3]), cur[4] + " " + q[4], cur[5] + q[5])
                else:
                    rows.append(cur)
                    cur = q
            rows.append(cur)
            band = []
        if r is not None:
            band.append(r)
    return rows, out


def _clusters(xs, tol):
    """Sizes of the groups of nearly equal values (sorted, a new group where the step exceeds tol)."""
    if not len(xs):
        return []
    xs = np.sort(xs)
    cut = np.flatnonzero(np.diff(xs) > tol) + 1
    return [len(g) for g in np.split(xs, cut)]


def features(page):
    rows, raw = _lines(page)
    f = dict.fromkeys(NAMES, 0.0)
    pw = page.rect.width or 1.0
    ph = page.rect.height or 1.0
    try:
        dr = page.get_drawings()
        f["drawings"] = min(len(dr), 400) / 100.0
        f["images"] = min(len(page.get_images()), 10)
        hs, vs, rc = [], [], 0
        for d in dr:
            for it in d["items"]:
                if it[0] == "l":
                    (ax, ay), (bx, by) = it[1], it[2]
                    if abs(ay - by) < 1.5 and abs(ax - bx) > 0.03 * pw:
                        hs.append(round(ay))
                    elif abs(ax - bx) < 1.5 and abs(ay - by) > 0.01 * ph:
                        vs.append(round(ax))
                elif it[0] == "re":
                    r = it[1]
                    if r.width > 0.03 * pw and r.height < 2.5:
                        hs.append(round(r.y0))
                    elif r.height > 0.01 * ph and r.width < 2.5:
                        vs.append(round(r.x0))
                    elif r.width > 0.02 * pw and r.height > 0.008 * ph and r.width < 0.95 * pw:
                        rc += 1
        f["hrules"] = min(len(set(hs)), 120) / 30.0
        f["vrules"] = min(len(set(vs)), 120) / 30.0
        f["rects"] = min(rc, 300) / 50.0
        f["grid"] = min(len(set(hs)), 60) * min(len(set(vs)), 60) / 400.0
    except Exception as e:                                              # the drawing features stay 0
        print("page drawings unreadable:", repr(e)[:120], flush=True)
    f["no_text"] = float(len(rows) < 2)
    if len(rows) < 2:
        f["lines"] = len(rows) / 40.0
        return np.array([f[k] for k in NAMES], np.float32)
    a = np.array([r[:4] for r in rows], float)
    x0, y0, x1, y1 = a.T
    n = len(rows)
    chars = Counter()
    for r in rows:
        for sz, fl, fn, c in r[5]:
            chars[round(sz * 2) / 2] += c
    body = chars.most_common(1)[0][0]
    tot = sum(chars.values()) or 1
    lh = np.median(y1 - y0) or 1.0

    # columns: groups of left edges far apart that each hold many lines side by side
    lefts = _clusters(x0, 0.04 * pw)
    big_left = [g for g in lefts if g >= max(3, 0.12 * n)]
    f["columns"] = max(len(big_left), 1)
    left, right = np.percentile(x0, 10), np.percentile(x1, 90)
    W = max(right - left, 1.0) / f["columns"]
    wid = x1 - x0
    f["lines"] = n / 40.0
    f["width_share"] = (right - left) / pw
    f["full"] = np.mean(wid >= 0.85 * W)
    f["short"] = np.mean(wid < 0.6 * W)
    f["very_short"] = np.mean(wid < 0.3 * W)
    f["right_flush"] = np.mean(np.abs(x1 - right) < 0.02 * pw)
    f["indent_levels"] = len([g for g in lefts if g >= 2])
    f["main_left"] = max(lefts) / n
    mid = (x0 + x1) / 2
    f["centered"] = np.mean((np.abs(mid - (left + right) / 2) < 0.03 * pw) & (wid < 0.8 * W) & (x0 > left + 0.05 * pw))
    f["len_cv"] = float(np.std(wid) / (np.mean(wid) + 1e-6))

    sizes = [s for s, c in chars.items() if c >= 0.02 * tot]
    f["sizes"] = len(sizes)
    f["big"] = sum(c for s, c in chars.items() if s > 1.15 * body) / tot
    f["small"] = sum(c for s, c in chars.items() if s < 0.87 * body) / tot
    f["size_mix"] = f["big"] + f["small"]
    sp = [s for r in rows for s in r[5]]
    f["bold"] = sum(c for _, fl, fn, c in sp if fl & 16 or "bold" in fn) / tot
    f["italic"] = sum(c for _, fl, fn, c in sp if fl & 2 or "italic" in fn) / tot
    fonts = Counter()
    for _, _, fn, c in sp:
        fonts[fn] += c
    f["fonts"] = sum(1 for c in fonts.values() if c >= 0.03 * tot)

    texts = [r[4] for r in rows]
    f["bullets"] = np.mean([bool(BULLET.match(t)) for t in texts])
    f["end_punct"] = np.mean([bool(END.search(t)) for t in texts])
    f["hyphen"] = np.mean([t.endswith(("-", "­")) for t in texts])
    f["colon_end"] = np.mean([t.endswith(":") for t in texts])
    first = [next((c for c in t if c.isalpha()), "") for t in texts]
    f["upper_start"] = np.mean([c.isupper() for c in first])
    f["lower_start"] = np.mean([c.islower() for c in first])
    allc = "".join(texts)
    base = Counter(round((r[1] + r[3]) / 4) for r in raw)           # separated text pieces on one baseline: table cells, form fields
    f["segments"] = len(raw) / max(len(base), 1)
    f["multi_seg"] = np.mean([c >= 3 for c in base.values()])
    xs = np.array([r[0] for r in raw])
    f["aligned_cols"] = len([g for g in _clusters(xs, 0.01 * pw) if g >= max(4, 0.08 * len(raw))])
    f["numeric_cells"] = np.mean([bool(NUM.match(r[4])) for r in raw])
    f["num_end"] = np.mean([bool(re.search(r"\d\s*$", t)) for t in texts])
    f["bullet_glyph"] = np.mean([r[4].strip() in "\u2022\u25aa\u25cf\u25e6\u2023\u2043\u25a0\u25a1\u27a2\u25ba\u2013-*\u00b7\uf0b7\uf0a7\uf0d8" for r in raw])
    f["leaders"] = np.mean([bool(LEADER.search(t)) for t in texts])
    f["boxes"] = np.mean([bool(BOX.search(t)) for t in texts])
    f["text_cover"] = float(np.sum((x1 - x0) * (y1 - y0)) / (pw * ph))
    f["digits"] = sum(c.isdigit() for c in allc) / max(len(allc), 1)
    rel = wid / W
    f["w10"], f["w25"], f["w50"], f["w75"] = np.percentile(rel, [10, 25, 50, 75])
    ends = [bool(END.search(t) or t.endswith((",", ";", ":"))) for t in texts]
    f["cap_after_open"] = np.mean([c.isupper() and not e for c, e in zip(first[1:], ends[:-1])]) if n > 1 else 0.0
    f["caps_lines"] = np.mean([sum(ch.isalpha() for ch in t) >= 3 and t.upper() == t for t in texts])
    f["bold_lines"] = np.mean([all(fl & 16 or "bold" in fn for _, fl, fn, _ in r[5]) for r in rows])
    f["big_lines"] = np.mean([max(sz for sz, *_ in r[5]) > 1.15 * body for r in rows])
    words = allc.split()
    f["word_len"] = np.mean([len(w) for w in words]) / 10 if words else 0.0
    f["chars_line"] = np.mean([len(t) for t in texts]) / 100
    full_ix = wid > 0.6 * W
    f["ragged"] = float(np.std(x1[full_ix]) / W) if full_ix.sum() > 2 else 0.0
    f["sentences_line"] = np.mean([len(re.findall(r"[.!?\u3002]\s", t + " ")) for t in texts])
    f["punct_inside"] = np.mean([bool(re.search(r"[.!?;\u3002]\s+\S", t)) for t in texts])

    # vertical rhythm inside the main column: stanza breaks are gaps of about one blank line between groups of short lines
    order = np.argsort(y0)
    gaps = np.diff(y0[order]) / lh
    gaps = gaps[(gaps > 0.3) & (gaps < 8)]
    if len(gaps):
        med = np.median(gaps)
        f["g25"], f["g75"], f["g90"] = np.percentile(gaps, [25, 75, 90]) / (med + 1e-6)
        f["gap_med"] = med
        f["gap_big"] = np.mean(gaps > 1.6 * med)
        f["gap_cv"] = float(np.std(gaps) / (med + 1e-6))
        f["stanza"] = f["gap_big"] * f["short"]
    return np.array([f[k] for k in NAMES], np.float32)


def reader_features(page):
    """Features of the page as the restyling reader sees it (layout.refont paragraphs, then the items of the handwriting pipeline):
    shares of characters by item kind, paragraph sizes, alignment, columns and tables. ~100-200 ms a page."""
    from ..layout import refont as R
    from .structure import to_items
    from .texts import blocks_of
    f = dict.fromkeys(READER, 0.0)
    try:
        blocks = R.page_blocks(R.Source(page.parent), page.number, False)
        items = blocks_of(page, blocks=[dict(b, lines=list(b["lines"])) for b in blocks])
    except Exception:
        return np.array([f[k] for k in READER], np.float32)
    tot = sum(len(it[1]) for it in items) or 1
    for k in ("h", "p", "li", "l", "table"):
        f["r_" + k] = sum(len(it[1]) for it in items if it[0] == k) / tot
    f["r_tables"] = sum(it[0] == "table" for it in items)
    f["r_cols"] = sum(i["k"] == "cols" for i in to_items(items)) if items else 0
    if blocks:
        nl = [len(b["lines"]) for b in blocks]
        chars = [sum(len(s["text"]) for l in b["lines"] for s in l["spans"]) for b in blocks]
        ct = sum(chars) or 1
        f["r_blocks"] = len(blocks) / 30
        f["r_lines_p"] = float(np.mean(nl)) / 10
        f["r_max_lines"] = max(nl) / 30
        f["r_one_line"] = np.mean([k == 1 for k in nl])
        for a in ("center", "right", "justify"):
            f["r_" + a] = sum(c for b, c in zip(blocks, chars) if b["align"] == a) / ct
    return np.array([f[k] for k in READER], np.float32)


# fitted by tools/pagekind_eval.py --fit (standardisation and logistic regression on all labelled pages)
MODEL_KINDS = ('plain', 'poems', 'formatted')
MEAN = np.array([0.860714, 0.614814, 0.240747, 0.108701, 0.457773, 1.83398, 0.837511, 0.0248648, 0.0588059, 1.89575, 0.0299604, 0.0288456, 0.0478618, 0.022028, 1.5251, 0.0480203, 1.1477, 0.123564, 0.435408, 0.0341373, 0.213436, 0.035805, 0.395977, 0.448096, 1.33977, 0.106293, 0.679537, 0.0142306, 0.0212372, 0.374332, 0.705904, 0.0550837, 0.0277992, 0.0398456, 0.0160811, 1.12693, 0.020648, 1.76448, 0.0120993, 0.00132592, 0.0265351, 0.339543, 0.003861, 0.0457288, 0.00581137, 0.484016, 0.765618, 1.00137, 1.14975, 0.95072, 1.17365, 1.56369, 0.185898, 0.138445, 0.0624418, 0.0568763, 1.30409, 0.524108, 0.175504, 0.475905, 0.228721])
SCALE = np.array([0.4076, 0.257003, 0.200547, 0.121773, 0.289224, 1.10433, 0.174925, 0.0531881, 0.111438, 1.35023, 0.0786863, 0.0756239, 0.09164, 0.0590275, 0.715485, 0.114493, 0.297852, 0.115062, 0.318423, 0.0516211, 0.156451, 0.0832784, 0.291991, 0.311975, 0.609134, 0.308907, 1.64211, 0.0202511, 0.0520732, 0.156595, 0.169604, 0.122478, 0.134101, 0.143287, 0.0828177, 0.322591, 0.0741473, 0.947733, 0.0911138, 0.0156176, 0.0471805, 0.180341, 0.0620169, 0.111323, 0.0275286, 0.315329, 0.388781, 0.418186, 0.503252, 0.119934, 0.312812, 0.629902, 0.210847, 0.302356, 0.101459, 0.12939, 6.39347, 0.236048, 0.192714, 0.557732, 0.209749])
COEF = np.array([np.array([0.235509, 0.12426, -0.0475401, 0.156943, 0.609314, -0.362651, 0.0889308, -0.026942, -0.179923, -0.266557, -0.198056, -0.0590551, -0.138311, -0.220582, 0.0129793, 0.0158189, 0.237913, -0.0189872, -0.15224, 0.252636, 0.12168, -0.016806, -0.497173, 0.261052, -0.341066, 0.33424, 0.0179613, 0.0783435, 0.284387, -0.203832, 0.356823, -0.0260423, -0.0168961, -0.0498501, 0.291871, 0.0137512, 0.344843, -0.131139, -0.185637, 0.24795, 0.247288, 0.370152, 0.423611, -0.289604, -0.116418, 0.11115, 0.373497, 0.175566, -0.0674024, -0.068114, 0.0467693, -0.559722, -0.101693, -0.042178, -0.171745, -0.156181, -0.060301, 0.0963663, 0.0959573, 0.185451, 0.64078]), np.array([-0.23783, -0.318022, 0.0394443, 0.0917118, -0.38469, 0.242538, -0.17564, 0.0272865, 0.143455, 0.333179, 0.100502, 0.106821, -0.0649768, 0.137969, -0.272529, -0.218977, -0.167423, 0.0135429, 0.0891158, -0.00900986, -0.0652633, -0.262936, 0.351828, -0.244849, 0.260735, -0.137145, -0.117318, -0.456261, -0.172234, -0.174831, -0.718189, -0.29699, -0.150225, 0.0748613, -0.134806, -0.274345, -0.276532, -0.228623, -0.0380502, -0.0366098, 0.116599, -0.166593, -0.423704, -0.0144287, 0.0827926, 0.141462, -0.0284912, -0.247277, 0.162692, 0.0669624, -0.262609, 0.188606, 0.0424394, -0.10998, -0.0802471, 0.159845, -0.180707, -0.744433, 0.229905, -0.306081, -0.585864]), np.array([0.00232107, 0.193761, 0.00809576, -0.248655, -0.224624, 0.120113, 0.0867088, -0.000344431, 0.036468, -0.066622, 0.0975541, -0.0477658, 0.203288, 0.0826136, 0.25955, 0.203158, -0.0704902, 0.00544444, 0.0631242, -0.243627, -0.0564165, 0.279742, 0.145345, -0.0162035, 0.0803309, -0.197095, 0.0993564, 0.377917, -0.112153, 0.378663, 0.361366, 0.323033, 0.167121, -0.0250111, -0.157066, 0.260593, -0.0683107, 0.359762, 0.223687, -0.21134, -0.363886, -0.203559, 9.28626e-05, 0.304033, 0.0336252, -0.252612, -0.345006, 0.0717111, -0.09529, 0.00115175, 0.21584, 0.371116, 0.0592537, 0.152158, 0.251992, -0.00366434, 0.241007, 0.648067, -0.325863, 0.12063, -0.0549157])])
INTERCEPT = np.array([0.457082, -1.33621, 0.879132])


def kind(page):
    x = (features(page) - MEAN) / SCALE
    z = COEF @ x + INTERCEPT
    p = np.exp(z - z.max())
    p /= p.sum()
    return MODEL_KINDS[int(p.argmax())], dict(zip(MODEL_KINDS, p.round(3).tolist()))
