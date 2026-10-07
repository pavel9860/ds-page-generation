"""Restyled pages: a digital-born PDF page set again in other fonts and scales. Text blocks keep the page's flow,
columns, alignment, bold/italic and heading/body hierarchy; rules, boxes and pictures follow the text they sit next to.
The text of the following pages flows in, so the page stays full. Fonts come from find_fonts, matched to the
page's scripts by the real glyph coverage of every font file that is drawn with."""
import bisect
import glob
import os
import pickle
import statistics
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pymupdf
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

from ..config import RefontCfg
from .text import _font_cmap, find_fonts

C = RefontCfg()
TEXT_FLAGS = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES      # lines only: pictures come from get_image_info
STYLE_OF = {"": 0, "regular": 0, "roman": 0, "book": 0, "bold": 1, "italic": 2, "oblique": 2, "bolditalic": 3,
            "boldoblique": 3}
LEADING_VOWELS = "เแโไใ"
JOINERS = "‌‍"
NO_SPACE_SCRIPTS = ("THAI", "LAO", "KHMER", "MYANMAR", "CJK", "HIRAGANA", "KATAKANA", "HANGUL", "IDEOGRAPHIC")
CJK_NAMES = ("hiragana", "katakana", "hangul", "bopomofo", "ideographic")
NOTO_PREFIXES = ("notosans", "notoserif")
TERMINAL = tuple(".;:!?…。")
THAI_PUA = {0xF700: 0x0E10, 0xF701: 0x0E34, 0xF702: 0x0E35, 0xF703: 0x0E36, 0xF704: 0x0E37, 0xF705: 0x0E48,
            0xF706: 0x0E49, 0xF707: 0x0E4A, 0xF708: 0x0E4B, 0xF709: 0x0E4C, 0xF70A: 0x0E48, 0xF70B: 0x0E49,
            0xF70C: 0x0E4A, 0xF70D: 0x0E4B, 0xF70E: 0x0E4C, 0xF70F: 0x0E0D, 0xF710: 0x0E31, 0xF711: 0x0E4D,
            0xF712: 0x0E47, 0xF713: 0x0E48, 0xF714: 0x0E49, 0xF715: 0x0E4A, 0xF716: 0x0E4B, 0xF717: 0x0E4C,
            0xF718: 0x0E38, 0xF719: 0x0E39, 0xF71A: 0x0E3A}


def _rtl_char(ch):
    return unicodedata.bidirectional(ch) in ("R", "AL")


def is_rtl(text):
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(_rtl_char(c) for c in letters) > C.rtl_share * len(letters)


def _script(ch):
    word = unicodedata.name(ch, "").split(" ")[0].lower()
    return "cjk" if word in CJK_NAMES else word


def _gray(rgb):
    return int(0.299 * (rgb >> 16 & 255) + 0.587 * (rgb >> 8 & 255) + 0.114 * (rgb & 255))


def _gray_f(c):
    return int(255 * (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])) if len(c) == 3 else int(255 * c[0])


def span_style(s):
    n = s["font"].lower()
    return (bool(s["flags"] & 16) or any(k in n for k in ("bold", "black", "heavy")),
            bool(s["flags"] & 2) or "italic" in n or "oblique" in n)


def span_mono(s):
    return bool(s["flags"] & 8) or any(k in s["font"].lower() for k in ("mono", "courier", "consolas"))


def _tilted(l, dy):
    """A tilted line is one picture-like glyph run: set as a pixmap of the page."""
    first = l["spans"][0]
    box = pymupdf.Rect(l["bbox"])
    span = dict(text="\ufffc", size=round(first["size"] * C.size_step) / C.size_step, gray=_gray(first["color"]), bold=False,
                italic=False, mono=False, nat=first["ascender"] - first["descender"] or C.line_em, glyph=box, tall=True, bbox=box + (0, dy, 0, dy))
    return [dict(bbox=box + (0, dy, 0, dy), spans=[span])]


def _span(s, text, dy):
    bold, italic = span_style(s)
    glyph = pymupdf.Rect(s["bbox"]) if all(unicodedata.category(c) == "Co" for c in text.strip()) else None
    return dict(text=text, size=round(s["size"] * C.size_step) / C.size_step, gray=_gray(s["color"]), bold=bold, italic=italic,
                mono=span_mono(s), nat=s["ascender"] - s["descender"] or C.line_em, glyph=glyph, bbox=pymupdf.Rect(s["bbox"]) + (0, dy, 0, dy))


def _runs(l, dy):
    """Segments of a line: its spans, cut where they are far apart (a gutter between columns, a tab)."""
    spans = []
    for s in l["spans"]:
        text = s["text"].replace("\x00", "").replace("\xa0", " ").translate(THAI_PUA)
        if text.strip() and round(s["size"] * C.size_step) > 0:
            spans.append(_span(s, text, dy))
    runs = []
    for span in spans:
        far = runs and max(span["bbox"].x0 - runs[-1][-1]["bbox"].x1, runs[-1][-1]["bbox"].x0 - span["bbox"].x1) > C.split_em * span["size"]
        if far or not runs:
            runs.append([])
        runs[-1].append(span)
    y0, y1 = pymupdf.Rect(l["bbox"]).y0 + dy, pymupdf.Rect(l["bbox"]).y1 + dy
    return [dict(bbox=pymupdf.Rect(min(sp["bbox"].x0 for sp in run), y0, max(sp["bbox"].x1 for sp in run), y1), spans=run) for run in runs]


def _embolden(line, over):
    """The words an overprinted run repeats become bold in the line under it (a span is split where the run covers part of it)."""
    for o in over["spans"]:
        word = o["text"].strip()
        for k, sp in enumerate(line["spans"]):
            width = sp["bbox"].width
            if width <= 0 or min(sp["bbox"].x1, o["bbox"].x1) - max(sp["bbox"].x0, o["bbox"].x0) < 0.5 * min(width, o["bbox"].width):
                continue
            at = sp["text"].find(word)
            if at < 0:
                continue
            n = len(sp["text"])
            edge = lambda c: sp["bbox"].x0 + width * c / n
            parts = [(sp["text"][:at], 0, at, sp["bold"]), (word, at, at + len(word), True), (sp["text"][at + len(word):], at + len(word), n, sp["bold"])]
            line["spans"][k:k + 1] = [dict(sp, text=t, bold=b, bbox=pymupdf.Rect(edge(a), sp["bbox"].y0, edge(z), sp["bbox"].y1)) for t, a, z, b in parts if t]
            break


def _overprint(segs):
    """A line drawn over another (fake bold: some words, or the whole line, printed again in the same place) is not a line of
    its own: the words it repeats are bold in the line under it. Lines are compared with those that start within a line's reach."""
    kept, rows, order = [], defaultdict(list), {id(l): i for i, l in enumerate(segs)}
    for seg in sorted(segs, key=lambda l: -l["bbox"].get_area()):
        box, bin_ = seg["bbox"], round(seg["bbox"].y0 / C.overprint_pt)
        text = "".join(_text(seg).split())
        under = next((k for b in (bin_ - 1, bin_, bin_ + 1) for k in rows[b]
                      if box.get_area() > 0 and (box & k["bbox"]).get_area() >= C.duplicate_overlap * box.get_area()
                      and text in "".join(_text(k).split())), None)
        if under is None:
            kept.append(seg)
            rows[bin_].append(seg)
        else:
            _embolden(under, seg)
    return sorted(kept, key=lambda l: order[id(l)])


def extract_segments(page, dy):
    """Text lines of a page, each cut into one segment per run."""
    lines = [l for b in page.get_text("dict", flags=TEXT_FLAGS)["blocks"] for l in b.get("lines", [])]
    segs = []
    for l in lines:
        if abs(l["dir"][1]) > C.tilt:
            segs += _tilted(l, dy) if "".join(s["text"] for s in l["spans"]).strip() else []
        else:
            segs += _runs(l, dy)
    return _overprint(segs)


def _rsize(r):
    """Font size of a line: the size most of its characters have (a drop cap or a superscript does not set it)."""
    cnt = Counter()
    for s in r["spans"]:
        cnt[s["size"]] += len(s["text"].strip())
    return cnt.most_common(1)[0][0]


def _text(r):
    return "".join(s["text"] for s in r["spans"])


def column_intervals(segs):
    """Text columns of a page: x ranges of the body-size lines, merged while they overlap. A heading that spans two
    columns does not take part, so a gutter stays a gutter."""
    cnt = Counter()
    for l in segs:
        for s in l["spans"]:
            cnt[round(s["size"] * 2) / 2] += len(s["text"])
    body = cnt.most_common(1)[0][0] if cnt else 0.0
    lefts = Counter(round(l["bbox"].x0 / C.edge_bin) for l in segs)
    rights = Counter(round(l["bbox"].x1 / C.edge_bin) for l in segs)
    spans = sorted((l["bbox"].x0, l["bbox"].x1) for l in segs
                   if _rsize(l) <= C.body_band * body and l["bbox"].width >= C.column_line * _rsize(l)
                   and (lefts[round(l["bbox"].x0 / C.edge_bin)] >= C.edge_lines or rights[round(l["bbox"].x1 / C.edge_bin)] >= C.edge_lines))
    if not spans:
        return []
    lo, hi = int(min(x0 for x0, _ in spans)), int(max(x1 for _, x1 in spans)) + 1
    cover = np.zeros(hi - lo + 1)
    for x0, x1 in spans:
        cover[int(x0) - lo:int(x1) - lo + 1] += 1
    solid = cover >= C.gutter_cover * cover.max()
    cols, start = [], None
    for i, on in enumerate(list(solid) + [False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            cols.append([start + lo, i - 1 + lo])
            start = None
    merged = []
    for c in cols:
        if merged and c[0] - merged[-1][1] < C.gutter_em * body:
            merged[-1][1] = c[1]
        else:
            merged.append(c)
    out = []
    for a, z in merged:
        inside = [(x0, x1) for x0, x1 in spans if x0 >= a - 2 and x1 <= z + 2]
        out.append((min((x0 for x0, _ in inside), default=a), max((x1 for _, x1 in inside), default=z)))
    return out


def column_of(bbox, cols):
    """Index of the column a box lies in; None when it spans several columns or lies in none."""
    hits = [i for i, (a, z) in enumerate(cols) if min(bbox.x1, z) - max(bbox.x0, a) >= C.column_hit * max(bbox.width, 1.0)]
    return hits[0] if len(hits) == 1 else None


def join_rows(segs, cols, vlines=()):
    rows, buckets = [], defaultdict(list)
    walls = sorted(vlines)
    wall_x = [w[0] for w in walls]
    for l in sorted(segs, key=lambda l: l["bbox"].x0):
        bb = l["bbox"]
        lx0, ly0, lx1, ly1 = bb.x0, bb.y0, bb.x1, bb.y1
        sz = max(s["size"] for s in l["spans"])
        col = column_of(bb, cols)
        key = int(((ly0 + ly1) / 2) // C.row_bin)
        target = None
        for k in range(key - C.row_reach, key + C.row_reach + 1):
            for r in buckets.get(k, ()):
                x1, (py0, py1, psz) = r["x1"], r["last"]
                if not -C.join_back_em * sz < lx0 - x1 < C.join_ahead_em * sz:
                    continue
                small = min(py1 - py0, ly1 - ly0)
                inline = abs((py0 + py1) / 2 - (ly0 + ly1) / 2) < C.inline_share * sz or (
                    min(sz, psz) < C.superscript_ratio * max(sz, psz) and min(py1, ly1) - max(py0, ly0) >= 0.5 * small)
                if not inline or not (col is None or r["col"] is None or col == r["col"]):
                    continue
                mid = ly0 + 0.5 * (ly1 - ly0)
                lo, hi = bisect.bisect_left(wall_x, x1 - 1), bisect.bisect_right(wall_x, lx0 + 1)
                if any(y0 <= mid <= y1 for _, y0, y1 in walls[lo:hi]):
                    continue
                target = r
                break
            if target:
                break
        if target:
            target["segs"].append(l)
            target["bbox"] |= bb
            target["x1"] = target["bbox"].x1
            target["last"] = (ly0, ly1, _rsize(l))
            target["col"] = target["col"] if target["col"] is not None else col
        else:
            target = dict(bbox=pymupdf.Rect(bb), segs=[l], col=col, x1=lx1, last=(ly0, ly1, _rsize(l)))
            rows.append(target)
        if target not in buckets[key]:
            buckets[key].append(target)
    for r in rows:
        ordered = r["segs"][::-1] if is_rtl("".join(s["text"] for g in r["segs"] for s in g["spans"])) else r["segs"]
        r["spans"] = [s for g in ordered for s in g["spans"]]
    return sorted(rows, key=lambda r: (r["bbox"].y0, r["bbox"].x0))


def paragraphs(segs, rules=((), ())):
    cols = column_intervals(segs)
    rows = join_rows(segs, cols, rules[0])
    colw = defaultdict(float)
    for r in rows:
        k = (round(_rsize(r)), round(r["bbox"].x0 / C.para_bin))
        colw[k] = max(colw[k], r["bbox"].width)
    blocks = []
    for r in rows:
        sz = _rsize(r)
        k = round(r["bbox"].x0 / C.para_bin)
        W = max(colw[(round(sz), j)] for j in (k - 1, k, k + 1))
        for b in reversed(blocks[-C.lookback:]):
            last = b["lines"][-1]["bbox"]
            pitch = r["bbox"].y0 - last.y0
            ref = b["pitch"] if len(b["lines"]) > 1 else C.single_pitch_em * sz
            edge = abs(r["bbox"].x0 - last.x0) < 3 or abs(r["bbox"].x1 - last.x1) < 3
            ruled = any(y0 <= y <= y1 and x0 < min(last.x1, r["bbox"].x1) and x1 > max(last.x0, r["bbox"].x0) for y, x0, x1 in rules[1]
                        for y0, y1 in [(last.y1 - 0.5 * sz, r["bbox"].y0 + 0.5 * sz)])
            first_ltr = len(b["lines"]) == 1 and C.indent_min_em * sz < last.x0 - r["bbox"].x0 < C.indent_max_em * sz and abs(last.x1 - r["bbox"].x1) < C.edge_match_em * sz + 3
            first_rtl = len(b["lines"]) == 1 and C.indent_min_em * sz < r["bbox"].x1 - last.x1 < C.indent_max_em * sz and abs(last.x0 - r["bbox"].x0) < C.edge_match_em * sz + 3
            step_ltr = (len(b["lines"]) <= C.indent_lines and all(l["bbox"].x0 > r["bbox"].x0 + 0.5 * sz for l in b["lines"])
                        and abs(last.x1 - r["bbox"].x1) < C.edge_match_em * sz + 3)
            step_rtl = (len(b["lines"]) <= C.indent_lines and all(l["bbox"].x1 < r["bbox"].x1 - 0.5 * sz for l in b["lines"])
                        and abs(last.x0 - r["bbox"].x0) < C.edge_match_em * sz + 3)
            if (abs(_rsize(b["lines"][-1]) - sz) < C.same_size_pt and (edge or first_ltr or first_rtl or step_ltr or step_rtl) and b["col"] == r["col"]
                    and not ruled
                    and C.indent_min_em * sz < pitch < ref * C.pitch_tol and b["W"] > C.min_width_em * sz
                    and (last.width > C.full_line * b["W"] or (last.width > C.ragged_fill * b["W"] and not _text(b["lines"][-1]).rstrip().endswith(TERMINAL)))):
                if len(b["lines"]) == 1:
                    b["pitch"] = pitch
                b["lines"].append(r)
                b["bbox"] |= r["bbox"]
                break
        else:
            blocks.append(dict(bbox=pymupdf.Rect(r["bbox"]), lines=[r], W=W, pitch=0, col=r["col"]))
    for b in blocks:
        b["rtl"] = is_rtl("".join(_text(l) for l in b["lines"]))
        b["size"] = max(_rsize(l) for l in b["lines"])
        b["align"], edge = _align(b)
        if b["align"] == "justify":
            b["bbox"] = pymupdf.Rect(edge, b["bbox"].y0, b["bbox"].x1, b["bbox"].y1) if b["rtl"] else pymupdf.Rect(
                b["bbox"].x0, b["bbox"].y0, edge, b["bbox"].y1)
        b["colspan"] = cols[b["col"]] if b["col"] is not None else None
        if b["align"] != "justify" and len(b["lines"]) >= C.ragged_lines:
            ends = sorted(l["bbox"].x0 if b["rtl"] else l["bbox"].x1 for l in b["lines"])
            cut = ends[int((1 - C.ragged_percentile) * (len(ends) - 1))] if b["rtl"] else ends[int(C.ragged_percentile * (len(ends) - 1))]
            b["bbox"] = pymupdf.Rect(max(b["bbox"].x0, cut), b["bbox"].y0, b["bbox"].x1, b["bbox"].y1) if b["rtl"] else pymupdf.Rect(
                b["bbox"].x0, b["bbox"].y0, min(b["bbox"].x1, cut), b["bbox"].y1)
        if b["colspan"] is not None:
            clamped = pymupdf.Rect(max(b["bbox"].x0, b["colspan"][0]), b["bbox"].y0, min(b["bbox"].x1, b["colspan"][1]), b["bbox"].y1)
            if clamped.width >= C.trim_max * b["bbox"].width:
                b["bbox"] = clamped
        nat = statistics.median(s["nat"] for l in b["lines"] for s in l["spans"])
        pitch = (statistics.median(b["lines"][i + 1]["bbox"].y0 - b["lines"][i]["bbox"].y0
                                   for i in range(len(b["lines"]) - 1)) if len(b["lines"]) > 1 else b["size"] * nat)
        b["lead"] = min(max(pitch / (b["size"] * nat), 1.0), C.max_lead)
        b["src_pitch"] = pitch
    return blocks


def _spread(v):
    return max(v) - min(v) if v else 0.0


def _align(b):
    """(alignment, end edge): where the lines of a block start flush (first-line indent aside) and end flush (last line
    aside). The end edge is the common right edge (left edge for right-to-left text) of a justified block."""
    ls = [l["bbox"] for l in b["lines"]]
    n, rtl = len(ls), b["rtl"]
    start = (lambda r: r.x1) if rtl else (lambda r: r.x0)
    end = (lambda r: r.x0) if rtl else (lambda r: r.x1)
    if n == 1:
        return "left", None
    starts = [start(r) for r in (ls[1:] if n > 2 else ls)]
    ends = [end(r) for r in (ls[1:-1] if n > 3 else ls[:-1] if n == 3 else ls)]
    ref = Counter(round(v) for v in starts).most_common(1)[0][0]
    start_flush = sum(abs(v - ref) <= 3 for v in starts) >= C.start_share * len(starts)
    text = "".join(_text(l) for l in b["lines"])
    tol = (C.short_line_em if sum(_nobreak(c) for c in text) > C.cjk_share * len(text) else C.justify_edge) * b["size"]
    edge = max(ends, key=lambda e: sum(abs(x - e) <= tol for x in ends))
    end_flush = sum(abs(e - edge) <= tol for e in ends) >= C.flush_share * len(ends)
    if start_flush and end_flush and min(r.width for r in ls[1:-1] or ls) > C.justify_fill * max(r.width for r in ls):
        return "justify", edge
    if _spread([(r.x0 + r.x1) / 2 for r in ls]) < 3 and not (start_flush or end_flush):
        return "center", None
    if start_flush:
        return "right" if rtl else "left", None
    if end_flush:
        return "left" if rtl else "right", None
    return "right" if rtl else "left", None


_TRAITS = {}
_TRAITS_FILE = Path("~/.cache/dspages/traits.pkl").expanduser()


def _traits(path):
    """(monospaced, variable) read from the font file itself; remembered on disk by file and modification time."""
    if not _TRAITS and _TRAITS_FILE.exists():
        try:
            with open(_TRAITS_FILE, "rb") as fh:
                _TRAITS.update(pickle.load(fh))
        except Exception:
            _TRAITS["_"] = None
    key = (path, os.path.getmtime(path))
    if key not in _TRAITS:
        try:
            f = TTFont(path, fontNumber=0, lazy=True)
            _TRAITS[key] = (bool(f["post"].isFixedPitch) if "post" in f else False, "fvar" in f)
        except Exception:
            _TRAITS[key] = (False, False)
        _TRAITS["new"] = True
    return _TRAITS[key]


def save_traits():
    if _TRAITS.pop("new", False):
        _TRAITS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _TRAITS_FILE.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump({k: v for k, v in _TRAITS.items() if isinstance(k, tuple)}, fh)
        os.replace(tmp, _TRAITS_FILE)


def _family_key(path):
    return os.path.splitext(os.path.basename(path))[0].split("[")[0].split("-")[0].split(",")[0].lower()


def _family_script(key):
    prefix = next((p for p in NOTO_PREFIXES if key.startswith(p)), None)
    if prefix is None:
        return ""
    return key[len(prefix):].replace("display", "").replace("ui", "")


@lru_cache(maxsize=4)
def families(fonts):
    fam = defaultdict(dict)
    for p in fonts:
        base = os.path.splitext(os.path.basename(p))[0].split("[")[0]
        suffix = base.split("-", 1)[1].lower() if "-" in base else ""
        if _traits(p)[1]:
            italic = suffix == "italic"
            fam[_family_key(p)].setdefault(2 * italic, (p, C.regular_weight))
            fam[_family_key(p)].setdefault(2 * italic + 1, (p, C.bold_weight))
        elif suffix in STYLE_OF:
            fam[_family_key(p)].setdefault(STYLE_OF[suffix], (p, None))
    save_traits()
    return {k: v for k, v in fam.items() if 0 in v}


def style_file(files, bold, italic):
    for key in (int(bold) + 2 * int(italic), int(bold), 2 * int(italic), 0):
        if key in files:
            return files[key]


def pick_family(rng, fams, chars, bold=False, italic=False, need=()):
    letters = {c for c in chars if c.isalpha()} or set(chars)
    scripts = {k for k, v in Counter(_script(c) for c in letters).items() if v >= C.script_share * len(letters)}
    cps = {ord(c) for c in chars}
    score = {}
    for k, v in fams.items():
        rest = _family_script(k)
        if rest == "" or any(s in rest for s in scripts):
            score[k] = len(cps & _font_cmap(style_file(v, bold, italic)[0]))
    styled = {k: v for k, v in score.items() if all(int(b) + 2 * int(i) in fams[k] for b, i in need)}
    score = styled or score
    best = max(score.values())
    pool = sorted(k for k, s in score.items() if s == best)
    return pool[rng.integers(0, len(pool))]


def font(path, px, wght):
    return _font(path, round(px * C.font_step) / C.font_step, wght)


@lru_cache(maxsize=4096)
def _font(path, px, wght):
    f = ImageFont.truetype(path, max(px, 1.0), layout_engine=ImageFont.Layout.RAQM)
    f.key = (path, px, wght)
    if wght is not None:
        f.set_variation_by_axes([min(max(wght, a["minimum"]), a["maximum"]) if a["name"] in (b"Weight", "Weight")
                                 else a["default"] for a in f.get_variation_axes()])
    return f


@lru_cache(maxsize=1 << 18)
def text_width(path, px, wght, text, direction):
    """Width of a text: shaped as a whole for the scripts that join or reorder, else the sum of the letters' advances
    (the cached ones, which render sets the letters by)."""
    f = font(path, px, wght)
    return f.getlength(text, direction=direction) if needs_shaping(text) else sum(_glyph(f, ch)[3] for ch in text)


class Typesetter:
    """Font file for a piece of text: the role's family, or another family when that file lacks a glyph."""

    def __init__(self, fonts, rng, chars, need=()):
        fams = families(fonts)
        self.rng = rng
        self.prose = {k: v for k, v in fams.items() if not _traits(v[0][0])[0]}
        self.mono = {k: v for k, v in fams.items() if _traits(v[0][0])[0]}
        self.role = {"head": pick_family(rng, self.prose, chars, need=need), "body": pick_family(rng, self.prose, chars, need=need),
                     "mono": pick_family(rng, self.mono, chars)}
        self.fallback = {}
        self.dropped = 0
        self.unresolved = Counter()

    def resolve(self, role, text, bold, italic):
        """Runs (path, weight, text) covering `text`: the role's family where it has the glyph, other families elsewhere."""
        pool = self.mono if role == "mono" else self.prose
        path, wght = style_file(pool[self.role[role]], bold, italic)
        cmap = _font_cmap(path)
        if all(ord(c) in cmap or unicodedata.category(c) == "Cf" for c in text):
            return [(path, wght, text)]
        runs = []
        for c in text:
            font_c = (path, wght) if ord(c) in cmap or c in JOINERS else self._covering(c, bold, italic)
            if font_c is None:
                self.unresolved[c] += 1
            elif runs and runs[-1][0] == font_c:
                runs[-1][1] += c
            else:
                runs.append([font_c, c])
        self.dropped = sum(self.unresolved.values())
        return [(f[0], f[1], t) for f, t in runs]

    def _covering(self, c, bold, italic):
        key = (c, bold, italic)
        if key not in self.fallback:
            guess = style_file(self.prose[pick_family(self.rng, self.prose, {c}, bold, italic)], bold, italic)
            if ord(c) not in _font_cmap(guess[0]):
                guess = next((style_file(v, bold, italic) for k, v in sorted(self.prose.items())
                              if ord(c) in _font_cmap(style_file(v, bold, italic)[0])), None)
            self.fallback[key] = guess
        return self.fallback[key]


def _clusters(text):
    out = []
    for ch in text:
        if out and (unicodedata.combining(ch) or unicodedata.category(ch) in ("Mn", "Mc", "Me") or ch in JOINERS + "ำຳ"):
            out[-1] += ch
        else:
            out.append(ch)
    return out


def _nobreak(text):
    return any(unicodedata.name(c, "").split(" ")[0] in NO_SPACE_SCRIPTS for c in text)


def _pieces(word):
    if not _nobreak(word):
        return [word]
    out, cur = [], ""
    for ch in word:
        if cur and _nobreak(cur) != _nobreak(ch) and not unicodedata.combining(ch):
            out.append(cur)
            cur = ""
        cur += ch
    return out + [cur]


def _glyph_token(s, px, space, tab):
    return dict(text="", glyph=(s["number"], s["glyph"]), size=s["size"], px=px, gray=s["gray"], gap=space, tab=tab, brk=False,
                rtl=False, path=None, wght=None, tall=s.get("tall", False))


def _word_tokens(s, typesetter, role, px, space, tab):
    """Tokens of a span's words: the pieces of a word each in the font that has its letters. -> (tokens, space, tab)."""
    toks = []
    for i, w in enumerate(s["text"].replace("\n", " ").split(" ")):
        space = space or i > 0
        for piece in _pieces(w) if w else ():
            for path, wght, text in typesetter.resolve(role, piece, s["bold"], s["italic"]):
                toks.append(dict(text=text, path=path, wght=wght, px=px, gray=s["gray"], gap=space, tab=tab, size=s["size"],
                                 sig=(round(s["size"], 1), s["bold"], s["italic"], s["mono"], s["gray"]),
                                 brk=_nobreak(text), rtl=any(_rtl_char(c) for c in text if c.isalpha())))
                space, tab = False, 0.0
    return toks, space, tab


def tokens(b, typesetter, role_of, px_of):
    toks, space, prev, tab = [], False, None, 0.0
    for l in b["lines"]:
        for s in l["spans"]:
            if prev is not None and abs(s["bbox"].y0 - prev.y0) < 0.5 * s["size"]:
                src_gap = max(s["bbox"].x0 - prev.x1, prev.x0 - s["bbox"].x1)
                if src_gap > C.source_gap_em * s["size"]:
                    space = True
                    tab = src_gap if src_gap > C.tab_em * s["size"] else 0.0
            prev = s["bbox"]
            if s["glyph"] is not None:
                toks.append(_glyph_token(s, px_of(s["size"]), space, tab))
                space, tab = False, 0.0
                continue
            role = "mono" if s["mono"] else role_of(s["size"])
            more, space, tab = _word_tokens(s, typesetter, role, px_of(s["size"]), space, tab)
            toks += more
        space = not _nobreak(_text(l)[-1:])
    return toks


def gap_w(t):
    if not t["gap"]:
        return 0.0
    space = C.glyph_gap * t["px"] if t["path"] is None else _glyph(font(t["path"], t["px"], t["wght"]), " ")[3]
    return max(space, t["tab"] * t["px"] / t["size"])


def measure(t):
    if t["path"] is None:
        return None, t["glyph"][1].width * t["px"] / t["size"]
    f = font(t["path"], t["px"], t["wght"])
    return f, text_width(t["path"], t["px"], t["wght"], t["text"], "rtl" if t["rtl"] else "ltr")


def _split(t, room):
    cl = _clusters(t["text"])
    lo, hi = 0, len(cl)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if measure(dict(t, text="".join(cl[:mid])))[1] <= room:
            lo = mid
        else:
            hi = mid - 1
    while lo > 1 and cl[lo - 1] in LEADING_VOWELS:
        lo -= 1
    return "".join(cl[:lo]), "".join(cl[lo:])


def wrap(toks, W, indent_first, indent_rest=0.0):
    lines, cur, curw = [], [], 0.0
    queue = toks[::-1]
    while queue:
        t = queue.pop()
        f, w = measure(t)
        sw = gap_w(t) if cur else 0.0
        avail = W - (indent_rest if lines else indent_first)
        if curw + sw + w > avail:
            if t["path"] is not None and (t["brk"] or w > W):
                head, tail = _split(t, avail - curw - sw)
                if not head and cur:
                    lines.append(cur)
                    cur, curw, sw = [], 0.0, 0.0
                    head, tail = _split(t, W)
                if not head:
                    cl = _clusters(t["text"])
                    head, tail = cl[0], "".join(cl[1:])
                if tail:
                    queue.append(dict(t, text=tail, gap=False))
                t = dict(t, text=head)
                f, w = measure(t)
            elif cur:
                lines.append(cur)
                cur, curw, sw = [], 0.0, 0.0
        cur.append((t, f, w, sw))
        curw += sw + w
    if cur:
        lines.append(cur)
    return lines


def visual(line):
    groups = []
    for item in line:
        ltr = not item[0]["rtl"] and any(c.isalnum() for c in item[0]["text"])
        if ltr and groups and groups[-1][1]:
            groups[-1][0].append(item)
        else:
            groups.append(([item], ltr))
    return [i for g, _ in groups[::-1] for i in g]


def _unshadowed(above, x0, x1):
    """The blocks above that are not hidden by nearer ones: a block whose width (within x0..x1) is mostly covered by blocks
    that start below its end has others between it and the block below, which carry it."""
    kept, top = [], {}
    for a in sorted(above, key=lambda a: -a["oy1"]):
        bins = range(int(max(a["x0"], x0) // C.shadow_bin), int(min(a["x1"], x1) // C.shadow_bin) + 1)
        hidden = sum(top.get(b, float("inf")) >= a["oy1"] - a["pitch"] / 2 for b in bins if b in top)
        if hidden < C.shadow_share * len(bins):
            kept.append(a)
        for b in bins:
            top[b] = min(top.get(b, float("inf")), a["oy0"])
    return kept


def follow_top(x0, x1, y0, pinned, k, done, gfx, gap_cap, sc):
    """Top of a block below the blocks that share its columns: their new bottoms plus the source gap, scaled. A gap of the
    following pages is cut to gap_cap unless one graphic (a frame, a cell border) spans it from block to block."""
    if pinned:
        return y0
    above = _unshadowed([a for a in done if min(a["x1"], x1) - max(a["x0"], x0) >= max(2.0, C.follow_overlap * min(a["x1"] - a["x0"], x1 - x0))
                         and a["oy0"] < y0 - 0.5 and not a["pinned"]], x0, x1)

    nearest = max(above, key=lambda a: a["oy1"], default=None)

    def follow(a):
        gap = max(y0 - a["oy1"], 0.0)
        if k and (a is not nearest or not any(r[0] < x1 and r[2] > x0 and r[1] <= a["oy1"] + 3 and r[3] >= y0 - 1 for r in gfx)):
            gap = min(gap, gap_cap)
        return a["ny1"] + gap * sc

    return max((follow(a) for a in above), default=y0)


def single_align(x0, x1, c0, c1, rtl):
    """Where a one-line block sits in its column: centred, flush to an edge, or left."""
    colw = max(c1 - c0, 1.0)
    left, right = x0 - c0, c1 - x1
    if left < C.edge_tol * colw + 2 and right < C.edge_tol * colw + 2:
        return "right" if rtl else "left"
    if left > C.align_inset * colw and abs(left - right) < C.edge_tol * colw + 3:
        return "center"
    if right < C.edge_tol * colw + 2 and left > C.align_inset * colw:
        return "right"
    if left < C.edge_tol * colw + 2:
        return "left"
    return "right" if rtl and right < left else "left"


def line_pitch(b, px, asc, desc):
    """Distance between the lines of a paragraph set at px: the source's, scaled with the letters, so a block takes the room
    it had; not tighter than min_lead of the font's own height."""
    return max(b["src_pitch"] * px / b["size"], C.min_lead * (asc + desc))


def indents(b, g, px):
    """First-line and following-line offsets from the block's start edge (hanging and nested indents)."""
    ls, bb = b["lines"], b.get("src_bbox", b["bbox"])
    if len(ls) < 2 or b["align"] == "center":
        return 0.0, 0.0
    if b["rtl"]:
        first, rest = bb.x1 - ls[0]["bbox"].x1, bb.x1 - statistics.median(l["bbox"].x1 for l in ls[1:])
    else:
        first, rest = ls[0]["bbox"].x0 - bb.x0, statistics.median(l["bbox"].x0 for l in ls[1:]) - bb.x0
    first, rest = first * g, rest * g
    return (first if first > 0.5 * px else 0.0), (rest if rest > 0.5 * px else 0.0)


def prepare(b, blocks, page_w, g, typesetter, role_of, px_of, cols, walls=()):
    bb = b["bbox"]
    x0, x1, y0, y1 = bb.x0 * g, bb.x1 * g, bb.y0 * g, bb.y1 * g
    if b.get("image"):
        return dict(b=b, image=True, x0=x0, x1=x1, y0=y0, y1=y1, pinned=b["pinned"], k=b["k"], h=b["rect"].height * g, tol=0.0)
    toks = tokens(b, typesetter, role_of, px_of)
    if not toks:
        return None
    multi = len(b["lines"]) > 1
    bpx = px_of(b["size"])
    lettered = next((t for t in toks if t["path"] is not None), None)
    asc, desc = font(lettered["path"], bpx, lettered["wght"]).getmetrics() if lettered else (C.ascent_share * bpx, (1 - C.ascent_share) * bpx)
    cx = (x0 + x1) / 2
    c0, c1 = (b["colspan"][0] * g, b["colspan"][1] * g) if b.get("colspan") else (min(a for a, _ in cols), max(z for _, z in cols))
    W, bx0, ind1, ind2 = x1 - x0, x0, 0.0, 0.0
    if multi:
        align = b["align"]
        ind1, ind2 = indents(b, g, bpx)
    else:
        align = single_align(x0, x1, c0, c1, b["rtl"])
        row = [(o["bbox"].x0 * g, o["bbox"].x1 * g) for o in blocks if o is not b and not o.get("image")
               and min(o["bbox"].y1, bb.y1) - max(o["bbox"].y0, bb.y0) > C.row_overlap * min(o["bbox"].height, bb.height)]
        row += [(x, x) for x, wy0, wy1 in walls if min(wy1, y1) - max(wy0, y0) > C.row_overlap * (y1 - y0)]
        slack = C.edge_slack * (c1 - c0)
        e0, e1 = max(c0 - slack, 2.0), min(c1 + slack, page_w * g - 2.0)
        right = min((a for a, z in row if a >= x1 - 1), default=e1)
        left = max((z for a, z in row if z <= x0 + 1), default=e0)
        cap = {"left": min(e1, right) - x0, "right": x1 - max(e0, left),
               "center": 2 * min(cx - max(e0, left), min(e1, right) - cx)}[align]
        Wa = max(min(C.room * W, cap), W)
        bx0, W = {"left": x0, "right": x1 - Wa, "center": cx - Wa / 2}[align], Wa
    lines = wrap(toks, W, ind1, ind2)
    natural = 0.0 if multi else sum(measure(t)[1] for t in toks) + sum(gap_w(t) for t in toks[1:])
    return dict(b=b, image=False, fit=natural / max(W, 1.0), x0=x0, x1=x1, y0=y0, y1=y1, bx0=bx0, W=W, align=align, ind=(ind1, ind2), lines=lines,
                pitch=line_pitch(b, bpx, asc, desc), asc=asc, desc=desc, multi=multi, pinned=b["pinned"], k=b["k"], rtl=b["rtl"],
                bpx=bpx, tol=C.row_tol_em * b["size"] * g, wrapped=not multi and len(lines) > 1)


def row_groups(prepared, gfx):
    """Blocks side by side that start on the same line and form a table row: one-line cells, or cells with a rule or box
    running across the whole row. They are set from one top and end at one bottom. Paragraphs of text columns stay
    independent, so a column never waits for its neighbour."""
    def ruled(members):
        x0, x1 = min(m["x0"] for m in members), max(m["x1"] for m in members)
        y0, y1 = min(m["y0"] for m in members), max(m["y1"] for m in members)
        slack = max(m["tol"] for m in members)
        return any(r[0] <= x0 + 2 and r[2] >= x1 - 2 and (y0 - slack <= r[1] <= y0 + slack or y1 - slack <= r[3] <= y1 + slack)
                   for r in gfx)

    def one_line(m):
        return len(m["b"]["lines"]) == 1

    def same_line(p, q):
        return abs(p["y0"] - q["y0"]) <= max(p["tol"], q["tol"])

    def same_band(p, q):
        tol = max(p["tol"], q["tol"])
        return p["band"] is not None and q["band"] is not None and all(abs(a - b) <= tol for a, b in zip(p["band"], q["band"]))

    levels = defaultdict(set)
    flat = [p for p in prepared if not p["pinned"] and not p["image"] and len(p["lines"]) <= C.row_lines]
    for i, p in enumerate(flat):
        for q in flat[i + 1:i + 1 + 4 * C.row_search]:
            if q["y0"] - p["y0"] > max(p["tol"], q["tol"]):
                break
            if abs(q["y0"] - p["y0"]) <= max(p["tol"], q["tol"]) and (q["x0"] >= p["x1"] - 1 or q["x1"] <= p["x0"] + 1):
                levels[tuple(sorted((round(p["x0"] / C.table_step), round(q["x0"] / C.table_step))))].add(round(p["y0"] / C.table_step))
    tabular = {k for k, v in levels.items() if len(v) >= C.table_rows}

    def columns_repeat(m, p):
        return tuple(sorted((round(m["x0"] / C.table_step), round(p["x0"] / C.table_step)))) in tabular

    groups = []
    for p in prepared:
        if p["pinned"] or p["image"] or len(p["lines"]) > C.row_lines or p["b"].get("at") is not None or p["b"].get("moved") or p["on"] is not None:
            groups.append([p])
            continue
        for gr in reversed(groups[-C.row_search:]):
            q = gr[0]
            if (not q["pinned"] and not q["image"] and len(q["lines"]) <= C.row_lines and q["k"] == p["k"]
                    and all(p["x0"] >= m["x1"] - 1 or p["x1"] <= m["x0"] + 1 for m in gr)
                    and (same_band(p, q) or same_line(p, q) and ((one_line(p) and all(one_line(m) for m in gr)) or ruled(gr + [p])
                                                                 or all(columns_repeat(m, p) for m in gr)))):
                gr.append(p)
                break
        else:
            groups.append([p])
    return groups


def overlay_top(p, done):
    """Top of text that lies on a picture: it keeps its place on the picture, wherever the picture went.
    None when the picture was not set: the text has no place without it."""
    a = next((a for a in done if a["id"] == p["on"]), None)
    return None if a is None else a["ny0"] + (p["y0"] - a["oy0"]) * a["scale"]


def place(blocks, page_w, g, typesetter, role_of, px_of, sc, limit, cols, gfx, rules, gap_cap, memo=None):
    order = sorted(blocks, key=lambda b: (b["bbox"].y0, b["bbox"].x0))
    memo = {} if memo is None else memo

    def prepared_of(b):
        key = (id(b), tuple(b["bbox"]), b.get("at"), b.get("skip"), b.get("count"))
        if key not in memo:
            memo[key] = prepare(b, blocks, page_w, g, typesetter, role_of, px_of, cols)
        return dict(memo[key]) if memo[key] else None

    prepared = [p for p in (prepared_of(b) for b in order) if p]
    pictures_ = [q for q in prepared if q["image"]]
    for p in prepared:
        p["on"] = None
        area = max((p["x1"] - p["x0"]) * (p["y1"] - p["y0"]), 1.0)
        below = [q for q in pictures_ if q is not p and (q["x1"] - q["x0"]) * (q["y1"] - q["y0"]) > area]
        if below:
            q, ov = max(((q, max(min(p["x1"], q["x1"]) - max(p["x0"], q["x0"]), 0) * max(min(p["y1"], q["y1"]) - max(p["y0"], q["y0"]), 0))
                         for q in below), key=lambda t: t[1])
            if ov >= C.overlay_share * area:
                p["on"], p["y_order"] = id(q["b"]), max(p["y0"], q["y0"]) + 1e-3
    prepared.sort(key=lambda p: (p.get("y_order", p["y0"]), p["x0"]))
    items, done, rects, glyphs, pics, used = [], [], [], [], [], []
    state = dict(truncated=False)

    def emit(p, top):
        pinned = p["pinned"]
        if p["image"]:
            scale = 1.0 if pinned else min(1.0, (limit - top) / p["h"])
            if scale < C.min_pic_scale:
                state["truncated"] = True
                return None
            h = p["h"] * scale
            pics.append((p["x0"], top, dict(p["b"], scale=scale)))
            return dict(x0=p["x0"], x1=p["x1"], oy0=p["y0"], oy1=p["y1"], ny0=top, ny1=p["y1"] if pinned else top + h,
                        pinned=pinned, src_lines=[(p["y0"], p["y1"])], n=1, pitch=h, id=id(p["b"]), scale=scale)
        lines, pitch, asc, desc = p["lines"], p["pitch"], p["asc"], p["desc"]
        if p["b"].get("count"):
            lines = lines[p["b"].get("skip", 0):p["b"].get("skip", 0) + p["b"]["count"]]
        if not pinned:
            room = min(limit - top, p["b"].get("fit", float("inf")))
            fit = max(0, int(room // pitch))
            state["truncated"] = state["truncated"] or fit < len(lines)
            lines = lines[:fit]
            if not lines:
                return None
        b, W, x0, rtl = p["b"], p["W"], p["bx0"], p["rtl"]
        height = max(len(lines) * pitch, 0.0 if p["multi"] else (p["y1"] - p["y0"]) * sc)
        y = top + pitch / 2 + (asc - desc) / 2
        for li, ln in enumerate(lines):
            lw = sum(w for _, _, w, _ in ln) + sum(sw for _, _, _, sw in ln[1:])
            ind_l = p["ind"][0] if li == 0 else p["ind"][1]
            extra = 0.0
            a = p["align"]
            if a == "justify" and (li == len(lines) - 1 or len(ln) < 2):
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
            if rtl and p["align"] == "justify" and a == "justify":
                x = x0
            start = x
            for t, f, w, sw in (visual(ln) if rtl else ln):
                x += 0.0 if rtl else sw
                if t["path"] is None:
                    glyphs.append((x, top + C.ascent_share * t["glyph"][1].height * t["px"] / t["size"] if t["tall"] else y, t))
                else:
                    items.append((x, y, f, t["text"], t["gray"], "rtl" if t["rtl"] else "ltr"))
                    used.append((t["sig"], t["path"], t["wght"], round(t["px"], 1)))
                x += w + (sw if rtl else 0.0) + extra
            rects.append(dict(x0=start, x1=x - extra, y0=y - asc, y1=y + desc, block=len(done), src=(p["bx0"], p["bx0"] + W),
                              pinned=pinned, align=a, line=li, lines=len(lines), tokens=len(ln)))
            y += pitch
        return dict(x0=p["x0"], x1=p["x1"], oy0=p["y0"], oy1=p["y1"], ny0=top, ny1=p["y1"] if pinned else top + height,
                    pinned=pinned, src_lines=[(l["bbox"].y0 * g, l["bbox"].y1 * g) for l in b["lines"]], n=len(lines), pitch=pitch,
                    id=id(b), total=len(p["lines"]))

    rules = sorted(rules, key=lambda r: r[2])
    ys = [r[2] for r in rules]
    for p in prepared:
        p["band"] = _band(rules, ys, p)
    for members in row_groups(prepared, gfx):
        tops = [m["b"]["at"] if m["b"].get("at") is not None else overlay_top(m, done) if m["on"] is not None
                else follow_top(m["x0"], m["x1"], m["y0"], m["pinned"], m["k"], done, gfx, gap_cap, sc) for m in members]
        top = max((t for t in tops if t is not None), default=0.0)
        entries = []
        for m, t in zip(members, tops):
            if t is None:
                continue
            e = emit(m, t if m["pinned"] or len(members) == 1 else top)
            if e:
                entries.append(e)
                done.append(e)
        if len(entries) > 1:
            bottom, src_bottom = max(e["ny1"] for e in entries), max(e["oy1"] for e in entries)
            for e in entries:
                e["ny1"], e["oy1"] = bottom, src_bottom
    wrapped = sum(1 for p in prepared if not p["image"] and p["wrapped"])
    return dict(items=items, anchors=done, rects=rects, glyphs=glyphs, pics=pics, used=used, truncated=state["truncated"],
                wrapped=wrapped)


class Mapper:
    """New y of a source y at an x range. Inside a block it stays on the block's line; in a gap it is placed in the same
    relation between the blocks that bound the gap in the range (above: the lowest bottom of the last row above, below:
    the highest top of the first row below), so a rule keeps its place between the cells of the rows it separates."""

    def __init__(self, anchors, sc):
        self.sc = sc
        self.anchors = [a for a in anchors if not a["pinned"]]
        self.memo = {}

    def __call__(self, xa, xb, y):
        key = (round(xa), round(xb), round(y, 1))
        if key not in self.memo:
            self.memo[key] = self._map(xa, xb, y)
        return self.memo[key]

    def _map(self, xa, xb, y):
        over = [a for a in self.anchors if a["x0"] < xb + 1 and a["x1"] > xa - 1]
        within = [a for a in over if a["oy0"] <= y <= a["oy1"]]
        if within:
            return _inside(max(within, key=lambda a: min(xb, a["x1"]) - max(xa, a["x0"])), y)
        above = [a for a in over if a["oy1"] < y]
        below = [a for a in over if a["oy0"] > y]
        last = max((a["oy1"] for a in above), default=None)
        first = min((a["oy0"] for a in below), default=None)
        row_above = [a for a in above if a["oy1"] >= last - a["pitch"] / self.sc] if above else []
        row_below = [a for a in below if a["oy0"] <= first + a["pitch"] / self.sc] if below else []
        bottom = max((a["ny1"] for a in row_above), default=None)
        top = min((a["ny0"] for a in row_below), default=None)
        if bottom is None:
            return y if top is None else top - (first - y) * self.sc
        if top is None:
            return bottom + (y - last) * self.sc
        return bottom + (y - last) / (first - last) * (top - bottom)


def map_y(anchors, sc, xa, xb, y):
    return Mapper(anchors, sc)(xa, xb, y)


def _inside(a, y):
    """A source y within a block, kept on the same line (or the same gap between lines) of the reset block."""
    src, n = a["src_lines"], a["n"]
    for j, (ly0, ly1) in enumerate(src):
        if y <= ly1:
            if y >= ly0:
                return a["ny0"] + min(int(j * n / len(src)), n - 1) * a["pitch"] + (y - ly0) / max(ly1 - ly0, 1e-6) * a["pitch"]
            return a["ny0"] + round(j * n / len(src)) * a["pitch"]
    return a["ny1"]


def _paths(d):
    """Point lists of one drawing, one per subpath."""
    paths, cur = [], []
    for it in d["items"]:
        op = it[0]
        if op in ("l", "c"):
            p1 = it[1]
            if cur and (abs(cur[-1][0] - p1.x) > 0.01 or abs(cur[-1][1] - p1.y) > 0.01):
                paths.append(cur)
                cur = []
            cur = cur or [(p1.x, p1.y)]
            if op == "l":
                cur.append((it[2].x, it[2].y))
            else:
                c1, c2, p2 = it[2:5]
                for k in range(1, C.bezier_steps + 1):
                    t = k / C.bezier_steps
                    u = 1 - t
                    cur.append((u ** 3 * p1.x + 3 * u * u * t * c1.x + 3 * u * t * t * c2.x + t ** 3 * p2.x,
                                u ** 3 * p1.y + 3 * u * u * t * c1.y + 3 * u * t * t * c2.y + t ** 3 * p2.y))
        else:
            if cur:
                paths.append(cur)
                cur = []
            if op == "re":
                r = it[1]
                paths.append([(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)])
            else:
                q = it[1]
                paths.append([(q.ul.x, q.ul.y), (q.ur.x, q.ur.y), (q.lr.x, q.lr.y), (q.ll.x, q.ll.y)])
    if cur:
        paths.append(cur)
    return paths


def _fill(img, pts, gray, alpha):
    """Antialiased polygon: each shape blends by its own coverage, so adjoining rows of one table leave their thin seam."""
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, y0 = max(int(min(xs)) - 1, 0), max(int(min(ys)) - 1, 0)
    x1, y1 = min(int(max(xs)) + 2, img.width), min(int(max(ys)) + 2, img.height)
    if x1 <= x0 or y1 <= y0:
        return
    big = Image.new("L", ((x1 - x0) * C.aa, (y1 - y0) * C.aa), 0)
    ImageDraw.Draw(big).polygon([((x - x0) * C.aa, (y - y0) * C.aa) for x, y in pts], fill=int(255 * alpha))
    img.paste(gray, (x0, y0), big.resize((x1 - x0, y1 - y0), Image.BOX))


def _key(r):
    return tuple(round(v / 2) for v in (r.x0, r.y0, r.x1, r.y1))


def page_frame(r, ph, pw):
    """A bar along the page edge that spans most of the page: a frame, kept whole where it is."""
    tall = r.height >= C.frame_span * ph and r.width <= C.frame_thin * pw and (r.x0 < C.edge_zone * pw or r.x1 > (1 - C.edge_zone) * pw)
    wide = r.width >= C.frame_span * pw and r.height <= C.frame_thin * ph and (r.y0 < C.edge_zone * ph or r.y1 > (1 - C.edge_zone) * ph)
    return tall or wide


def graphics(doc, number, dy, first, ph, pw, home=None, below=None):
    """Vector drawings of a page as (kind, data, y offset, page, fixed), in source points. On the following pages,
    what repeats from the home page (running headers, footers, frames) is left out."""
    out = []
    repeated = set() if first else {_key(d["rect"]) for d in doc[home].get_drawings()}
    for d in doc[number].get_drawings():
        r = d["rect"]
        fixed = r.y1 < C.header_pt or r.y0 > ph - C.footer_pt or page_frame(r, ph, pw) or (below is not None and r.y0 >= below)
        if not first and (fixed or _key(r) in repeated or r.width * r.height > C.repeat_cover * pw * ph):
            continue
        out.append(("draw", d, dy, number, fixed and first))
    return out


def pictures(doc, number, dy, first, ph, pw, home=None, below=None):
    """Pictures of a page as blocks of the flow, in source points; pictures that only repeat the page's text are left out."""
    page = doc[number]
    words = [pymupdf.Rect(w[:4]) for w in page.get_text("words")]
    out = []
    repeated = set() if first else {_key(pymupdf.Rect(im["bbox"])) for im in doc[home].get_image_info()}
    for im in page.get_image_info():
        r = pymupdf.Rect(im["bbox"])
        fixed = r.y1 < C.header_pt or r.y0 > ph - C.footer_pt or (below is not None and r.y0 >= below)
        if r.width * r.height > C.fixed_cover * pw * ph or (not first and (fixed or _key(r) in repeated)):
            continue
        if sum((w & r).get_area() for w in words) >= C.text_image_share * r.get_area():
            continue
        out.append(dict(image=True, rect=r, number=number, bbox=r + (0, dy, 0, dy), pinned=fixed and first, k=0 if first else 1))
    return out


def _hrules(elems, g):
    """Horizontal rules of the graphics as (x0, x1, y) in output px: the lines that make the rows of a table."""
    return [(x0 * g, x1 * g, (y + dy) * g) for kind, d, dy, number, fixed in elems if kind == "draw"
            for x0, y, x1 in _horizontal(d)]


def _vrules(elems, g):
    """Vertical rules of the graphics as (x, y0, y1) in output px: cell borders and frame sides a text must not cross."""
    out = []
    for kind, d, dy, number, fixed in elems:
        if kind != "draw":
            continue
        for path in _paths(d):
            for (xa, ya), (xb, yb) in zip(path, path[1:] + ([path[0]] if d.get("closePath") else [])):
                if abs(xa - xb) < 0.5 and abs(ya - yb) >= C.rule_min_pt:
                    out.append((xa * g, (min(ya, yb) + dy) * g, (max(ya, yb) + dy) * g))
        r = d["rect"]
        if d.get("fill") is not None and r.width <= 2 and r.height >= C.rule_min_pt:
            out.append((r.x0 * g, (r.y0 + dy) * g, (r.y1 + dy) * g))
    return out


def _band(rules, ys, p):
    """The rules above and below a block that run along all of its width: the row of a table it sits in, or None."""
    def covers(r):
        return r[0] <= p["x0"] + 2 and r[1] >= p["x1"] - 2
    up = next((rules[k][2] for k in range(bisect.bisect_right(ys, p["y0"] + p["tol"]) - 1, -1, -1) if covers(rules[k])), None)
    down = next((rules[k][2] for k in range(bisect.bisect_left(ys, p["y1"] - p["tol"]), len(rules)) if covers(rules[k])), None)
    return None if up is None or down is None or down <= up else (up, down)


def _gfx_rects(elems, g, page_area):
    out = []
    for kind, data, dy, number, fixed in elems:
        r = data["rect"]
        white = data.get("fill") == (1.0, 1.0, 1.0) and data.get("color") is None
        if not white and r.width * r.height * g * g < 0.5 * page_area:
            out.append((r.x0 * g, (r.y0 + dy) * g, r.x1 * g, (r.y1 + dy) * g))
    return out


def text_free(doc, number):
    cache = doc.__dict__.setdefault("_text_free", {})
    if number not in cache:
        cache[number] = _text_free(doc, number)
    return cache[number]


def _text_free(doc, number):
    one = pymupdf.open()
    one.insert_pdf(doc, from_page=number, to_page=number)
    page = one[0]
    pictures_ = [pymupdf.Rect(im["bbox"]) for im in page.get_image_info()]
    for b in page.get_text("dict", flags=TEXT_FLAGS)["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                if any(pymupdf.Rect(s["bbox"]).intersects(r) for r in pictures_):
                    page.add_redact_annot(pymupdf.Rect(s["bbox"]))
    page.apply_redactions(images=0, graphics=0)
    for r in pictures_:
        page.add_redact_annot(r)
    page.apply_redactions(images=0, graphics=2)
    return page


def touching_ranges(elems, g):
    """x range (px) shared by drawings that touch each other (the lines of one table or box), keyed by drawing id."""
    ds = [(e[1], e[2]) for e in elems if not e[4]]
    parent = list(range(len(ds)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    order = sorted((i for i in range(len(ds)) if ds[i][0]["rect"].get_area() < C.background_share * C.page_area_pt), key=lambda i: ds[i][0]["rect"].x0)
    for a, i in enumerate(order):
        ri, dyi = ds[i][0]["rect"], ds[i][1]
        for j in order[a + 1:]:
            rj, dyj = ds[j][0]["rect"], ds[j][1]
            if rj.x0 > ri.x1 + C.touch_pt:
                break
            if rj.y0 + dyj <= ri.y1 + dyi + C.touch_pt and ri.y0 + dyi <= rj.y1 + dyj + C.touch_pt:
                parent[find(i)] = find(j)
    span, size = {}, defaultdict(int)
    for i, (d, dy) in enumerate(ds):
        root = find(i)
        size[root] += 1
        lo, hi = span.get(root, (1e9, -1e9))
        span[root] = (min(lo, d["rect"].x0 * g), max(hi, d["rect"].x1 * g))
    return {id(d): span[find(i)] for i, (d, dy) in enumerate(ds) if size[find(i)] > 1}, {id(d): find(i) for i, (d, dy) in enumerate(ds)}


def drawing_paths(d, dy, fixed, anchors, sc, g, spans=None):
    """Subpaths of a drawing in output px, carried by the blocks it sits with. Every point of a path takes the same
    shift, so rules stay straight; a small shape (logo, bullet, checkbox) moves as a whole instead of stretching."""
    r = d["rect"]
    shared = (spans or {}).get(id(d))
    rigid = r.width <= C.rigid_pt and r.height <= C.rigid_pt and shared is None
    mp = anchors if isinstance(anchors, Mapper) else Mapper(anchors, sc)
    out = []
    for path in _paths(d):
        xa, xb = shared or (min(x for x, _ in path) * g, max(x for x, _ in path) * g)
        if fixed:
            out.append([(x * g, (y + dy) * g) for x, y in path])
        elif rigid:
            top = mp(xa, xb, (r.y0 + dy) * g)
            out.append([(x * g, top + (y - r.y0) * g) for x, y in path])
        else:
            out.append([(x * g, mp(xa, xb, (y + dy) * g)) for x, y in path])
    return out


def visible(elems, anchors, sc, g, cutoff, spans, roots, limit):
    """Drawings that are drawn: not those below the last placed text, nor boxes of the following pages cut by the page bottom."""
    cut = set()
    anchors = anchors if isinstance(anchors, Mapper) else Mapper(anchors, sc)
    for kind, d, dy, number, fixed in elems:
        if not fixed and dy > 0 and any(y > limit for pts in drawing_paths(d, dy, fixed, anchors, sc, g, spans) for _, y in pts):
            cut.add(roots[id(d)])
    return [e for e in elems if e[4] or ((e[1]["rect"].y0 + e[2]) * g <= cutoff and not (e[2] > 0 and roots[id(e[1])] in cut))]


def draw_graphics(img, elems, anchors, sc, g, cutoff, spans, roots, limit):
    for kind, d, dy, number, fixed in visible(elems, anchors, sc, g, cutoff, spans, roots, limit):
        fill, stroke = d.get("fill"), d.get("color")
        alpha = 1.0 if d.get("fill_opacity") is None else d["fill_opacity"]
        width = max(1, round((d.get("width") or 1.0) * g))
        for pts in drawing_paths(d, dy, fixed, anchors, sc, g, spans):
            if fill is not None and alpha > C.min_alpha and len(pts) > 2:
                _fill(img, pts, _gray_f(fill), alpha)
            if stroke is not None and len(pts) > 1:
                ImageDraw.Draw(img).line(pts + ([pts[0]] if d.get("closePath") else []), fill=_gray_f(stroke), width=width)


def text_page(page):
    """False for scanned or picture pages: little text, or pictures covering most of the page."""
    chars = sum(len(s["text"].strip()) for l in parsed(page, 0.0)[0] for s in l["spans"])
    image = sum((pymupdf.Rect(im["bbox"]) & page.rect).get_area() for im in page.get_image_info())
    return chars >= C.min_text_chars and image <= C.max_image_share * page.rect.get_area()


def running_bands(blocks):
    """Vertical positions of the header and footer lines of a page, in C.run_step bins."""
    return {round(b["bbox"].y0 / C.run_step) for b in blocks if not b.get("image") and b["edge"]}


def _rule_boxes(d):
    """Boxes (x0, y0, x1, y1) of the straight lines and rectangles of a drawing."""
    boxes = []
    for it in d["items"]:
        if it[0] == "l":
            (xa, ya), (xb, yb) = (it[1].x, it[1].y), (it[2].x, it[2].y)
            boxes.append((min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb)))
        elif it[0] == "re":
            boxes.append((it[1].x0, it[1].y0, it[1].x1, it[1].y1))
    return boxes


def line_rules(page, dy):
    """Vertical lines (x, y0, y1) and horizontal lines (y, x0, x1) of a page: cell borders and rules."""
    v, h = [], []
    for x0, y0, x1, y1 in (b for d in page.get_drawings() for b in _rule_boxes(d)):
        if x1 - x0 <= C.rule_thin and y1 - y0 >= C.rule_long:
            v.append(((x0 + x1) / 2, y0 + dy, y1 + dy))
        elif y1 - y0 <= C.rule_thin and x1 - x0 >= C.rule_long:
            h.append(((y0 + y1) / 2 + dy, x0, x1))
    return v, h


def _cut(segs, walls):
    """Segments cut where a vertical rule runs between two of their spans: the cells of a table are not one line."""
    out = []
    for l in segs:
        groups = [[l["spans"][0]]]
        for s in l["spans"][1:]:
            prev = groups[-1][-1]["bbox"]
            lo, hi = min(prev.x1, s["bbox"].x1), max(prev.x0, s["bbox"].x0)
            cut = any(lo - 1 <= x <= hi + 1 and y0 < l["bbox"].y1 and y1 > l["bbox"].y0 for x, y0, y1 in walls)
            groups.append([s]) if cut else groups[-1].append(s)
        out += [dict(l, bbox=pymupdf.Rect(min(s["bbox"].x0 for s in grp), l["bbox"].y0, max(s["bbox"].x1 for s in grp), l["bbox"].y1), spans=grp)
                for grp in groups]
    return out


def parsed(page, dy):
    """Lines and rules of a page, parsed once per document and moved down by dy (fresh copies: callers change them)."""
    cache = page.parent.__dict__.setdefault("_refont", {})
    if page.number not in cache:
        v, h = line_rules(page, 0.0)
        cache[page.number] = (_cut(extract_segments(page, 0.0), v), (v, h))
    segs, (v, h) = cache[page.number]
    moved = [dict(l, bbox=pymupdf.Rect(l["bbox"]) + (0, dy, 0, dy),
                  spans=[dict(s, bbox=pymupdf.Rect(s["bbox"]) + (0, dy, 0, dy)) for s in l["spans"]]) for l in segs]
    return moved, ([(x, y0 + dy, y1 + dy) for x, y0, y1 in v], [(y + dy, x0, x1) for y, x0, x1 in h])


def _absorb(blocks):
    """A one-line block that lies inside a paragraph on one of its lines is a part of that line (the fields of a spaced
    line): its spans join the line, and the block goes."""
    gone = set()
    for o in blocks:
        if len(o["lines"]) != 1 or o.get("image"):
            continue
        for b in blocks:
            if b is o or id(b) in gone or len(b["lines"]) < 2 or (b["bbox"] & o["bbox"]).get_area() < C.contain_share * o["bbox"].get_area():
                continue
            line = next((l for l in b["lines"] if min(l["bbox"].y1, o["bbox"].y1) - max(l["bbox"].y0, o["bbox"].y0) > 0.5 * o["bbox"].height), None)
            if line:
                line["spans"] = sorted(line["spans"] + o["lines"][0]["spans"], key=lambda s: s["bbox"].x0)
                line["bbox"] = line["bbox"] | o["lines"][0]["bbox"]
                gone.add(id(o))
                break
    return [b for b in blocks if id(b) not in gone]


def page_blocks(page, dy, first, running=()):
    """Paragraphs of a page moved down by dy. Header and footer zones are those of the whole page, whatever its crop."""
    segs, rules = parsed(page, dy)
    blocks = _absorb(paragraphs(segs, rules))
    offset, full = page.cropbox_position.y - dy, page.mediabox.height
    for b in blocks:
        for l in b["lines"]:
            for sp in l["spans"]:
                sp["number"] = page.number
        top, bottom = b["bbox"].y0 + offset, b["bbox"].y1 + offset
        b["pinned"] = top > full - C.footer_pt or bottom < C.header_pt
        b["edge"] = top < C.run_zone * full or top > (1 - C.run_zone) * full
        b["k"] = 0 if first else 1

    def repeats(b):
        return b["edge"] and any(abs(round((b["bbox"].y0 - dy) / C.run_step) - r) <= 1 for r in running)

    return [b for b in blocks if first or not (b["pinned"] or repeats(b))]


def extent(blocks, dy):
    body = [b for b in blocks if not b["pinned"]]
    return min(b["bbox"].y0 for b in body) - dy, max(b["bbox"].y1 for b in body) - dy


def _columns(blocks):
    """Column intervals of the page's blocks; the whole text width when no block lies in a single column."""
    cols = sorted({b["colspan"] for b in blocks if not b["pinned"] and not b.get("image") and b.get("colspan")})
    if cols:
        return [list(c) for c in cols]
    xs = [(b["bbox"].x0, b["bbox"].x1) for b in blocks if not b["pinned"] and not b.get("image")]
    return [[min(a for a, _ in xs), max(z for _, z in xs)]] if xs else [[0.0, 1.0]]


def _horizontal(d):
    """Horizontal rules of a drawing as (x0, y, x1): thin fills and straight stroked edges."""
    segs = []
    if d.get("color") is not None:
        for path in _paths(d):
            for (xa, ya), (xb, yb) in zip(path, path[1:] + ([path[0]] if d.get("closePath") else [])):
                if abs(ya - yb) < 0.5 and abs(xa - xb) >= C.rule_min_pt:
                    segs.append((min(xa, xb), ya, max(xa, xb)))
    r = d["rect"]
    if d.get("fill") is not None and r.height <= 2 and r.width >= C.rule_min_pt:
        segs.append((r.x0, r.y0, r.x1))
    return segs


def rule_segments(elems, anchors, sc, g):
    """Horizontal rules as (x0, y, x1) in output px."""
    mp = anchors if isinstance(anchors, Mapper) else Mapper(anchors, sc)
    return [(x0 * g, (y + dy) * g if fixed else mp(x0 * g, x1 * g, (y + dy) * g), x1 * g)
            for kind, d, dy, number, fixed in elems if kind == "draw" for x0, y, x1 in _horizontal(d)]


def crossings(rects, rules):
    """Text lines a rule runs through the middle of."""
    n = 0
    for r in rects:
        h = r["y1"] - r["y0"]
        n += sum(1 for x0, y, x1 in rules if x0 < r["x1"] and x1 > r["x0"] and r["y0"] + C.cross_share * h < y < r["y1"] - C.cross_share * h)
    return n


def column_gap(done, cols, pitch, sc):
    """First gap a block left in a column by waiting for its neighbours: (block above, block below, column, size)."""
    for c0, c1 in cols:
        col = sorted((a for a in done if not a["pinned"] and a["n"]
                      and min(a["x1"], c1) - max(a["x0"], c0) >= C.gap_overlap * min(a["x1"] - a["x0"], c1 - c0)), key=lambda a: a["ny0"])
        for a, b in zip(col, col[1:]):
            gap = b["ny0"] - a["ny1"]
            if gap > C.gap_lines * pitch and gap > 2 * max(b["oy0"] - a["oy1"], 0.0) * sc + 2 * pitch:
                return a, b, (c0, c1), gap
    return None


def pour(res, blocks, cols, limit, g):
    """Text of the columns re-poured in reading order: the first columns are filled to the bottom, the last stay short.
    Used when there is no more text to fill every column."""
    done = {a["id"]: a for a in res["anchors"] if not a["pinned"] and a["n"]}
    members, region = [], 0.0
    for b in blocks:
        a = done.get(id(b))
        if a is None or b["pinned"]:
            continue
        x0, x1 = b["bbox"].x0 * g, b["bbox"].x1 * g
        best = max(range(len(cols)), key=lambda i: min(x1, cols[i][1]) - max(x0, cols[i][0]))
        share = (min(x1, cols[best][1]) - max(x0, cols[best][0])) / max(x1 - x0, 1.0)
        if share >= C.touch_share:
            members.append((best, b, a))
        else:
            region = max(region, a["ny1"])
    members = [m for m in members if m[2]["ny0"] >= region - 1]
    used = sorted({c for c, _, _ in members})
    widths = [cols[c][1] - cols[c][0] for c in used]
    if len(used) < 2 or max(widths) > C.width_match * min(widths):
        return False
    rtl = sum(b["rtl"] for _, b, _ in members if not b.get("image")) > sum(1 for _, b, _ in members if not b.get("image")) / 2
    order = sorted(used, reverse=rtl)
    tops = {c: min(a["ny0"] for cc, _, a in members if cc == c) for c in used}
    stream = []
    for c in order:
        col = sorted((m for m in members if m[0] == c), key=lambda m: m[2]["ny0"])
        for i, (cc, b, a) in enumerate(col):
            stream.append((b, a, max(col[i + 1][2]["ny0"] - a["ny1"], 0.0) if i + 1 < len(col) else 0.0, b["bbox"].x0 * g - cols[cc][0]))
    ci, y, extra = 0, tops[order[0]], []
    for b, a, gap, offset in stream:
        pitch = a["pitch"]
        n = a.get("total", a["n"])
        h = a["ny1"] - a["ny0"] if b.get("image") or n == a["n"] else n * pitch
        skip = 0
        while True:
            col = order[ci]
            room = limit - y
            height = h if skip == 0 else (n - skip) * pitch
            lines_left = n - skip
            if height <= room + 0.5 or ci == len(order) - 1:
                part = b if skip == 0 else dict(b)
                _assign(part, cols[col], g, y, skip, lines_left if skip else 0, b, offset)
                if skip:
                    extra.append(part)
                y += height + gap
                break
            fit = int(room // pitch) if not b.get("image") else 0
            if fit >= C.pour_min_lines and lines_left - fit >= C.pour_min_lines:
                part = b if skip == 0 else dict(b)
                _assign(part, cols[col], g, y, skip, fit, b, offset)
                if skip:
                    extra.append(part)
                skip += fit
            ci += 1
            y = tops[order[ci]]
    blocks.extend(extra)
    return True


def _assign(part, col, g, y, skip, count, original, offset):
    part.setdefault("src_bbox", pymupdf.Rect(original.get("src_bbox", original["bbox"])))
    width = part["bbox"].width
    x0 = (col[0] + offset) / g
    part["bbox"] = pymupdf.Rect(x0, y / g, x0 + width, y / g + part["bbox"].height)
    part["at"], part["skip"], part["count"] = y, skip, count


def flow(doc, number, first, typesetter, body, mult, target, sc_font, scale, cols_px, elems0, g, pw, ph):
    """Blocks of the page and of the following pages placed until every column is full; text of the following pages
    also fills gaps a block left in a column."""
    role_of = lambda sz: "head" if sz > body * C.head_ratio else "body"
    px_of = lambda sz: sz * sc_font * mult * ratio(sz, body, scale)
    sc = sc_font * mult / g
    gap_cap = C.gap_body * body * g
    elems = list(elems0)
    gfx = _gfx_rects(elems, g, pw * g * ph * g)
    rules = _hrules(elems, g)
    text_bottom = max((b["bbox"].y1 for b in first if not b["pinned"]), default=None)
    first = first + pictures(doc, number, 0.0, True, ph, pw, below=text_bottom)
    typesetter.dropped = 0
    typesetter.unresolved.clear()
    st = dict(limit=max(b["bbox"].y1 for b in first if not b["pinned"] and not b.get("image")) * g, used=1, dy=0.0,
              bottom=extent(first, 0.0)[1])
    blocks, memo = list(first), {}
    running = running_bands(first)

    def load_next():
        if st["used"] >= C.max_pages or number + st["used"] >= len(doc):
            return False
        nxt = doc[number + st["used"]]
        probe = page_blocks(nxt, 0.0, False, running) if text_page(nxt) else []
        if probe:
            pics = pictures(doc, number + st["used"], 0.0, False, ph, pw, number)
            top, bot = extent(probe + pics, 0.0)
            st["dy"] += st["bottom"] + body - top
            blocks.extend(page_blocks(nxt, st["dy"], False, running) + [dict(p, bbox=p["bbox"] + (0, st["dy"], 0, st["dy"])) for p in pics])
            st["bottom"] = bot
            st["limit"] = max(st["limit"], bot * g)
            more = graphics(doc, number + st["used"], st["dy"], False, ph, pw, number)
            elems.extend(more)
            gfx.extend(_gfx_rects(more, g, pw * g * ph * g))
            rules.extend(_hrules(more, g))
        st["used"] += 1
        return True

    def run():
        return place(blocks, pw, g, typesetter, role_of, px_of, sc, st["limit"], cols_px, gfx, rules, gap_cap, memo)

    res = run()
    slack = 2 * px_of(body)
    pitch = px_of(body) * C.line_pitch
    while True:
        full = all(any(a["ny1"] >= st["limit"] - slack and a["x0"] < c[1] and a["x1"] > c[0] for a in res["anchors"] if not a["pinned"])
                   for c in cols_px)
        if full or not load_next():
            break
        res = run()
    exhausted = not (st["used"] < C.max_pages and number + st["used"] < len(doc))
    if not full and exhausted and C.pour and not res["truncated"] and pour(res, blocks, cols_px, st["limit"], g):
        res = run()
        full = all(any(a["ny1"] >= st["limit"] - slack and a["x0"] < c[1] and a["x1"] > c[0] for a in res["anchors"] if not a["pinned"])
                   for c in cols_px)
    tried = set()
    for _ in range(C.max_fill):
        if C.pour and not full and exhausted:
            break
        gap = column_gap(res["anchors"], cols_px, pitch, sc)
        if gap is None:
            break
        above, below, (c0, c1), size = gap
        if (above["id"], below["id"]) in tried:
            break
        tried.add((above["id"], below["id"]))
        placed = {a["id"] for a in res["anchors"]}
        pool = lambda: sorted((b for b in blocks if b["k"] and not b.get("image") and not b["pinned"] and id(b) not in placed
                               and c0 - 2 <= (b["bbox"].x0 + b["bbox"].x1) * g / 2 <= c1 + 2), key=lambda b: b["bbox"].y0)
        cand = pool()
        while not cand and load_next():
            cand = pool()
        if not cand:
            break
        c = cand[0]
        top = above["oy1"] + 0.5 * g
        if top > below["oy0"] - C.fill_clearance * g:
            top = (above["oy0"] + below["oy0"]) / 2
        c["bbox"] = pymupdf.Rect(c["bbox"].x0, top / g, c["bbox"].x1, top / g + c["bbox"].height)
        c["fit"], c["moved"] = size, True
        res = run()
        full = all(any(a["ny1"] >= st["limit"] - slack and a["x0"] < c_[1] and a["x1"] > c_[0] for a in res["anchors"] if not a["pinned"])
                   for c_ in cols_px)
    done = res["anchors"]
    text_done = [a for a in done if not a["pinned"] and a["n"]]
    cutoff = max((a["oy1"] for a in text_done), default=0.0) + 2 if res["truncated"] else float("inf")
    return dict(res, elems=elems, limit=st["limit"], sc=sc, slack=slack, used_pages=st["used"], full=full, body_pt=target * mult,
                cutoff=cutoff)


def ratio(size, body, scale):
    """Factor of a font size: the sizes that are not the body's (headings, notes) are scaled by `scale`."""
    return scale if size > body * C.head_ratio or size * C.head_ratio < body else 1.0


def layout(doc, number, rng, fonts, body_pt, scale, width_px):
    """Layout of a page with its body text at body_pt on the sheet (210 mm wide, width_px wide), the other sizes
    keeping their ratio to the body's times `scale`."""
    page = doc[number]
    pw, ph = page.rect.width, page.rect.height
    g = width_px / pw
    first = page_blocks(page, 0.0, True)
    cnt = Counter()
    for b in first:
        for l in b["lines"]:
            for s in l["spans"]:
                cnt[round(s["size"] * 2) / 2] += len(s["text"])
    body = cnt.most_common(1)[0][0]
    target = body_pt
    sc_font = target * (width_px / C.sheet_pt_w) / body
    chars = {c for b in first for l in b["lines"] for s in l["spans"] for c in s["text"] if not c.isspace()}
    need = {(s["bold"], s["italic"]) for b in first for l in b["lines"] for s in l["spans"] if not s["mono"]}
    cols_px = [(c[0] * g, c[1] * g) for c in _columns(first)]
    elems0 = graphics(doc, number, 0.0, True, ph, pw, below=max((b["bbox"].y1 for b in first if not b["pinned"]), default=None))
    role_of = lambda sz: "head" if sz > body * C.head_ratio else "body"
    px_of = lambda sz: sz * sc_font * ratio(sz, body, scale)
    best = None
    for attempt in range(C.font_tries):
        typesetter = Typesetter(fonts, rng, chars, need)
        fit = max((p["fit"] for p in (prepare(b, first, pw, g, typesetter, role_of, px_of, cols_px) for b in first if not b.get("image") and len(b["lines"]) == 1) if p),
                  default=0.0)
        if best is None or fit < best[0]:
            best = (fit, typesetter)
        if fit <= 1.0:
            break
    fit, typesetter = best
    mult = max(min(1.0, 1.0 / fit) if fit else 1.0, C.fit_floor)
    lay = flow(doc, number, first, typesetter, body, mult, target, sc_font, scale, cols_px, elems0, g, pw, ph)
    info = dict(head=typesetter.role["head"], body=typesetter.role["body"], mono=typesetter.role["mono"],
                body_pt=round(target * mult, 1), body_src_pt=body, pages=lay["used_pages"], dropped=typesetter.dropped,
                full=lay["full"], wrapped=lay["wrapped"],
                unresolved={f"U+{ord(c):04X} {unicodedata.name(c, '?')}": n for c, n in typesetter.unresolved.items()})
    spans, roots = touching_ranges(lay["elems"], g)
    return dict(items=lay["items"], glyphs=lay["glyphs"], pics=lay["pics"], tokens=lay["used"], anchors=lay["anchors"],
                rects=lay["rects"], elems=lay["elems"], spans=spans, roots=roots, cols=cols_px, limit=lay["limit"], sc=lay["sc"], g=g, cutoff=lay["cutoff"],
                size=(width_px, round(ph * g)), info=info, slack=lay["slack"],
                role_paths={f[0] for k in typesetter.role.values() for f in (typesetter.prose.get(k) or typesetter.mono[k]).values()})


def needs_shaping(text):
    """Scripts whose letters join, reorder or stack (Hebrew to Indic, Thai, Khmer, Arabic forms) are set by the shaping engine."""
    return any(0x0590 <= ord(c) <= 0x1CFF or 0xA800 <= ord(c) <= 0xABFF or 0xFB1D <= ord(c) <= 0xFEFF or c in JOINERS for c in text)


_GLYPHS = {}
_NEW = {}
_GLYPH_DIR = Path("~/.cache/dspages").expanduser()
_GLYPH_FILE = _GLYPH_DIR / "glyphs.pkl"


def load_glyphs():
    """Glyph bitmaps of earlier runs (the merged file), by (font file, size, weight, letter)."""
    if _GLYPH_FILE.exists():
        with open(_GLYPH_FILE, "rb") as fh:
            _GLYPHS.update(pickle.load(fh))


def save_glyphs():
    """The glyphs this process made, in a file of its own; merge_glyphs joins them into the cache."""
    if _NEW:
        _GLYPH_DIR.mkdir(parents=True, exist_ok=True)
        with open(_GLYPH_DIR / f"glyphs.{os.getpid()}.part", "wb") as fh:
            pickle.dump(_NEW, fh)
        _NEW.clear()


def merge_glyphs():
    """The cache file and the files of the processes joined into the one cache file."""
    parts = sorted(_GLYPH_DIR.glob("glyphs.*.part"))
    load_glyphs()
    for f in parts:
        with open(f, "rb") as fh:
            _GLYPHS.update(pickle.load(fh))
    with open(_GLYPH_FILE.with_suffix(".tmp"), "wb") as fh:
        pickle.dump(_GLYPHS, fh)
    os.replace(_GLYPH_FILE.with_suffix(".tmp"), _GLYPH_FILE)
    for f in parts:
        f.unlink()


def _bitmap(f, text, **kw):
    mask, (ox, oy) = f.getmask2(text, "L", anchor="ls", **kw)
    w, h = mask.size
    return (np.array(mask, dtype=np.uint8).reshape(h, w) if w and h else np.zeros((0, 0), np.uint8)), ox, oy


def _glyph(f, ch):
    key = f.key + (ch,)
    if key not in _GLYPHS:
        _GLYPHS[key] = _NEW[key] = (*_bitmap(f, ch), f.getlength(ch))
    return _GLYPHS[key]


def _word(f, text, direction):
    key = f.key + (text, direction)
    if key not in _GLYPHS:
        _GLYPHS[key] = _NEW[key] = (*_bitmap(f, text, direction=direction), 0.0)
    return _GLYPHS[key]


def blit_text(img, items):
    """Text laid on the page from cached coverage with a darken blend: letter by letter for scripts without shaping, whole
    words (shaped as a unit) for the others."""
    arr = np.asarray(img).copy()
    H, W = arr.shape
    for x, y, f, text, gray, direction in items:
        for ch in ([text] if needs_shaping(text) else text):
            cov, ox, oy, adv = _word(f, ch, direction) if needs_shaping(text) else _glyph(f, ch)
            h, w = cov.shape
            x0, y0 = int(round(x)) + ox, int(round(y)) + oy
            if w and h and x0 < W and y0 < H and x0 + w > 0 and y0 + h > 0:
                cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
                part = cov[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
                ink = 255 - (part.astype(np.uint16) * (255 - gray) // 255).astype(np.uint8)
                np.minimum(arr[cy0:cy1, cx0:cx1], ink, out=arr[cy0:cy1, cx0:cx1])
            x += adv
    img.paste(Image.fromarray(arr))


def render(doc, lay):
    img = Image.new("L", lay["size"], 255)
    cleans = {}
    g = lay["g"]
    anchors = Mapper(lay["anchors"], lay["sc"])

    def paste_pictures(pinned):
        for x, top, b in lay["pics"]:
            if b["pinned"] == pinned:
                if b["number"] not in cleans:
                    cleans[b["number"]] = text_free(doc, b["number"])
                k = g * b.get("scale", 1.0)
                pm = cleans[b["number"]].get_pixmap(matrix=pymupdf.Matrix(k, k), clip=b["rect"], colorspace=pymupdf.csGRAY)
                img.paste(Image.frombytes("L", (pm.w, pm.h), pm.samples), (round(x), round(top)))

    draw_graphics(img, [e for e in lay["elems"] if not e[4]], anchors, lay["sc"], g, lay["cutoff"], lay["spans"], lay["roots"], lay["limit"])
    paste_pictures(False)
    ImageDraw.Draw(img).rectangle([0, min(lay["limit"], img.height), img.width, img.height], fill=255)
    draw_graphics(img, [e for e in lay["elems"] if e[4]], anchors, lay["sc"], g, lay["cutoff"], lay["spans"], lay["roots"], lay["limit"])
    paste_pictures(True)
    blit_text(img, lay["items"])
    for x, y, t in lay["glyphs"]:
        number, rect = t["glyph"]
        k = t["px"] / t["size"]
        pm = doc[number].get_pixmap(matrix=pymupdf.Matrix(k, k), clip=rect, colorspace=pymupdf.csGRAY)
        img.paste(Image.frombytes("L", (pm.w, pm.h), pm.samples), (round(x), round(y - C.ascent_share * pm.h)))
    return img


def content_box(page):
    """Rectangle of what the page shows: text lines, pictures and graphics (a background fill over the page does not count)."""
    boxes = [l["bbox"] for l in extract_segments(page, 0.0)] + [pymupdf.Rect(i["bbox"]) for i in page.get_image_info()]
    boxes += [d["rect"] for d in page.get_drawings() if d["rect"].get_area() < C.background_share * page.rect.get_area()]
    box = pymupdf.Rect()
    for b in boxes:
        box |= b
    return box & page.rect if box.is_valid and not box.is_empty else page.rect


def sheet(path, number, seed, fonts, body_pt, scale, margin, size):
    """The content of a PDF page restyled on a sheet of size (width, height) px with a blank margin (share of the sheet's
    sides) all around: -> (uint8 sheet, layout info). The same seed gives the same sheet."""
    sw, sh = size
    mx, my = round(margin * sw), round(margin * sh)
    with pymupdf.open(path) as doc:
        box = content_box(doc[number])
        for k in range(number, min(number + C.max_pages, len(doc))):
            doc[k].set_cropbox(box & doc[k].mediabox)
        w = min(sw - 2 * mx, round((sh - 2 * my) * box.width / box.height))
        lay = layout(doc, number, np.random.default_rng(seed), fonts, body_pt, scale, w)
        img = render(doc, lay)
    out = Image.new("L", size, 255)
    out.paste(img, (mx + (sw - 2 * mx - img.width) // 2, my))
    return np.asarray(out), lay["info"]


def garbled(segs, latin):
    """Text that did not come out as text of one script (a font without a text map): mixed scripts, private-use or control codes.
    Latin letters in a page of a language that is not written in Latin are such text too (a legacy font's own codes)."""
    text = "".join(s["text"] for l in segs for s in l["spans"])
    letters = [c for c in text if c.isalpha()]
    marks = [c for c in text if not c.isspace()]
    if not letters or not marks:
        return True
    name, top = Counter(_script(c) for c in letters).most_common(1)[0]
    private = sum(unicodedata.category(c) in ("Co", "Cc") for c in marks)
    return (name == "latin" and not latin) or top < C.script_purity * len(letters) or private > C.private_share * len(marks)


def over_pictures(page, segs):
    """Share of the text lines that lie inside a picture: a scan with its recognized text hidden under it, not set text."""
    images = [pymupdf.Rect(im["bbox"]) for im in page.get_image_info()]
    return sum(any(l["bbox"] in im for im in images) for l in segs) / max(len(segs), 1)


def usable(page, latin=True):
    """A page of text set in the PDF: a body (not only header and footer), text of one script (Latin only if the language
    is written in it), no more than
    max_columns columns, no scan with a hidden text layer."""
    segs = parsed(page, 0.0)[0]
    body = [l for l in segs if l["bbox"].y1 >= C.header_pt and l["bbox"].y0 <= page.rect.height - C.footer_pt]
    return (bool(body) and text_page(page) and not garbled(segs, latin) and len(column_intervals(segs)) <= C.max_columns
            and over_pictures(page, segs) <= C.over_picture_share)


def font_set(font_dirs):
    """find_fonts plus every other Noto family with real glyph coverage (Kufi, Naskh, Nastaliq, Rashi, ...)."""
    base = find_fonts(tuple(font_dirs))
    dirs = [os.path.expanduser(d) for d in font_dirs]
    extra = sorted(f for d in dirs if os.path.isdir(d) for ext in ("ttf", "otf") for f in glob.glob(f"{d}/**/*.{ext}", recursive=True)
                   if os.path.basename(f).lower().startswith("noto") and f not in set(base) and len(_font_cmap(f)) >= C.min_noto_cmap)
    return base + tuple(extra)
