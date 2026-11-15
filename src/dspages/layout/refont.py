"""Restyled pages: a digital-born PDF page set again in other fonts and scales. Text blocks keep the page's flow,
columns, alignment, bold/italic and heading/body hierarchy; rules, boxes and pictures follow the text they sit next to.
The text of the following pages flows in, so the page stays full. Fonts come from find_fonts, matched to the
page's scripts by the real glyph coverage of every font file that is drawn with."""

import bisect
import glob
import io
import os
import re
import statistics
import struct
import unicodedata
from collections import Counter, defaultdict
from functools import cache, lru_cache

import cv2
import numpy as np
import pymupdf
import regex
from fontTools import agl
from fontTools.ttLib import TTFont, TTLibError
from numba import njit
from PIL import Image, ImageDraw, ImageFont
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from ..config import RefontCfg
from . import indic, legacy, reffonts
from .config import GSUB_ROUNDS, STYLE_OF, THAI_PUA, TRACE_NAME  # noqa: F401
from .text import _font_cmap, find_fonts, font_info, save_font_info

C = RefontCfg()
TEXT_FLAGS = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES  # lines only: pictures come from get_image_info
LEADING_VOWELS = "เแโไใ"
JOINERS = "‌‍"
NO_SPACE_SCRIPTS = ("THAI", "LAO", "KHMER", "MYANMAR", "CJK", "HIRAGANA", "KATAKANA", "HANGUL", "IDEOGRAPHIC")
CJK_NAMES = ("hiragana", "katakana", "hangul", "bopomofo", "ideographic")
BELOW_FEATURES = {
    "blwf",
    "pstf",
    "vatu",
}  # GSUB features of the below / post-base consonant forms (ra-phala ...): virama + consonant
SYMBOL_FONTS = (
    "wingding",
    "webding",
    "symbol",
    "dingbat",
    "zapf",
)  # fonts whose letters are pictures: set as the page's glyph
NOTO_PREFIXES = (
    "notosans",
    "notoserif",
    "noto",
)  # longest first: NotoKufiArabic, NotoTraditionalNushu are script fonts too
TERMINAL = tuple(".;:!?…。")
LEADERS = set(".…_-–—=")
ENTRY_END = re.compile(r"(\.{3,}|…{2,}|_{4,}|(\. ){4,})\s*[\divxlcIVXLC]*\s*$")
_T = np.arange(1, C.bezier_steps + 1)[:, None] / C.bezier_steps
BEZIER = np.hstack(
    [(1 - _T) ** 3, 3 * (1 - _T) ** 2 * _T, 3 * (1 - _T) * _T**2, _T**3]
)  # cubic Bezier weights per step
MATH_LETTERS = {
    cp: unicodedata.normalize("NFKC", chr(cp))
    for cp in range(0x1D400, 0x1D800)
    if len(unicodedata.normalize("NFKC", chr(cp))) == 1 and unicodedata.normalize("NFKC", chr(cp)) != chr(cp)
}
ITEM = re.compile(r"\s*(\d{1,3}[.)]\s*|[•·▪■◦○●∙–—*-]\s+)\S")
VALUE = re.compile(r"[\s.…·_\-–—\d/,:()%+]+")  # a table value: numbers, leaders
MARKER = re.compile(r"[(\[]?(\d{1,3}|[ivxIVX]{1,4}|[a-zA-Zа-яА-Я])[.)\]]?|[•·▪■□◦○●∙\-–—*]")


def _rtl_char(ch):
    return unicodedata.bidirectional(ch) in ("R", "AL")


def is_rtl(text):
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(_rtl_char(c) for c in letters) > C.rtl_share * len(letters)


@lru_cache(maxsize=1 << 16)
def _script(ch):
    word = unicodedata.name(ch, "").split(" ")[0].lower()
    return "cjk" if word in CJK_NAMES else word


def _gray(rgb):
    return int(0.299 * (rgb >> 16 & 255) + 0.587 * (rgb >> 8 & 255) + 0.114 * (rgb & 255))


def _gray_f(c):
    return int(255 * (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])) if len(c) == 3 else int(255 * c[0])


def span_style(s):
    n = s["font"].lower()
    return (
        bool(s["flags"] & 16) or any(k in n for k in ("bold", "black", "heavy")),
        bool(s["flags"] & 2) or "italic" in n or "oblique" in n,
    )


def span_mono(s):
    return bool(s["flags"] & 8) or any(k in s["font"].lower() for k in ("mono", "courier", "consolas"))


def _tilted(l):
    """A tilted line is one picture-like glyph run: set as a pixmap of the page."""
    first = l["spans"][0]
    box = pymupdf.Rect(l["bbox"])
    span = dict(
        text="\ufffc",
        size=round(first["size"] * C.size_step) / C.size_step,
        gray=_gray(first["color"]),
        bold=False,
        italic=False,
        mono=False,
        nat=_nat(first),
        glyph=box,
        tall=True,
        bbox=box,
        font=first["font"],
    )
    return [dict(bbox=box, spans=[span])]


def _cmap_subtables(data):
    """(platform, encoding) of the cmap subtables of TrueType font data, read from its table directory (empty when the
    data holds no cmap)."""
    n = struct.unpack_from(">H", data, 4)[0] if len(data) >= 12 else 0
    for k in range(min(n, (len(data) - 12) // 16)):
        tag, _, off, length = struct.unpack_from(">4sIII", data, 12 + 16 * k)
        if tag == b"cmap" and off + 4 <= len(data):
            m = min(struct.unpack_from(">H", data, off + 2)[0], (min(length, len(data) - off) - 4) // 8)
            return {struct.unpack_from(">HH", data, off + 4 + 8 * j) for j in range(max(m, 0))}
    return set()


def _winansi_only(cmap):
    """Whether a ToUnicode CMap (its stream) maps every code it has to the code's own WinAnsi (cp1252) reading: the
    encoding spelled out, nothing about the glyphs."""
    text = cmap.decode("latin-1")
    pairs = []
    for block in re.findall(r"beginbfrange(.*?)endbfrange", text, re.S):
        for a, b, u in re.findall(r"<([0-9a-fA-F]+)>\s*<([0-9a-fA-F]+)>\s*<([0-9a-fA-F]+)>", block):
            pairs += [(k, int(u, 16) + k - int(a, 16)) for k in range(int(a, 16), int(b, 16) + 1)]
    for block in re.findall(r"beginbfchar(.*?)endbfchar", text, re.S):
        pairs += [(int(a, 16), int(u, 16)) for a, u in re.findall(r"<([0-9a-fA-F]+)>\s*<([0-9a-fA-F]+)>", block)]

    def winansi(k):
        try:
            return ord(bytes([k]).decode("cp1252")) if k < 256 else None
        except UnicodeDecodeError:
            return None

    return bool(pairs) and all(winansi(k) == u for k, u in pairs)


def symbolic_fonts(page):
    """Names of a page's embedded simple TrueType fonts whose text cannot be trusted: mapped only by the Symbol cmap (no
    Unicode one) and with no ToUnicode map or one that only spells out the WinAnsi reading of the codes, so their text is
    the font's own codes (legacy encodings), readable only in that font. A name counts when every font of the page that has it is such a font. The font program is read from its stream."""
    doc, bad, ok = page.parent, set(), set()

    def ref(key):
        n = int(key[1].split()[0]) if key[0] == "xref" else 0
        return n if 0 < n < doc.xref_length() else 0

    for xref, _, kind, base, *_ in page.get_fonts():
        name = base.split("+")[-1]
        desc = (
            ref(doc.xref_get_key(xref, "FontDescriptor")) if kind == "TrueType" and 0 < xref < doc.xref_length() else 0
        )
        ff = ref(doc.xref_get_key(desc, "FontFile2")) if desc else 0
        subs = _cmap_subtables(doc.xref_stream(ff) or b"") if ff else set()
        tu = ref(doc.xref_get_key(xref, "ToUnicode")) if 0 < xref < doc.xref_length() else 0
        legacy = (
            (3, 0) in subs
            and not subs & {(3, 1), (3, 10), (0, 3), (0, 4)}
            and (not tu or _winansi_only(doc.xref_stream(tu) or b""))
        )
        (bad if legacy else ok).add(name)
    return bad - ok


def _icon(c):
    """A character that is no text: a private-use code or a control code other than white space (the glyph of an icon font)."""
    return unicodedata.category(c) == "Co" or (unicodedata.category(c) == "Cc" and c not in "\t\n\r\x0b\x0c")


def _ink(s):
    """Rectangle of a rawdict span with the vertical extent its font gives it capped (some fonts claim lines three ems tall,
    which make neighbouring lines overlap): at most line_asc_em above its baseline and line_desc_em below."""
    x0, y0, x1, y1 = s["bbox"]
    base = s["origin"][1]
    return pymupdf.Rect(x0, max(y0, base - C.line_asc_em * s["size"]), x1, min(y1, base + C.line_desc_em * s["size"]))


def _union(rects):
    out = pymupdf.Rect()
    for r in rects:
        out |= r
    return out


def _nat(s):
    """Natural line height of a span's font in em, its ascent and descent capped as the line boxes are."""
    return min(s["ascender"], C.line_asc_em) - max(s["descender"], -C.line_desc_em) or C.line_em


def _span(s, text, symbolic):
    bold, italic = span_style(s)
    symbol = s["font"] in symbolic or any(k in s["font"].lower() for k in SYMBOL_FONTS)
    marks = [c for c in text if _icon(c) or not c.isspace()]
    icon = bool(marks) and all(_icon(c) for c in marks) and len(marks) <= C.icon_chars
    glyph = pymupdf.Rect(s["bbox"]) if symbol or icon or (marks and all(_icon(c) for c in marks)) else None
    return dict(
        text=text,
        size=round(s["size"] * C.size_step) / C.size_step,
        gray=_gray(s["color"]),
        bold=bold,
        italic=italic,
        mono=span_mono(s),
        nat=_nat(s),
        glyph=glyph,
        icon=icon,
        bbox=_ink(s),
        font=s["font"],
    )


def _chars_text(s):
    """Text of a rawdict span: its characters, with a space where two grapheme clusters (a letter with its marks, UAX #29) stand apart by more than
    pdfminer's word margin (some PDFs place words apart without a space character)."""
    out, prev, right = [], None, None
    for ch in s["chars"]:
        c = ch["c"]
        if (
            prev is not None
            and c != " "
            and prev["c"] != " "
            and not unicodedata.category(c[:1] or " ").startswith("M")
            and ch["bbox"][0] - right
            > C.word_margin * max(ch["bbox"][2] - ch["bbox"][0], ch["bbox"][3] - ch["bbox"][1])
        ):
            out.append(" ")
        out.append(c)
        right = (
            ch["bbox"][2]
            if right is None or not unicodedata.category(c[:1] or " ").startswith("M")
            else max(right, ch["bbox"][2])
        )  # the cluster's right edge
        prev = ch
    return "".join(out)


def _spans(l, symbolic, visual=False):
    spans = []
    for s in l["spans"]:
        text = _chars_text(s).replace("\x00", "").replace("\xa0", " ").translate(THAI_PUA).translate(MATH_LETTERS)
        if visual and not s.get("logical"):  # a span decoded by a table is in logical order already
            text = indic.to_logical(text)
        text = text.replace(indic.REPH, "")
        if (text.strip() or any(_icon(c) for c in text)) and round(s["size"] * C.size_step) > 0:
            spans.append(_span(s, text, symbolic))
        elif (
            text and spans and not spans[-1]["text"].endswith(" ")
        ):  # a space set in a font of its own: the words it parts stay apart
            spans[-1]["text"] += " "
    return spans


def _line_index(rows):
    """Arrays of a page's lines [(box, spans)] for _crossed: line boxes (n, 4), line sizes, span extents (m, 2) and lines."""
    box = np.array([tuple(b) for b, _ in rows], float).reshape(-1, 4)
    size = np.array([max(s["size"] for s in sp) for _, sp in rows], float)
    spans = np.array([(s["bbox"].x0, s["bbox"].x1, k) for k, (_, sp) in enumerate(rows) for s in sp], float).reshape(
        -1, 3
    )
    return box, size, spans[:, :2], spans[:, 2].astype(int)


def _crossed(a, b, y, size, index):
    """Whether the lines of the same size next to a line (within gap_reach lines) mostly have ink over the gap a..b of it,
    two of them at least: a wide space of justified text, not a gutter or the gap between table cells (a heading over the
    columns does not count). index: _line_index of the page."""
    box, sizes, ext, line = index
    dy = np.abs((box[:, 1] + box[:, 3]) / 2 - y)
    near = (
        (dy > 0)
        & (dy < C.gap_reach * size)
        & (box[:, 0] < a)
        & (box[:, 2] > b)
        & (np.abs(sizes - size) <= C.same_size_pt)
    )
    n = int(near.sum())
    if n < 2:
        return False
    ink = (np.minimum(ext[:, 1], b) - np.maximum(ext[:, 0], a) >= 0.5 * (b - a)) & near[line]
    hit = np.unique(line[ink]).size
    return hit >= 2 and hit >= C.gap_cross * n


def _runs(l, spans, rows):
    """Segments of a line: its spans, cut where they are far apart (a gutter between columns, a tab), unless the gap is a
    wide space of justified text or follows a list marker. A line cut into a wide and a narrow part (a label and its value,
    the cells of a table row) is tabbed: its parts are boxes, not flowing text."""
    runs = []
    y = (l["bbox"][1] + l["bbox"][3]) / 2
    for span in spans:
        if runs:
            last = runs[-1][-1]["bbox"]
            gap = max(span["bbox"].x0 - last.x1, last.x0 - span["bbox"].x1)
            far = gap > C.split_em * span["size"] and not _wide_space(
                "".join(s["text"] for s in runs[-1]), last.x1, span["bbox"].x0, y, span["size"], rows
            )
        if not runs or far:
            runs.append([])
        runs[-1].append(span)
    y0, y1 = l["bbox"][1], l["bbox"][3]
    out = [
        dict(bbox=pymupdf.Rect(min(sp["bbox"].x0 for sp in run), y0, max(sp["bbox"].x1 for sp in run), y1), spans=run)
        for run in runs
    ]
    tabbed = len(out) > 1 and min(r["bbox"].width for r in out) < C.min_width_em * max(s["size"] for s in spans)
    for r in out:
        r["tabbed"] = tabbed
    return out


def _embolden(line, over):
    """The words an overprinted run repeats become bold in the line under it (a span is split where the run covers part of it)."""
    for o in over["spans"]:
        word = o["text"].strip()
        for k, sp in enumerate(line["spans"]):
            width = sp["bbox"].width
            if width <= 0 or min(sp["bbox"].x1, o["bbox"].x1) - max(sp["bbox"].x0, o["bbox"].x0) < 0.5 * min(
                width, o["bbox"].width
            ):
                continue
            at = sp["text"].find(word)
            if at < 0:
                continue
            n = len(sp["text"])
            edge = lambda c: sp["bbox"].x0 + width * c / n
            parts = [
                (sp["text"][:at], 0, at, sp["bold"]),
                (word, at, at + len(word), True),
                (sp["text"][at + len(word) :], at + len(word), n, sp["bold"]),
            ]
            line["spans"][k : k + 1] = [
                dict(sp, text=t, bold=b, bbox=pymupdf.Rect(edge(a), sp["bbox"].y0, edge(z), sp["bbox"].y1))
                for t, a, z, b in parts
                if t
            ]
            break


def _overprint(segs):
    """A line drawn over another (fake bold: some words, or the whole line, printed again in the same place) is not a line of
    its own: the words it repeats are bold in the line under it. Lines are compared with those that start within a line's reach."""
    order = {id(l): i for i, l in enumerate(segs)}
    box = np.array([tuple(l["bbox"]) for l in segs], float).reshape(-1, 4)
    text = ["".join(_text(l).split()) for l in segs]
    area = [l["bbox"].get_area() for l in segs]
    bins = [round(l["bbox"].y0 / C.overprint_pt) for l in segs]
    kept, rows = [], defaultdict(list)  # the kept lines (indices), and by row bin
    for i in sorted(range(len(segs)), key=lambda i: -area[i]):
        under = None
        near = rows[bins[i] - 1] + rows[bins[i]] + rows[bins[i] + 1] if area[i] > 0 else []
        if near:
            k = np.array(
                near
            )  # the overlap with every line near, at once; the first that holds the words is the one under
            w = np.minimum(box[i, 2], box[k, 2]) - np.maximum(box[i, 0], box[k, 0])
            h = np.minimum(box[i, 3], box[k, 3]) - np.maximum(box[i, 1], box[k, 1])
            hit = k[(w > 0) & (h > 0) & (w * h >= C.duplicate_overlap * area[i])]
            under = next((j for j in hit.tolist() if text[i] in text[j]), None)
        if under is None:
            kept.append(i)
            rows[bins[i]].append(i)
        else:
            _embolden(segs[under], segs[i])
    return sorted((segs[i] for i in kept), key=lambda l: order[id(l)])


def _cmap_pairs(raw, off):
    """(code, glyph id) pairs of the cmap subtable at `off` (formats 0, 4, 6, 12); struct.error when it is cut short. Other formats: none."""
    fmt = struct.unpack_from(">H", raw, off)[0]
    if fmt == 0:
        return list(enumerate(struct.unpack_from(">256B", raw, off + 6)))
    if fmt == 4:
        n = struct.unpack_from(">H", raw, off + 6)[0] // 2
        end, start = struct.unpack_from(f">{n}H", raw, off + 14), struct.unpack_from(f">{n}H", raw, off + 16 + 2 * n)
        delta, rng = (
            struct.unpack_from(f">{n}h", raw, off + 16 + 4 * n),
            struct.unpack_from(f">{n}H", raw, off + 16 + 6 * n),
        )
        out = []
        for i in range(n):
            for c in range(start[i], min(end[i], 0xFFFE) + 1):  # 0xFFFF ends the table, its offset points nowhere
                if rng[i] == 0:
                    out.append((c, (c + delta[i]) & 0xFFFF))
                else:
                    g = struct.unpack_from(">H", raw, off + 16 + 6 * n + 2 * i + rng[i] + 2 * (c - start[i]))[0]
                    out.append((c, (g + delta[i]) & 0xFFFF if g else 0))
        return out
    if fmt == 6:
        first, count = struct.unpack_from(">HH", raw, off + 6)
        return [(first + i, g) for i, g in enumerate(struct.unpack_from(f">{count}H", raw, off + 10))]
    if fmt == 12:
        n = struct.unpack_from(">L", raw, off + 12)[0]
        groups = struct.unpack_from(f">{3 * n}L", raw, off + 16)
        return [
            (c, groups[3 * i + 2] + c - groups[3 * i])
            for i in range(n)
            for c in range(groups[3 * i], min(groups[3 * i + 1], 0x10FFFF) + 1)
        ]
    return []


def _unicode_cmap(raw):
    """(code, glyph id) pairs of the Unicode subtables of a cmap table, in the order of the table. A subtable that is cut short is left out, the others
    stay (fontTools refuses the whole table)."""
    out = []
    for i in range(struct.unpack_from(">H", raw, 2)[0] if len(raw) >= 4 else 0):
        if len(raw) < 12 + 8 * i:
            break
        platform, encoding, off = struct.unpack_from(">HHL", raw, 4 + 8 * i)
        if platform == 0 or (
            platform == 3 and encoding in (0, 1, 10)
        ):  # as fontTools counts a table as Unicode (the Symbol cmap too)
            try:
                out += [(c, g) for c, g in _cmap_pairs(raw, off) if g]  # glyph 0 (.notdef) is no mapping
            except struct.error:
                pass
    return out


def _pua(u):
    return 0xE000 <= u <= 0xF8FF or u >= 0xF0000


CID_TO_UNICODE = "/usr/share/poppler/cidToUnicode"  # poppler-data: Unicode of each CID of the Adobe character collections (a hex code a line)


class _CidText(dict):
    """Glyph texts from a CID collection's table: a generic reading of the CID (a variant form: ⋯ for …), for the glyphs the text map gives U+FFFD only."""


@lru_cache(maxsize=8)
def _cid_unicode(ordering):
    try:
        return [int(l, 16) for l in open(os.path.join(CID_TO_UNICODE, f"Adobe-{ordering}"))]
    except (OSError, ValueError):
        return []


@lru_cache(maxsize=64)
def _font_glyph_text(data):
    """Text of each glyph of a font program (glyph id -> text): its Unicode cmap, and through its GSUB single and ligature
    substitutions the glyphs made from mapped ones (conjuncts, half forms); the glyphs its cmap maps directly; its glyph count.
    A damaged font gives what can be read: the cmap subtables that are whole, no GSUB part when that table is damaged, glyph names made from the
    glyph numbers when the 'post' table cannot be read. A bare CFF CID font of an Adobe collection (Adobe-Korea1 ...): its glyph names give the CIDs, the
    collection's CID table the text."""
    if data[:2] == b"\x01\x00":
        from fontTools.cffLib import CFFFontSet

        try:
            cff = CFFFontSet()
            cff.decompile(io.BytesIO(data), None)
            top = cff[cff.fontNames[0]]
            ros, names = getattr(top, "ROS", None), list(top.charset)
        except Exception:
            return {}, frozenset(), 0
        uni = _cid_unicode(ros[1]) if ros and ros[0] == "Adobe" else []
        cid = [
            int(n[3:]) for n in names if n.startswith("cid") and n[3:].isdigit()
        ]  # a CID font's glyphs are numbered by CID (as the text trace gives them)
        text = _CidText({c: chr(uni[c]) for c in cid if c < len(uni) and uni[c] >= 0x20})
        return text, frozenset(text), max(cid, default=-1) + 1
    f = TTFont(io.BytesIO(data), lazy=True)
    try:
        order = f.getGlyphOrder()
    except (TTLibError, KeyError, ValueError, AssertionError, struct.error):
        order = []
    n = f["maxp"].numGlyphs if "maxp" in f.reader.tables else len(order)
    if len(order) != n:
        f.setGlyphOrder([f"glyph{i:05d}" for i in range(n)])
        order = f.getGlyphOrder()
    gid = {name: i for i, name in enumerate(order)}
    text = {}
    for u, g in _unicode_cmap(f.reader["cmap"]) if "cmap" in f.reader.tables else ():
        if 0x80 <= u < 0xA0:  # a C1 control: the byte code of an 8-bit cmap, its cp1252 character
            u = ord(bytes([u]).decode("cp1252", "ignore") or "\0")
        if g < n and not _pua(u) and u >= 0x20:  # a private-use code names no letter: the glyph's text comes from GSUB
            text.setdefault(g, chr(u))
    for g, name in enumerate(
        order
    ):  # a glyph the cmap leaves out: its name (Adobe Glyph List: uni0E33, periodcentered)
        if g not in text and not re.fullmatch(r"(glyph|cid|gid|index|g)\d+|\.notdef", name):
            t = agl.toUnicode(name)
            if t and not any(_pua(ord(c)) for c in t):
                text[g] = t
    direct = frozenset(text)
    subs, below, reph = [], set(), set()
    try:
        gsub = f["GSUB"].table if "GSUB" in f else None
        for fr in gsub.FeatureList.FeatureRecord if gsub and gsub.FeatureList else ():
            if fr.FeatureTag in BELOW_FEATURES:
                below |= set(fr.Feature.LookupListIndex)
            if fr.FeatureTag == "rphf":
                reph |= set(fr.Feature.LookupListIndex)
        for li, lk in enumerate(gsub.LookupList.Lookup if gsub and gsub.LookupList else ()):
            for st in lk.SubTable:
                st = st.ExtSubTable if lk.LookupType == 7 else st
                if getattr(st, "LookupType", lk.LookupType) == 4:
                    subs += [
                        ([first, *lg.Component], lg.LigGlyph, "below" if li in below else "reph" if li in reph else "")
                        for first, ligs in st.ligatures.items()
                        for lg in ligs
                    ]
                elif getattr(st, "LookupType", lk.LookupType) == 1:
                    subs += [([a], b, "reph" if li in reph else "") for a, b in st.mapping.items()]
    except (TTLibError, KeyError, ValueError, IndexError, AssertionError, struct.error):
        subs = []
    for _ in range(GSUB_ROUNDS):
        grown = False
        for src, dst, low in subs:
            d = gid.get(dst)
            if d is not None and d not in text and all(gid.get(g) in text for g in src):
                t = "".join(text[gid[g]] for g in src)
                if (
                    low == "below"
                    and len(t) == 2
                    and unicodedata.category(t[0]) == "Lo"
                    and unicodedata.category(t[1]) == "Mn"
                ):
                    t = t[1] + t[0]  # a below / post-base form (ra + virama in the font's order) stands for virama + ra
                elif low == "reph":
                    t = indic.REPH + t  # drawn after the syllable it stands on: indic.to_logical puts it before
                text[d] = t
                grown = True
        if not grown:
            break
    return text, direct, n


class Glyphs(dict):
    """_glyph_texts' result; logical: the keys whose text is in logical order already (words ordered by the reference font's shaping);
    unverified: words whose shaping matched no order of their glyph texts."""

    def __init__(self):
        super().__init__()
        self.logical, self.unverified, self.trace = set(), 0, {}


def _codepage(own, mapped):
    """An 8-bit code page font (Cyrillic, Greek, Arabic ... at the Latin-1 positions of its cmap): its program names a Latin letter, its text map
    the letter of the other script the code stands for; the text map is right, not wrong."""
    return (
        len(own) == 1
        and len(mapped) == 1
        and own.isalpha()
        and ord(own) < 0x250
        and mapped.isalpha()
        and ord(mapped) >= 0x370
    )


def _glyph_texts(page):
    """(font name, origin) -> text of the glyph there, for the runs whose text map (ToUnicode) gives some glyph another
    character than the cmap of the font program maps it to directly: the map is wrong, the glyphs say what the text is.
    A run's program is the one of its name that has all its glyphs and agrees with most of its characters (more than half, else the run is left
    as it is). Glyphs at one origin (a mark on its letter) give their texts in order; the text is in drawing order (an Indic
    prefix sign before its consonant), as a text map gives it."""
    doc, programs, refs = page.parent, defaultdict(list), defaultdict(list)
    known = doc.__dict__.setdefault(
        "_glyph_programs", {}
    )  # xref -> glyph texts of its font program (None: none), read once per document
    known_ref = doc.__dict__.setdefault(
        "_glyph_refs", {}
    )  # xref -> glyph texts read through a reference font (reffonts), {} for none
    traces, out = legacy.typed_traces(page), Glyphs()
    for tr in traces:  # the codes typed where the text map gives no text (legacy.typed_traces)
        for i in tr.get("typed", ()):
            u, g, o, _ = tr["chars"][i]
            out[(tr["font"], round(o[0], 1), round(o[1], 1))] = chr(u)
    out.trace = {
        (tr["font"], round(o[0], 1), round(o[1], 1)): chr(u)
        for tr in traces
        for u, g, o, _ in tr["chars"]
        if g > 0 and not legacy._no_text(u)
    }
    fonts = [f for f in page.get_fonts() if f[1] in ("ttf", "otf", "cid") and 0 < f[0] < doc.xref_length()]
    if not fonts:  # no run has a program to read
        return out
    used = {tr["font"] for tr in traces}  # only the fonts the text of the page is set in (the resources list more)
    nm = lambda base: base.split("+")[-1][:TRACE_NAME]  # the name as the trace gives it (MuPDF cuts it)
    for xref, ext, kind, base, *_ in fonts:
        if nm(base) in used:
            if xref not in known:
                known[xref] = None
                try:
                    data = doc.extract_font(xref)[3]
                except (
                    pymupdf.mupdf.FzErrorFormat
                ):  # the font file of a broken PDF is not a stream: the text of the page is read without it
                    data = None
                if data:
                    known[xref] = _font_glyph_text(data)
                known_ref[xref] = reffonts.glyph_texts(base.split("+")[-1], data)
            if known_ref.get(xref, ({}, None))[0]:
                refs[nm(base)].append(known_ref[xref])
            if known[xref] is not None:
                programs[nm(base)].append(known[xref])
    if not programs and not refs:
        return out
    for tr in traces:
        chars = [(chr(u), g, o) for u, g, o, _ in tr["chars"] if g > 0]
        ink = [(c, g) for c, g, _ in chars if not c.isspace()]
        ref, path = max(refs.get(tr["font"], ()), key=lambda r: sum(g in r[0] for _, g in ink), default=(None, None))
        if (
            ref and ink and sum(g in ref for _, g in ink) >= 0.9 * len(ink)
        ):  # the full font names the glyphs: exact, before any map
            words, drawn, mine = [[]], set(), set()
            em = tr["size"] or 1
            for u, g, o, b in tr["chars"]:
                key = (tr["font"], round(o[0], 1), round(o[1], 1))
                t, rg = ref.get(g, (chr(u), None))  # the reference's text (the map may call a letter a space)
                if (
                    g <= 0 or (key, g) in drawn
                ):  # a 2nd character the text map gives one glyph, a glyph drawn again (fake bold)
                    words[-1].append((key, "", None, 0))
                elif t.isspace():
                    words += [[(key, " ", None, 0)], []]
                else:
                    words[-1].append((key, t, rg, o[0] / em))
                drawn.add((key, g))
            for w in words:
                pieces = [(t, rg, x) for _, t, rg, x in w if rg is not None]
                text, ok = (
                    reffonts.word(path, pieces)
                    if pieces and len(pieces) == sum(bool(t) for _, t, _, _ in w)
                    else ("".join(t for _, t, _, _ in w), True)
                )
                out.unverified += not ok
                for k, (key, _, _, _) in enumerate(w):
                    if key in mine:  # two glyphs of this run at one place (a word after a zero-width space)
                        out[key] += text if k == 0 else ""
                    elif key not in out:  # (a key of an earlier run: the same text drawn again, fake bold)
                        out[key] = text if k == 0 else ""
                        out.logical.add(key)
                        mine.add(key)
            continue

        clusters = []  # the text map's characters of each glyph: a glyph the map gives several characters
        for u, g, o, _ in tr["chars"]:  # (a conjunct: pa + virama + ra on one glyph) keeps them all where the program
            if g > 0:  # names it no text
                clusters.append([chr(u), g, o])
            elif (
                clusters and unicodedata.category(chr(u))[0] in "LM"
            ):  # a letter or mark of the cluster (a hyphenation point the map adds is no part)
                clusters[-1][0] += chr(u)

        def agree(p):
            text, direct, n = p
            return (
                -1 if any(g >= n for _, g, _ in chars) else sum(text.get(g) == c or c == "\ufffd" for c, g, _ in chars)
            )  # U+FFFD says nothing

        best = max(programs.get(tr["font"], ()), key=agree, default=None)
        if best is None or 2 * agree(best) <= len(
            chars
        ):  # the program of its name the map mostly agrees with (subsets share a name)
            continue
        text, direct, _ = best
        if any(
            g in text
            and text[g] != c
            and (g in direct or _pua(ord(c)) or not c.isalpha())
            and not _codepage(text[g], c)
            for c, g, _ in chars
        ):  # a wrong map: a glyph the cmap
            # names otherwise, a private-use code, or a sign (@ $ _) where the program's GSUB makes a conjunct
            ts = [
                (
                    (tr["font"], round(o[0], 1), round(o[1], 1)),
                    text.get(g, c) if c == "\ufffd" or not isinstance(text, _CidText) else c,
                )
                for c, g, o in clusters
            ]
            run = {}
            for key, t in ts:
                run[key] = run.get(key, "") + t
            for key, t in run.items():
                out.setdefault(key, t)
    return out


def _one_cluster(c, n):
    """U+FFFD c and its neighbour n are characters of one ActualText span spread over its glyphs: n has text and one of the two draws no width
    (MuPDF gives the span's characters beyond its glyphs none)."""
    w = lambda x: round(x["bbox"][2] - x["bbox"][0], 1)
    return n["c"] != "\ufffd" and not n["c"].isspace() and (w(c) == 0 or w(n) == 0)


def extract_segments(page, textpage=None, fix_fonts=True):
    """Text lines of a page, each cut into one segment per run. fix_fonts: the text of the runs whose font has a wrong text map is read from the font
    program (_glyph_texts); it matters for old and rare fonts and costs a good part of a page."""
    lines = [
        l for b in page.get_text("rawdict", flags=TEXT_FLAGS, textpage=textpage)["blocks"] for l in b.get("lines", [])
    ]
    if legacy.LANG.get():  # text drawn as glyph outlines (filled paths) is read like a legacy font without a program
        lines += legacy.path_lines(page)
    symbolic = symbolic_fonts(page)
    fixed = _glyph_texts(page) if fix_fonts else {}
    trace = getattr(fixed, "trace", {})  # the character the glyph trace gives where the text extraction gives U+FFFD
    drawn, tabled = (
        set(fixed) - getattr(fixed, "logical", set()),
        set(getattr(fixed, "logical", set())),
    )  # glyph texts in drawing order: put in logical order with the text left as it is
    if legacy.LANG.get():  # a document with a known Indic language: its legacy-font runs are read by glyph shape
        lf = legacy.fixed(page, legacy.LANG.get())
        if lf.guess:  # a shape guess only where the text stays broken: no program text, no known code (Thai PUA)
            shown = {
                (sp["font"], round(ch["origin"][0], 1), round(ch["origin"][1], 1)): ch["c"]
                for l in lines
                for sp in l["spans"]
                for ch in sp["chars"]
            }
            seen = lambda k: trace.get(k, "\ufffd") if shown.get(k, "\ufffd") == "\ufffd" else shown[k]
            drop = {
                k
                for k in lf.guess
                if k in fixed
                or k not in shown
                or not (seen(k) == "\ufffd" or legacy.misread(seen(k), None))
                or ord(seen(k)) in THAI_PUA
                or ord(seen(k)) < 0x20
            }  # a C0 control is whitespace or a format code (a space), not a lost letter
            for k in drop:
                del lf[k]
        tabled |= set(lf) - lf.fitted - drawn  # decoded by a table or code page: logical order already
        fixed = {
            **fixed,
            **{k: v for k, v in lf.items() if k not in fixed or k not in lf.fitted},
        }  # tables and code pages first, the font program's glyph text before the shape fit
    untouched = []  # the text the fixes do not replace: the fixed text is in logical order already
    for l in lines:
        for sp in l["spans"]:
            for ch in sp["chars"]:
                if (
                    ord(ch["c"]) < 0x20
                ):  # a control code no glyph name or trace reads (a tab glyph before dot leaders): a space
                    ch["c"] = " "
        cs = [c for sp in l["spans"] for c in sp["chars"]]
        unknown = {
            id(c)
            for i, c in enumerate(cs)
            if c["c"] == "\ufffd" and any(_one_cluster(c, cs[j]) for j in (i - 1, i + 1) if 0 <= j < len(cs))
        }
        for sp in l["spans"]:
            seen, after = set(), False
            for ch in sp["chars"]:
                key = (sp["font"], round(ch["origin"][0], 1), round(ch["origin"][1], 1))
                if id(ch) in unknown:  # U+FFFD ("text unknown", as a producer's map or ActualText writes it) on a
                    ch["c"] = ""  # piece of a glyph cluster whose text a neighbour carries
                    continue
                if ch["c"] == "\ufffd":  # a glyph of its own: the character the glyph trace gives it
                    ch["c"] = trace.get(key, "\ufffd")
                if (
                    key not in fixed and after and ch["bbox"][2] - ch["bbox"][0] < 1e-3
                ):  # a zero-width 2nd character the text map gives a fixed glyph: in its text
                    ch["c"] = ""
                    continue
                after = key in fixed
                if key in fixed:
                    ch["c"] = "" if key in seen else fixed[key]
                    if key in drawn and key not in seen:
                        untouched.append(ch["c"])
                    elif key in tabled:
                        sp["logical"] = True
                    seen.add(key)
                else:
                    untouched.append(ch["c"])
    visual = indic.visual_order("".join(untouched))
    segs, rows = [], []
    for l in lines:
        if abs(l["dir"][1]) > C.tilt:
            segs += _tilted(l) if "".join(c["c"] for s in l["spans"] for c in s["chars"]).strip() else []
        else:
            spans = _spans(l, symbolic, visual)
            if spans:
                ink = _union(_ink(sp) for sp in l["spans"])
                rows.append(((l["bbox"][0], ink.y0, l["bbox"][2], ink.y1), spans))
    index = _line_index(rows)
    for box, spans in rows:
        segs += _runs(dict(bbox=box), spans, index)
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
    spans = sorted(
        (l["bbox"].x0, l["bbox"].x1)
        for l in segs
        if _rsize(l) <= C.body_band * body
        and l["bbox"].width >= C.column_line * _rsize(l)
        and (
            lefts[round(l["bbox"].x0 / C.edge_bin)] >= C.edge_lines
            or rights[round(l["bbox"].x1 / C.edge_bin)] >= C.edge_lines
        )
    )
    if not spans:
        return []
    lo, hi = int(min(x0 for x0, _ in spans)), int(max(x1 for _, x1 in spans)) + 1
    cover = np.zeros(hi - lo + 1)
    for x0, x1 in spans:
        cover[int(x0) - lo : int(x1) - lo + 1] += 1
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
    hits = [
        i for i, (a, z) in enumerate(cols) if min(bbox.x1, z) - max(bbox.x0, a) >= C.column_hit * max(bbox.width, 1.0)
    ]
    return hits[0] if len(hits) == 1 else None


def _wide_space(prev, x1, x0, y, size, near):
    """Whether a gap x1..x0 after the text `prev` on a line is a space that joins (it follows a list marker and is no wider
    than a tab, or the lines around it have ink over it: justified text)."""
    gap = x0 - x1
    return gap > 0 and (
        (gap < C.tab_max_em * size and MARKER.fullmatch(prev.strip()) is not None) or _crossed(x1, x0, y, size, near)
    )


def join_rows(segs, cols, vlines=()):
    rows, buckets = [], defaultdict(list)
    near = _line_index([(tuple(l["bbox"]), l["spans"]) for l in segs])
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
                if not (
                    -C.join_back_em * sz < lx0 - x1 < C.join_ahead_em * sz
                    or _wide_space(_text(r["segs"][-1]), x1, lx0, (ly0 + ly1) / 2, sz, near)
                ):
                    continue
                small = min(py1 - py0, ly1 - ly0)
                inline = abs((py0 + py1) / 2 - (ly0 + ly1) / 2) < C.inline_share * sz or (
                    min(sz, psz) < C.superscript_ratio * max(sz, psz) and min(py1, ly1) - max(py0, ly0) >= 0.5 * small
                )
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
        r["tabbed"] = any(g.get("tabbed") for g in r["segs"])
    for a in rows:
        for b in buckets[int(((a["bbox"].y0 + a["bbox"].y1) / 2) // C.row_bin)]:
            sz = max(_rsize(a), _rsize(b))
            narrow = min((a, b), key=lambda r: r["bbox"].width)
            if (
                b is not a
                and min(a["bbox"].y1, b["bbox"].y1) - max(a["bbox"].y0, b["bbox"].y0)
                > 0.5 * min(a["bbox"].height, b["bbox"].height)
                and (a["bbox"].x1 <= b["bbox"].x0 or b["bbox"].x1 <= a["bbox"].x0)
                and narrow["bbox"].width < C.min_width_em * sz
                and (a["col"] is None or b["col"] is None or a["col"] == b["col"] or VALUE.fullmatch(_text(narrow)))
            ):
                a["tabbed"] = b["tabbed"] = True
    return sorted(rows, key=lambda r: (r["bbox"].y0, r["bbox"].x0))


def _form_marks(rows, shapes):
    """Rows next to drawn shapes (n, 4): with a field box on their line (a form row: tabbed) or with a small mark just
    before them (a radio button or a check box: a new item)."""
    if not len(shapes):
        return
    for r in rows:
        bb, sz = r["bbox"], _rsize(r)
        over = np.minimum(shapes[:, 3], bb.y1) - np.maximum(shapes[:, 1], bb.y0) > 0.5 * np.minimum(
            shapes[:, 3] - shapes[:, 1], bb.height
        )
        h, w = shapes[:, 3] - shapes[:, 1], shapes[:, 2] - shapes[:, 0]
        field = (
            over
            & (h <= C.field_em * sz)
            & (w > C.marker_em * sz)
            & ((shapes[:, 0] >= bb.x1 - 1) | (shapes[:, 2] <= bb.x0 + 1))
        )
        mark = (
            over
            & (h <= C.marker_em * sz)
            & (w <= C.marker_em * sz)
            & (bb.x0 - shapes[:, 2] >= -1)
            & (bb.x0 - shapes[:, 2] < C.marker_em * sz)
        )
        r["tabbed"] = r.get("tabbed", False) or bool(field.any())
        r["marked"] = bool(mark.any())


def paragraphs(segs, rules=((), ()), shapes=()):
    cols = column_intervals(segs)
    rows = join_rows(segs, cols, rules[0])
    _form_marks(rows, np.asarray(shapes, float).reshape(-1, 4))
    colw = defaultdict(float)
    for r in rows:
        k = (round(_rsize(r)), round(r["bbox"].x0 / C.para_bin))
        colw[k] = max(colw[k], r["bbox"].width)
    blocks = []
    for r in rows:
        sz = _rsize(r)
        k = round(r["bbox"].x0 / C.para_bin)
        W = max(colw[(round(sz), j)] for j in (k - 1, k, k + 1))
        passed = []
        for b in reversed(blocks[-C.lookback :]):
            last = b["lines"][-1]["bbox"]
            pitch = r["bbox"].y0 - last.y0
            ref = b["pitch"] if len(b["lines"]) > 1 else C.single_pitch_em * sz
            edge = abs(r["bbox"].x0 - last.x0) < 3 or abs(r["bbox"].x1 - last.x1) < 3
            ruled = any(
                y0 <= y <= y1 and x0 < min(last.x1, r["bbox"].x1) and x1 > max(last.x0, r["bbox"].x0)
                for y, x0, x1 in rules[1]
                for y0, y1 in [(last.y1 - 0.5 * sz, r["bbox"].y0 + 0.5 * sz)]
            )
            open_end = not _text(b["lines"][-1]).rstrip().endswith(TERMINAL)
            first_ltr = (
                len(b["lines"]) == 1
                and C.indent_min_em * sz < last.x0 - r["bbox"].x0 < C.indent_max_em * sz
                and (abs(last.x1 - r["bbox"].x1) < C.edge_match_em * sz + 3 or (r["bbox"].x1 < last.x1 and open_end))
            )
            first_rtl = (
                len(b["lines"]) == 1
                and C.indent_min_em * sz < r["bbox"].x1 - last.x1 < C.indent_max_em * sz
                and (abs(last.x0 - r["bbox"].x0) < C.edge_match_em * sz + 3 or (r["bbox"].x0 > last.x0 and open_end))
            )
            indented = (
                len(b["lines"]) >= 2
                and C.indent_min_em * sz
                < ((last.x1 - r["bbox"].x1) if is_rtl(_text(r)) else (r["bbox"].x0 - last.x0))
                < C.indent_max_em * sz
            )
            step_ltr = (
                len(b["lines"]) <= C.indent_lines
                and all(l["bbox"].x0 > r["bbox"].x0 + 0.5 * sz for l in b["lines"])
                and abs(last.x1 - r["bbox"].x1) < C.edge_match_em * sz + 3
            )
            step_rtl = (
                len(b["lines"]) <= C.indent_lines
                and all(l["bbox"].x1 < r["bbox"].x1 - 0.5 * sz for l in b["lines"])
                and abs(last.x0 - r["bbox"].x0) < C.edge_match_em * sz + 3
            )
            entry = ENTRY_END.search(_text(b["lines"][-1])) is not None  # a contents entry or form field: complete
            goes_on = open_end and last.width > C.full_line * b["W"] and _text(r).lstrip()[:1].islower()
            if (
                abs(_rsize(b["lines"][-1]) - sz) < C.same_size_pt
                and (edge or first_ltr or first_rtl or step_ltr or step_rtl)
                and b["col"] == r["col"]
                and not ruled
                and not indented
                and not r.get("marked")
                and not r.get("tabbed")
                and not b["lines"][-1].get("tabbed")
                and r["spans"][0]["glyph"] is None
                and not (ITEM.match(_text(r)) and (not open_end or last.width < C.full_line * b["W"]))
                and C.indent_min_em * sz < pitch < (C.open_pitch_em * sz if goes_on else ref * C.pitch_tol)
                and b["W"] > C.min_width_em * sz
                and not any(o["bbox"].x1 > r["bbox"].x0 and o["bbox"].x0 < r["bbox"].x1 for o in passed)
                and not entry
                and (
                    last.width > C.full_line * b["W"]
                    or (last.width > C.ragged_fill * b["W"] and not _text(b["lines"][-1]).rstrip().endswith(TERMINAL))
                )
            ):
                if len(b["lines"]) == 1:
                    b["pitch"] = pitch
                b["lines"].append(r)
                b["bbox"] |= r["bbox"]
                break
            passed.append(b)
        else:
            blocks.append(dict(bbox=pymupdf.Rect(r["bbox"]), lines=[r], W=W, pitch=0, col=r["col"]))
    for b in blocks:
        b["rtl"] = is_rtl("".join(_text(l) for l in b["lines"]))
        b["size"] = max(_rsize(l) for l in b["lines"])
        b["align"], edge = _align(b)
        if b["align"] == "justify":
            b["bbox"] = (
                pymupdf.Rect(edge, b["bbox"].y0, b["bbox"].x1, b["bbox"].y1)
                if b["rtl"]
                else pymupdf.Rect(b["bbox"].x0, b["bbox"].y0, edge, b["bbox"].y1)
            )
        b["colspan"] = cols[b["col"]] if b["col"] is not None else None
        if b["align"] != "justify" and len(b["lines"]) >= C.ragged_lines:
            ends = sorted(l["bbox"].x0 if b["rtl"] else l["bbox"].x1 for l in b["lines"])
            cut = (
                ends[int((1 - C.ragged_percentile) * (len(ends) - 1))]
                if b["rtl"]
                else ends[int(C.ragged_percentile * (len(ends) - 1))]
            )
            b["bbox"] = (
                pymupdf.Rect(max(b["bbox"].x0, cut), b["bbox"].y0, b["bbox"].x1, b["bbox"].y1)
                if b["rtl"]
                else pymupdf.Rect(b["bbox"].x0, b["bbox"].y0, min(b["bbox"].x1, cut), b["bbox"].y1)
            )
        if b["colspan"] is not None:
            clamped = pymupdf.Rect(
                max(b["bbox"].x0, b["colspan"][0]), b["bbox"].y0, min(b["bbox"].x1, b["colspan"][1]), b["bbox"].y1
            )
            if clamped.width >= C.trim_max * b["bbox"].width:
                b["bbox"] = clamped
        nat = statistics.median(s["nat"] for l in b["lines"] for s in l["spans"])
        pitch = (
            statistics.median(
                b["lines"][i + 1]["bbox"].y0 - b["lines"][i]["bbox"].y0 for i in range(len(b["lines"]) - 1)
            )
            if len(b["lines"]) > 1
            else b["size"] * nat
        )
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


RENAMED = {
    "notoloopedthai": "notosansthailooped",
    "notoloopedlao": "notosanslaolooped",
}  # Noto families renamed by their newer releases


def _family_key(path):
    return os.path.splitext(os.path.basename(path))[0].split("[")[0].split("-")[0].split(",")[0].lower()


@lru_cache(maxsize=1024)
def _family_script(key):
    prefix = next((p for p in NOTO_PREFIXES if key.startswith(p)), None)
    if prefix is None:
        return ""
    rest = key[len(prefix) :].replace("display", "").replace("ui", "")
    return "" if rest == "mono" else rest


@lru_cache(maxsize=4)
def families(fonts):
    fam = defaultdict(dict)
    for p in fonts:
        base = os.path.splitext(os.path.basename(p))[0].split("[")[0]
        suffix = base.split("-", 1)[1].lower() if "-" in base else ""
        styles = (
            {2 * (suffix == "italic"): C.regular_weight, 2 * (suffix == "italic") + 1: C.bold_weight}
            if font_info(p)[2]
            else {STYLE_OF[suffix]: None}
            if suffix in STYLE_OF
            else {}
        )
        for (
            st,
            wght,
        ) in styles.items():  # one file of a family and style in several places: the newest version (as fontconfig)
            have = fam[_family_key(p)].get(st)
            if have is None or _revision(p) > _revision(have[0]):
                fam[_family_key(p)][st] = (p, wght)
    save_font_info()
    fam = {k: v for k, v in fam.items() if 0 in v}
    return {
        k: v for k, v in fam.items() if RENAMED.get(k) not in fam
    }  # a family released again under a new name: the new release only


@cache
def _revision(path):
    from fontTools.ttLib import TTFont

    return TTFont(path, lazy=True, fontNumber=0)["head"].fontRevision


def style_file(files, bold, italic):
    for key in (int(bold) + 2 * int(italic), int(bold), 2 * int(italic), 0):
        if key in files:
            return files[key]


def pick_family(rng, fams, chars, bold=False, italic=False, need=(), choice=None):
    letters = {c for c in chars if c.isalpha()} or set(chars)
    scripts = {k for k, v in Counter(_script(c) for c in letters).items() if v >= C.script_share * len(letters)}
    cps = {ord(c) for c in chars}
    main = {ord(c) for c in letters if _script(c) in scripts}
    score = {}
    for k, v in fams.items():
        rest = _family_script(k)
        if rest == "" or any(s in rest for s in scripts):
            cmap = _font_cmap(style_file(v, bold, italic)[0])
            styled = all(int(b) + 2 * int(i) in v for b, i in need)
            score[k] = (
                len(main & cmap),
                styled,
                len(cps & cmap),
            )  # the letters of the page's own scripts, then the styles it needs, then all its characters
    best = max(score.values())
    pool = sorted(k for k, s in score.items() if s == best)
    return pool[
        rng.integers(0, len(pool)) if choice is None else choice % len(pool)
    ]  # choice: a chosen family of the fitting ones (a balanced design)


def font(path, px, wght):
    return _font(path, round(px * C.font_step) / C.font_step, wght)


@lru_cache(maxsize=4096)
def _font(path, px, wght):
    f = ImageFont.truetype(path, max(px, 1.0), layout_engine=ImageFont.Layout.RAQM)
    f.key = (path, px, wght)
    if wght is not None:
        f.set_variation_by_axes(
            [
                min(max(wght, a["minimum"]), a["maximum"]) if a["name"] in (b"Weight", "Weight") else a["default"]
                for a in f.get_variation_axes()
            ]
        )
    return f


@lru_cache(maxsize=1 << 18)
def text_width(path, px, wght, text, direction):
    """Width of a text: shaped as a whole for the scripts that join or reorder, else the sum of the letters' advances
    (the cached ones, which render sets the letters by)."""
    f = font(path, px, wght)
    return f.getlength(text, direction=direction) if needs_shaping(text) else sum(_advance(f.key, ch) for ch in text)


class Typesetter:
    """Font file for a piece of text: the role's family, or another family when that file lacks a glyph."""

    def __init__(self, fonts, rng, chars, need=(), pick=None):
        fams = families(fonts)
        self.rng = rng
        self.prose = {k: v for k, v in fams.items() if not font_info(v[0][0])[1]}
        self.mono = {k: v for k, v in fams.items() if font_info(v[0][0])[1]}
        hk, bk = pick or (None, None)
        self.role = {
            "head": pick_family(rng, self.prose, chars, need=need, choice=hk),
            "body": pick_family(rng, self.prose, chars, need=need, choice=bk),
            "mono": pick_family(rng, self.mono, chars),
        }
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
        for c in regex.findall(r"\X", text):  # by grapheme cluster (UAX #29): a mark stays in its base letter's font
            has = lambda m: all(ord(x) in m or x in JOINERS for x in c)
            font_c = (path, wght) if has(cmap) else self._covering(c, bold, italic)
            if font_c is None:
                self.unresolved[c] += 1
            elif runs and runs[-1][0] == font_c:
                runs[-1][1] += c
            else:
                runs.append([font_c, c])
        self.dropped = sum(self.unresolved.values())
        return [(f[0], f[1], t) for f, t in runs]

    def _covering(self, c, bold, italic):
        """A font file with every character of the cluster c."""
        key = (c, bold, italic)
        if key not in self.fallback:
            has = lambda f: all(ord(x) in _font_cmap(f) or x in JOINERS for x in c)
            guess = style_file(self.prose[pick_family(self.rng, self.prose, set(c), bold, italic)], bold, italic)
            if not has(guess[0]):
                guess = next(
                    (
                        style_file(v, bold, italic)
                        for k, v in sorted(self.prose.items())
                        if has(style_file(v, bold, italic)[0])
                    ),
                    None,
                )
            self.fallback[key] = guess
        return self.fallback[key]


def _clusters(text):
    out = []
    for ch in text:
        if out and (
            unicodedata.combining(ch) or unicodedata.category(ch) in ("Mn", "Mc", "Me") or ch in JOINERS + "ำຳ"
        ):
            out[-1] += ch
        else:
            out.append(ch)
    return out


@lru_cache(maxsize=1 << 16)
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


def _leader(text):
    """A run of one dot, dash or underscore (the filler of a contents line or a form field): it can be as long as the room."""
    return len(text) >= C.leader_min and len(set(text)) == 1 and text[0] in LEADERS


def _glyph_token(s, px, space, tab):
    return dict(
        text="",
        glyph=(s["number"], s["glyph"]),
        size=s["size"],
        px=px,
        gray=s["gray"],
        gap=space,
        tab=tab,
        brk=False,
        rtl=False,
        path=None,
        wght=None,
        tall=s.get("tall", False),
        icon=s.get("icon", False),
        leader=False,
    )


def _word_tokens(s, typesetter, role, px, space, tab):
    """Tokens of a span's words: the pieces of a word each in the font that has its letters. -> (tokens, space, tab)."""
    toks = []
    for i, w in enumerate(s["text"].replace("\n", " ").split(" ")):
        space = space or i > 0
        for piece in _pieces(w) if w else ():
            for path, wght, text in typesetter.resolve(role, piece, s["bold"], s["italic"]):
                toks.append(
                    dict(
                        text=text,
                        path=path,
                        wght=wght,
                        px=px,
                        gray=s["gray"],
                        gap=space,
                        tab=tab,
                        size=s["size"],
                        brk=_nobreak(text),
                        rtl=any(_rtl_char(c) for c in text if c.isalpha()),
                        leader=_leader(text),
                    )
                )
                space, tab = False, 0.0
    return toks, space, tab


def _whole_clusters(spans):
    """The spans of a line with no grapheme cluster (UAX #29) cut between two of them: the marks a span starts with go to the end of the text span
    before it (a PDF may change font or size inside a cluster); copies, the spans read are kept as they are."""
    out = []
    for s in spans:
        head = regex.match(r"\p{M}+", s["text"] or "")
        last = next((o for o in reversed(out) if o["glyph"] is None), None)
        if head and s["glyph"] is None and last is not None and last is out[-1] and last["text"]:
            out[-1] = dict(last, text=last["text"] + head.group())
            s = dict(s, text=s["text"][head.end() :])
            if not s["text"]:
                continue
        out.append(s)
    return out


def tokens(b, typesetter, role_of, px_of):
    toks, space, prev, tab = [], False, None, 0.0
    for l in b["lines"]:
        for s in _whole_clusters(l["spans"]):
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
    space = C.glyph_gap * t["px"] if t["path"] is None else _advance(font(t["path"], t["px"], t["wght"]).key, " ")
    return max(space, t["tab"] * t["px"] / t["size"])


def measure(t):
    if t["path"] is None:
        return None, t["glyph"][1].width * t["px"] / t["size"]
    f = font(t["path"], t["px"], t["wght"])
    return f, text_width(t["path"], t["px"], t["wght"], t["text"], "rtl" if t["rtl"] else "ltr")


def _split(t, room):
    """(head, tail) of a token at the last line break opportunity (UAX #14, ICU: dictionary breaks for Thai, ...) whose head fits room; a
    grapheme cluster only when no break fits (a word longer than the line)."""
    from .linebreak import breaks

    fits = [b for b in breaks(t["text"]) if b < len(t["text"]) and measure(dict(t, text=t["text"][:b]))[1] <= room]
    if fits:
        return t["text"][: fits[-1]], t["text"][fits[-1] :]
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
            if t["path"] is not None and t["brk"]:
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


def single_align(x0, x1, c0, c1, rtl):
    """Where a one-line block sits in its column: centred, flush to an edge, or left. A line that covers most of the column
    from an indent is left-set text that reaches the edge, not a right-flush one."""
    colw = max(c1 - c0, 1.0)
    left, right = x0 - c0, c1 - x1
    if left < C.edge_tol * colw + 2 and right < C.edge_tol * colw + 2:
        return "right" if rtl else "left"
    if left > C.align_inset * colw and abs(left - right) < C.edge_tol * colw + 3:
        return "center"
    if right < C.edge_tol * colw + 2 and left > C.align_inset * colw and x1 - x0 < C.right_share * colw:
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


def prepare(b, others, page_w, g, typesetter, role_of, px_of, cols, walls=()):
    """A text block set in its box at its place. others: (n, 4) boxes of the page's other text blocks, in points."""
    bb = b["bbox"]
    x0, x1, y0, y1 = bb.x0 * g, bb.x1 * g, bb.y0 * g, bb.y1 * g
    toks = tokens(b, typesetter, role_of, px_of)
    if not toks:
        return None
    multi = len(b["lines"]) > 1
    bpx = px_of(b["size"])
    lettered = next((t for t in toks if t["path"] is not None), None)
    asc, desc = (
        font(lettered["path"], bpx, lettered["wght"]).getmetrics()
        if lettered
        else (C.ascent_share * bpx, (1 - C.ascent_share) * bpx)
    )
    cx = (x0 + x1) / 2
    c0, c1 = (
        (b["colspan"][0] * g, b["colspan"][1] * g)
        if b.get("colspan")
        else (min(a for a, _ in cols), max(z for _, z in cols))
    )
    W, bx0, ind1, ind2 = x1 - x0, x0, 0.0, 0.0
    if multi:
        align = b["align"]
        ind1, ind2 = indents(b, g, bpx)
    else:
        align = (
            ("right" if b["rtl"] else "left")
            if b["lines"][0].get("marked")
            else b.get("src_align") or single_align(x0, x1, c0, c1, b["rtl"])
        )
        beside = np.minimum(others[:, 3], bb.y1) - np.maximum(others[:, 1], bb.y0) > C.row_overlap * np.minimum(
            others[:, 3] - others[:, 1], bb.height
        )
        row = (others[beside][:, [0, 2]] * g).tolist()
        row += [(x, x) for x, wy0, wy1 in walls if min(wy1, y1) - max(wy0, y0) > C.row_overlap * (y1 - y0)]
        slack = C.edge_slack * (c1 - c0)
        e0, e1 = max(c0 - slack, 2.0), min(c1 + slack, page_w * g - 2.0)
        right = min((a for a, z in row if a >= x1 - 1), default=e1)
        left = max((z for a, z in row if z <= x0 + 1), default=e0)
        cap = {
            "left": min(e1, right) - x0,
            "right": x1 - max(e0, left),
            "center": 2 * min(cx - max(e0, left), min(e1, right) - cx),
        }[align]
        Wa = max(min(C.room * W, cap), W)
        bx0, W = {"left": x0, "right": x1 - Wa, "center": cx - Wa / 2}[align], Wa
        if b.get("flush"):
            align = "justify"
    lines = wrap(toks, W, ind1, ind2)
    return dict(
        b=b,
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        bx0=bx0,
        W=W,
        align=align,
        ind=(ind1, ind2),
        lines=lines,
        pitch=line_pitch(b, bpx, asc, desc),
        asc=asc,
        desc=desc,
        multi=multi,
        pinned=b["pinned"],
        rtl=b["rtl"],
        bpx=bpx,
        tol=C.row_tol_em * b["size"] * g,
        wrapped=not multi and len(lines) > 1,
    )


def _paths(d):
    """Point lists of one drawing, one per subpath (rectangles and quads closed)."""
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
                cur += (BEZIER @ [(q.x, q.y) for q in (p1, *it[2:5])]).tolist()
        else:
            if cur:
                paths.append(cur)
                cur = []
            if op == "re":
                r = it[1]
                paths.append([(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1), (r.x0, r.y0)])
            else:
                q = it[1]
                paths.append([(q.ul.x, q.ul.y), (q.ur.x, q.ur.y), (q.lr.x, q.lr.y), (q.ll.x, q.ll.y), (q.ul.x, q.ul.y)])
    if cur:
        paths.append(cur)
    return paths


def _fill(img, paths, gray, alpha, even_odd=False):
    """Antialiased polygons filled as one shape: a subpath that runs the other way round than the first (nonzero rule), or
    lies inside another (even-odd rule), is a hole; each shape blends by its own coverage, so adjoining rows of one table
    leave their thin seam."""
    xs, ys = [p[0] for pts in paths for p in pts], [p[1] for pts in paths for p in pts]
    x0, y0 = max(int(min(xs)) - 1, 0), max(int(min(ys)) - 1, 0)
    x1, y1 = min(int(max(xs)) + 2, img.width), min(int(max(ys)) + 2, img.height)
    if x1 <= x0 or y1 <= y0:
        return
    sign = lambda pts: np.sign(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(pts, pts[1:] + pts[:1])))
    first = sign(paths[0])
    big = np.zeros(((y1 - y0) * C.aa, (x1 - x0) * C.aa), np.uint8)
    for pts in paths:
        one = Image.new("L", big.shape[::-1], 0)
        ImageDraw.Draw(one).polygon([((x - x0) * C.aa, (y - y0) * C.aa) for x, y in pts], fill=255)
        one = np.asarray(one)
        if even_odd:
            big ^= one
        elif len(paths) > 1 and sign(pts) == -first != 0:
            big[one > 0] = 0
        else:
            big |= one
    mask = Image.fromarray((big * (alpha)).astype(np.uint8)).resize((x1 - x0, y1 - y0), Image.BOX)
    img.paste(gray, (x0, y0), mask)


def page_frame(r, ph, pw):
    """A bar along the page edge that spans most of the page: a frame, kept whole where it is."""
    tall = (
        r.height >= C.frame_span * ph
        and r.width <= C.frame_thin * pw
        and (r.x0 < C.edge_zone * pw or r.x1 > (1 - C.edge_zone) * pw)
    )
    wide = (
        r.width >= C.frame_span * pw
        and r.height <= C.frame_thin * ph
        and (r.y0 < C.edge_zone * ph or r.y1 > (1 - C.edge_zone) * ph)
    )
    return tall or wide


def _key(r):
    return tuple(round(v / 2) for v in (r.x0, r.y0, r.x1, r.y1))


def graphics(src, number, first, ph, pw, home=None, below=None):
    """Vector drawings of a page as (drawing, fixed), in source points. On the following pages, what repeats from the home
    page (running headers, footers, frames) is left out."""
    out = []
    repeated = set() if first else {_key(d["rect"]) for d in src.drawings(home)}
    for d in src.drawings(number):
        r = d["rect"]
        fixed = (
            r.y1 < C.header_pt
            or r.y0 > ph - C.footer_pt
            or page_frame(r, ph, pw)
            or (below is not None and r.y0 >= below)
        )
        if not first and (fixed or _key(r) in repeated or r.width * r.height > C.repeat_cover * pw * ph):
            continue
        out.append((d, fixed and first))
    return out


def join_tiles(boxes):
    """Picture boxes (x0, y0, x1, y1) with the ones that touch each other (the tiles of one picture) joined into one
    rectangle when together they cover tile_cover of it; in the order of their first box."""
    r = np.array([tuple(b) for b in boxes], float).reshape(-1, 4)
    if len(r) < 2:
        return [pymupdf.Rect(*b.tolist()) for b in r]
    t = C.touch_pt
    order = np.argsort(r[:, 0], kind="stable")
    rs, x0 = r[order], np.sort(r[:, 0])
    widest = float((rs[:, 2] - rs[:, 0]).max())
    rows, cols = [], []
    for a in range(
        0, len(rs), 512
    ):  # the touching pairs in blocks of rows (sorted by x0, so only the columns near them can touch)
        q = rs[a : a + 512]
        lo, hi = (
            np.searchsorted(x0, q[:, 0].min() - t - widest, "left"),
            np.searchsorted(x0, q[:, 2].max() + t, "right"),
        )
        c = rs[lo:hi]
        near = (
            (q[:, None, 0] <= c[None, :, 2] + t)
            & (c[None, :, 0] <= q[:, None, 2] + t)
            & (q[:, None, 1] <= c[None, :, 3] + t)
            & (c[None, :, 1] <= q[:, None, 3] + t)
        )
        i, j = np.nonzero(near)
        rows.append(i + a), cols.append(j + lo)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    _, found = connected_components(
        csr_matrix((np.ones(len(rows), bool), (rows, cols)), shape=(len(r), len(r))), directed=False
    )
    label = np.empty(len(r), found.dtype)
    label[order] = found
    size = np.bincount(label)
    joined = {}  # label -> the joined rectangle, or None when the tiles do not cover enough of it
    for k in np.flatnonzero(size > 1):
        m = r[label == k]
        box = np.r_[m[:, :2].min(0), m[:, 2:].max(0)]
        joined[k] = (
            pymupdf.Rect(*box.tolist())
            if np.prod(m[:, 2:] - m[:, :2], axis=1).sum() >= C.tile_cover * np.prod(box[2:] - box[:2])
            else None
        )
    out, done = [], set()
    for i, k in enumerate(label):
        if joined.get(k) is not None:
            if k not in done:
                out.append(joined[k])
                done.add(k)
        else:
            out.append(pymupdf.Rect(*r[i].tolist()))
    return out


def _inked(src, number, r):
    """The part of a picture's rectangle that shows anything (an image is often placed larger than it is seen: clipped,
    or with white or transparent margins); None when nothing shows. Cut from one render of the clean page: every picture of a page
    would run its display list again, decoding its images each time."""
    full = src.clean_pixmap(number)
    bounds = pymupdf.IRect(full.irect)
    box = pymupdf.IRect(r.irect) & bounds
    a = np.frombuffer(full.samples, np.uint8).reshape(full.h, full.stride)[
        box.y0 - bounds.y0 : box.y1 - bounds.y0, box.x0 - bounds.x0 : box.x1 - bounds.x0
    ]
    rows, cols = np.flatnonzero((a < C.ink_level).any(1)), np.flatnonzero((a < C.ink_level).any(0))
    if not len(rows):
        return None
    sx, sy = r.width / box.width, r.height / box.height
    return pymupdf.Rect(
        r.x0 + cols[0] * sx, r.y0 + rows[0] * sy, r.x0 + (cols[-1] + 1) * sx, r.y0 + (rows[-1] + 1) * sy
    )


def _covered(boxes, r):
    """Area of rectangle r covered by boxes (n, 4) (overlaps between the boxes counted twice)."""
    w = np.minimum(boxes[:, 2], r.x1) - np.maximum(boxes[:, 0], r.x0)
    h = np.minimum(boxes[:, 3], r.y1) - np.maximum(boxes[:, 1], r.y0)
    return float((np.clip(w, 0, None) * np.clip(h, 0, None)).sum())


def pictures(src, number, ph, pw, below=None):
    """Pictures of a page as blocks of the flow, in source points, cut at the page border, the tiles of a picture joined;
    pictures that only repeat the page's text are left out."""
    out = []
    page = pymupdf.Rect(0, 0, pw, ph)
    rects = [
        r
        for r in join_tiles(r for r in (pymupdf.Rect(im["bbox"]) & page for im in src.images(number)) if not r.is_empty)
        if max(r.width, r.height) >= src.min_picture
    ]  # the letters of a page set as bitmaps are no pictures
    if not rects:
        return out
    words = np.array([w[:4] for w in src.words(number)], float).reshape(-1, 4)
    for r in rects:
        r = _inked(src, number, r) if max(r.width, r.height) > C.dot_pt else r
        if r is None:
            continue
        fixed = r.y1 < C.header_pt or r.y0 > ph - C.footer_pt or (below is not None and r.y0 >= below)
        if _covered(words, r) >= C.text_image_share * r.get_area():
            continue
        out.append(dict(image=True, rect=r, clip=r, number=number, bbox=pymupdf.Rect(r), pinned=fixed))
    return out


def cell_boxes(elems, page_area):
    """Rectangles of the filled or closed drawings no larger than cell_share of the page: table cells, boxes, panels."""
    return [
        d["rect"]
        for d, fixed in elems
        if (d.get("fill") is not None or d.get("closePath") or any(it[0] == "re" for it in d["items"]))
        and d["rect"].get_area() <= C.cell_share * page_area
    ]


def _hrules(elems, g):
    """Horizontal rules of the graphics as (x0, x1, y) in output px: the lines that make the rows of a table."""
    return [(x0 * g, x1 * g, y * g) for d, fixed in elems for x0, y, x1 in _horizontal(d)]


def _straight(d):
    """The straight edges ((xa, ya), (xb, yb)) of a drawing: its lines and the sides of its rectangles and quads (a curve
    is never a rule)."""
    for it in d["items"]:
        if it[0] == "l":
            yield (it[1].x, it[1].y), (it[2].x, it[2].y)
        elif it[0] in ("re", "qu"):
            q = it[1].quad if it[0] == "re" else it[1]
            pts = [(q.ul.x, q.ul.y), (q.ur.x, q.ur.y), (q.lr.x, q.lr.y), (q.ll.x, q.ll.y)]
            yield from zip(pts, pts[1:] + pts[:1])


def _vrules(elems, g):
    """Vertical rules of the graphics as (x, y0, y1) in output px: cell borders and frame sides a text must not cross."""
    out = []
    for d, fixed in elems:
        for (xa, ya), (xb, yb) in _straight(d):
            if abs(xa - xb) < 0.5 and abs(ya - yb) >= C.rule_min_pt:
                out.append((xa * g, min(ya, yb) * g, max(ya, yb) * g))
        r = d["rect"]
        if d.get("fill") is not None and r.width <= 2 and r.height >= C.rule_min_pt:
            out.append((r.x0 * g, r.y0 * g, r.y1 * g))
    return out


def _band(rules, ys, p):
    """The rules above and below a block that run along all of its width: the row of a table it sits in, or None."""

    def covers(r):
        return r[0] <= p["x0"] + 2 and r[1] >= p["x1"] - 2

    up = next(
        (rules[k][2] for k in range(bisect.bisect_right(ys, p["y0"] + p["tol"]) - 1, -1, -1) if covers(rules[k])), None
    )
    down = next(
        (rules[k][2] for k in range(bisect.bisect_left(ys, p["y1"] - p["tol"]), len(rules)) if covers(rules[k])), None
    )
    return None if up is None or down is None or down <= up else (up, down)


def _text_free(doc, number, min_picture=0.0):
    """The display list of the page without the text and line art on its pictures (dots aside: nothing lies under them): what its pictures
    are taken from. One redaction of the picture rectangles removes the text that touches them and the line art; only the pictures are rendered from it, so
    the part of a line that lies outside them does not matter."""
    one = pymupdf.open()
    one.insert_pdf(doc, from_page=number, to_page=number, links=0, annots=0, widgets=0)
    page = one[0]
    for r in join_tiles(im["bbox"] for im in page.get_image_info()):
        if max(r.width, r.height) > max(C.dot_pt, min_picture):
            page.add_redact_annot(r)
    page.apply_redactions(images=0, graphics=2)
    return page.get_displaylist()


def draw_graphics(img, drawings, g):
    """Drawings (drawing, dy px) in output px, each moved up by its dy."""
    for d, dy in drawings:
        fill, stroke = d.get("fill"), d.get("color")
        alpha = 1.0 if d.get("fill_opacity") is None else d["fill_opacity"]
        width = max(1, round((d.get("width") or 1.0) * g))
        paths = [[(x * g, y * g - dy) for x, y in path] for path in _paths(d)]
        flat = [p for path in paths for p in path]
        if flat:
            lo, hi = (
                (min(p[0] for p in flat), min(p[1] for p in flat)),
                (max(p[0] for p in flat), max(p[1] for p in flat)),
            )
            shift = [0.0, 0.0]
            for k, size in enumerate((img.width, img.height)):
                if hi[k] < 1 and hi[k] > -C.edge_snap_px:
                    shift[k] = 1 - hi[k]
                elif lo[k] > size - 1 and lo[k] < size + C.edge_snap_px:
                    shift[k] = size - 1 - lo[k]
            paths = [[(x + shift[0], y + shift[1]) for x, y in path] for path in paths]
        if fill is not None and alpha > C.min_alpha:
            shapes = [pts for pts in paths if len(pts) > 2]
            if shapes:
                _fill(img, shapes, _gray_f(fill), alpha, bool(d.get("even_odd")))
        for pts in paths:
            if stroke is not None and len(pts) > 1:
                ImageDraw.Draw(img).line(
                    pts + ([pts[0]] if d.get("closePath") else []), fill=_gray_f(stroke), width=width
                )


def text_page(src, number):
    """False for scanned or picture pages: little text, or pictures covering most of the page."""
    chars = sum(len(s["text"].strip()) for l in src.lines(number)[0] for s in l["spans"])
    rect = src[number].rect
    image = sum((pymupdf.Rect(im["bbox"]) & rect).get_area() for im in src.images(number))
    return chars >= C.min_text_chars and image <= C.max_image_share * rect.get_area()


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


def line_rules(drawings):
    """Vertical lines (x, y0, y1) and horizontal lines (y, x0, x1) of a page's drawings: cell borders and rules."""
    v, h = [], []
    for x0, y0, x1, y1 in (b for d in drawings for b in _rule_boxes(d)):
        if x1 - x0 <= C.rule_thin and y1 - y0 >= C.rule_long:
            v.append(((x0 + x1) / 2, y0, y1))
        elif y1 - y0 <= C.rule_thin and x1 - x0 >= C.rule_long:
            h.append(((y0 + y1) / 2, x0, x1))
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
        out += [
            dict(
                l,
                bbox=pymupdf.Rect(
                    min(s["bbox"].x0 for s in grp), l["bbox"].y0, max(s["bbox"].x1 for s in grp), l["bbox"].y1
                ),
                spans=grp,
            )
            for grp in groups
        ]
    return out


class Source:
    """The pages of an open PDF, each read once: text lines and rules, drawings, pictures, words, and the page without the
    text that lies on its pictures."""

    def __init__(self, doc, min_picture=0.0):
        self.doc = doc
        self.min_picture = (
            min_picture  # an image (joined tiles taken as one) with no side of that many points is no picture
        )
        self._read = {}

    def __len__(self):
        return len(self.doc)

    def __getitem__(self, number):
        return self._get("page", number, lambda: self.doc[number])

    def _get(self, kind, number, make):
        if (kind, number) not in self._read:
            self._read[kind, number] = make()
        return self._read[kind, number]

    def textpage(self, number):
        return self._get("textpage", number, lambda: self[number].get_textpage(flags=TEXT_FLAGS))

    def drawings(self, number):
        return self._get("drawings", number, lambda: self[number].get_drawings())

    def cdrawings(self, number):
        """The drawings as get_cdrawings gives them (tuples, no Rect objects: a page of 100 000 line segments costs seconds as get_drawings); a key a
        drawing does not have stands for None."""
        return self._get("cdrawings", number, lambda: self[number].get_cdrawings())

    def images(self, number):
        return self._get("images", number, lambda: self[number].get_image_info())

    def words(self, number):
        return self._get("words", number, lambda: self[number].get_text("words", textpage=self.textpage(number)))

    def clean(self, number):
        return self._get("clean", number, lambda: self._clean(number))

    def clean_pixmap(self, number):
        """The clean page rendered once (grey, 1 px a point): what the pictures are cut from."""
        return self._get(
            "clean_pm", number, lambda: self.clean(number).get_pixmap(colorspace=pymupdf.csGRAY, alpha=False)
        )

    def _clean(self, number):
        """The display list the pictures are taken from: that of the page itself when no text and no line art touches a picture (the redaction of
        _text_free would remove nothing), else _text_free."""
        pics = [
            r
            for r in join_tiles(im["bbox"] for im in self.images(number))
            if max(r.width, r.height) > max(C.dot_pt, self.min_picture)
        ]
        if pics:
            p = np.array([tuple(r) for r in pics])
            spans = np.array(
                [
                    sp["bbox"]
                    for b in self[number].get_text("dict", flags=TEXT_FLAGS, textpage=self.textpage(number))["blocks"]
                    for l in b.get("lines", [])
                    for sp in l["spans"]
                ],
                float,
            ).reshape(-1, 4)
            art = np.array([d["rect"] for d in self.cdrawings(number)], float).reshape(-1, 4)
            touch = lambda a: any(
                (
                    (b[:, None, 0] <= p[None, :, 2])
                    & (p[None, :, 0] <= b[:, None, 2])
                    & (b[:, None, 1] <= p[None, :, 3])
                    & (p[None, :, 1] <= b[:, None, 3])
                ).any()
                for b in (a[i : i + 2048] for i in range(0, len(a), 2048))
            )  # in blocks: 250 000 drawings x 5 000 pictures are 1 GB
            if not touch(spans) and not touch(art):
                return self[number].get_displaylist()
        return _text_free(self.doc, number, self.min_picture)

    def lines(self, number):
        """Lines and rules of a page (fresh copies: callers change them)."""

        def parse():
            v, h = line_rules(self.drawings(number))
            return _cut(extract_segments(self[number], self.textpage(number)), v), (v, h)

        segs, (v, h) = self._get("lines", number, parse)
        return (
            [
                dict(l, bbox=pymupdf.Rect(l["bbox"]), spans=[dict(s, bbox=pymupdf.Rect(s["bbox"])) for s in l["spans"]])
                for l in segs
            ],
            (list(v), list(h)),
        )


def _overlaps(a, b, share):
    """Whether rectangle a covers at least `share` of the area of b."""
    w, h = min(a.x1, b.x1) - max(a.x0, b.x0), min(a.y1, b.y1) - max(a.y0, b.y0)
    return w > 0 and h > 0 and w * h >= share * b.width * b.height


def _absorb(blocks):
    """A one-line block that lies inside a paragraph on one of its lines is a part of that line (the fields of a spaced
    line): its spans join the line, and the block goes."""
    gone = set()
    for o in blocks:
        if len(o["lines"]) != 1 or o.get("image"):
            continue
        for b in blocks:
            if b is o or id(b) in gone or len(b["lines"]) < 2 or not _overlaps(b["bbox"], o["bbox"], C.contain_share):
                continue
            line = next(
                (
                    l
                    for l in b["lines"]
                    if min(l["bbox"].y1, o["bbox"].y1) - max(l["bbox"].y0, o["bbox"].y0) > 0.5 * o["bbox"].height
                ),
                None,
            )
            if line:
                line["spans"] = sorted(line["spans"] + o["lines"][0]["spans"], key=lambda s: s["bbox"].x0)
                line["bbox"] = line["bbox"] | o["lines"][0]["bbox"]
                gone.add(id(o))
                break
    return [b for b in blocks if id(b) not in gone]


def page_blocks(src, number, first, running=()):
    """Paragraphs of a page. Header and footer zones are those of the whole page, whatever its crop."""
    segs, rules = src.lines(number)
    shapes = [
        tuple(d["rect"])
        for d in src.drawings(number)
        if d["rect"].get_area() < C.background_share * src[number].rect.get_area()
    ]
    blocks = _absorb(paragraphs(segs, rules, shapes))
    page = src[number]
    offset, full = page.cropbox_position.y, page.mediabox.height
    for b in blocks:
        for l in b["lines"]:
            for sp in l["spans"]:
                sp["number"] = number
        top, bottom = b["bbox"].y0 + offset, b["bbox"].y1 + offset
        b["pinned"] = top > full - C.footer_pt or bottom < C.header_pt
        b["edge"] = top < C.run_zone * full or top > (1 - C.run_zone) * full

    def repeats(b):
        return b["edge"] and any(abs(round(b["bbox"].y0 / C.run_step) - r) <= 1 for r in running)

    return [b for b in blocks if first or not (b["pinned"] or repeats(b))]


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


def body_size(blocks):
    """The body font size of a page's blocks: the size (to 0.5 pt) of most of their characters (0: no text)."""
    spans = [s for b in blocks for l in b["lines"] for s in l["spans"]]
    sizes = Counter(
        {
            k: sum(len(s["text"]) for s in spans if round(s["size"] * 2) / 2 == k)
            for k in {round(s["size"] * 2) / 2 for s in spans}
        }
    )
    return sizes.most_common(1)[0][0] if sizes else 0.0


@lru_cache(maxsize=1 << 16)
def needs_shaping(text):
    """Scripts whose letters join, reorder or stack (Hebrew to Indic, Thai, Khmer, Arabic forms) are set by the shaping engine."""
    return any(
        0x0590 <= ord(c) <= 0x1CFF or 0xA800 <= ord(c) <= 0xABFF or 0xFB1D <= ord(c) <= 0xFEFF or c in JOINERS
        for c in text
    )


@lru_cache(maxsize=1 << 16)
def _advance(key, ch):
    return _font(*key).getlength(ch)


@lru_cache(maxsize=1 << 15)
def _coverage(key, text, direction):
    """Coverage of a letter (direction None, with its advance) or of a word shaped as a unit (advance 0)."""
    mask, (ox, oy) = _font(*key).getmask2(
        text, "L", anchor="ls", **({} if direction is None else dict(direction=direction))
    )
    w, h = mask.size
    cov = np.array(mask, dtype=np.uint8).reshape(h, w) if w and h else np.zeros((0, 0), np.uint8)
    return cov, ox, oy, 0.0 if direction else _advance(key, text)


def blit_text(img, items, s=1.0):
    """Text laid over the page from cached coverage (light text on a dark fill stays visible): letter by letter for scripts
    without shaping, whole words (shaped as a unit) for the others; each cut to its clip box (None: the page). s: the
    page drawn s times larger, letters at s times their size where the layout's advances put them."""
    covs, at = [], []
    H, W = img.size[1], img.size[0]
    for x, y, f, text, gray, direction, clip in items:
        cx0, cy0, cx1, cy1 = (0, 0, W, H) if clip is None else (round(c * s) for c in clip)
        key = f.key if s == 1.0 else (f.key[0], f.key[1] * s, f.key[2])
        for ch in [text] if needs_shaping(text) else text:
            d = direction if needs_shaping(text) else None
            cov, ox, oy, _ = _coverage(key, ch, d)
            if cov.size:
                covs.append(cov.ravel())
                at.append(
                    (
                        int(round(x * s)) + ox,
                        int(round(y * s)) + oy,
                        cov.shape[0],
                        cov.shape[1],
                        gray,
                        cx0,
                        cy0,
                        cx1,
                        cy1,
                    )
                )
            x += 0.0 if d else _advance(f.key, ch)
    if covs:
        arr = np.asarray(img).copy()
        _composite(arr, np.concatenate(covs), np.array(at, np.int64))
        img.paste(Image.fromarray(arr))


@njit(cache=True)
def _composite(arr, cov, at):
    """Glyph coverages (flattened one after another) laid over arr: at = (x0, y0, h, w, gray, clip x0, y0, x1, y1) of each."""
    H, W = arr.shape
    o = 0
    for g in range(at.shape[0]):
        x0, y0, h, w, gray = at[g, 0], at[g, 1], at[g, 2], at[g, 3], at[g, 4]
        cx0, cy0, cx1, cy1 = max(at[g, 5], 0), max(at[g, 6], 0), min(at[g, 7], W), min(at[g, 8], H)
        for r in range(max(0, cy0 - y0), min(h, cy1 - y0)):
            for c in range(max(0, cx0 - x0), min(w, cx1 - x0)):
                a = np.int64(cov[o + r * w + c])
                if a:
                    arr[y0 + r, x0 + c] = (np.int64(arr[y0 + r, x0 + c]) * (255 - a) + gray * a + 127) // 255
        o += h * w


def render(src, lay, s=1.0):
    """The layout drawn s times its size (the same layout, every length scaled)."""
    img = Image.new("L", tuple(round(v * s) for v in lay["size"]), 255)
    g = lay["g"] * s

    def paste_pictures(pinned):
        for x, top, b in lay["pics"]:
            if b["pinned"] == pinned:
                pm = src.clean(b["number"]).get_pixmap(
                    matrix=pymupdf.Matrix(g, g), clip=b["clip"], colorspace=pymupdf.csGRAY, alpha=False
                )
                pic = Image.frombytes("L", (pm.w, pm.h), pm.samples)
                if b["clip"] != b["rect"]:
                    pic = pic.resize((round(b["rect"].width * g), round(b["rect"].height * g)), Image.LANCZOS)
                img.paste(pic, (round(x * s), round(top * s)))

    draw_graphics(img, [(d, dy * s) for d, fixed, dy in lay["elems"] if not fixed], g)
    draw_graphics(img, [(d, dy * s) for d, fixed, dy in lay["elems"] if fixed], g)
    paste_pictures(False)
    paste_pictures(True)
    blit_text(img, lay["items"], s)
    for x, y, t in lay["glyphs"]:
        x, y = x * s, y * s
        number, rect = t["glyph"]
        k = t["px"] * s / t["size"]
        pm = src[number].get_pixmap(matrix=pymupdf.Matrix(k, k), clip=rect, colorspace=pymupdf.csGRAY)
        run = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.stride)[:, : pm.w]
        if not run.size:  # a glyph box of no area (an empty symbol glyph): nothing is drawn
            continue
        edge = np.concatenate([run[0], run[-1], run[:, 0], run[:, -1]])
        ink = np.abs(run.astype(int) - int(np.median(edge))) > C.glyph_ink
        if t.get("icon"):
            _, label = cv2.connectedComponents(
                cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2)
            )
            mid = label[pm.h // 3 : pm.h - pm.h // 3, pm.w // 3 : pm.w - pm.w // 3]
            ink &= np.isin(
                label, np.unique(mid[mid > 0])
            )  # the icon, not the letters of the lines its box reaches into
        mask = Image.fromarray((ink * 255).astype(np.uint8))
        img.paste(Image.fromarray(run), (round(x), round(y - C.ascent_share * pm.h)), mask)
    return img


def content_box(page, src=None, number=None, fix_fonts=True):
    """Rectangle of what the page shows: text lines, pictures and graphics (a background fill over the page does not count).
    With the Source and number of the page, what it has read of the page is used again."""
    if src is None:
        segs, images, drawings = (
            extract_segments(page, fix_fonts=fix_fonts),
            page.get_image_info(),
            page.get_cdrawings(),
        )
    else:
        segs, images, drawings = (
            extract_segments(page, src.textpage(number), fix_fonts),
            src.images(number),
            src.cdrawings(number),
        )
    boxes = np.array([tuple(l["bbox"]) for l in segs] + [tuple(i["bbox"]) for i in images], float).reshape(-1, 4)
    if drawings:  # a drawing with half its stroke width around it, unless it is a fill over most of the page
        r = np.array([d["rect"] for d in drawings], float).reshape(-1, 4)
        half = np.array([(d.get("width") or 1.0) / 2 if d.get("color") is not None else 0.0 for d in drawings])
        small = np.array([d.get("fill") is None for d in drawings]) | (
            (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1]) < C.background_share * page.rect.get_area()
        )
        boxes = np.vstack([boxes, (r + np.stack([-half, -half, half, half], 1))[small]])
    boxes = boxes[
        (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    ]  # an empty rectangle adds nothing to a union
    box = pymupdf.Rect(*np.r_[boxes[:, :2].min(0), boxes[:, 2:].max(0)].tolist()) if len(boxes) else pymupdf.Rect()
    return box & page.rect if box.is_valid and not box.is_empty else page.rect


def over_pictures(src, number, segs):
    """Share of the text lines that lie inside a picture: a scan with its recognized text hidden under it, not set text."""
    images = [pymupdf.Rect(im["bbox"]) for im in src.images(number)]
    return sum(any(l["bbox"] in im for im in images) for l in segs) / max(len(segs), 1)


def usable(src, number, latin=True):
    """A page of text set in the PDF: a body (not only header and footer), text of one script (Latin only if the language
    is written in it), no more than max_columns columns, no scan with a hidden text layer."""
    segs = src.lines(number)[0]
    body = [l for l in segs if l["bbox"].y1 >= C.header_pt and l["bbox"].y0 <= src[number].rect.height - C.footer_pt]
    return (
        bool(body)
        and text_page(src, number)
        and len(column_intervals(segs)) <= C.max_columns
        and over_pictures(src, number, segs) <= C.over_picture_share
    )


def font_set(font_dirs):
    """find_fonts plus every other Noto family with real glyph coverage (Kufi, Naskh, Nastaliq, Rashi, ...)."""
    base = find_fonts(tuple(font_dirs))
    known = set(base)
    dirs = [os.path.expanduser(d) for d in font_dirs]
    extra = sorted(
        f
        for d in dirs
        if os.path.isdir(d)
        for ext in ("ttf", "otf")
        for f in glob.glob(f"{d}/**/*.{ext}", recursive=True)
        if os.path.basename(f).lower().startswith("noto") and f not in known and len(_font_cmap(f)) >= C.min_noto_cmap
    )
    save_font_info()
    return base + tuple(extra)
