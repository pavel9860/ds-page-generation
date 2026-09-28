"""Text pages: font discovery by real cmap coverage, snippets from book texts, glyph layout and blitting."""
import glob
import os
import re
from functools import lru_cache

import numpy as np
from fontTools.ttLib import TTFont
from numba import njit
from numba.typed import List as NumbaList
from arabic_reshaper import reshape
from bidi import get_display
from PIL import ImageFont


# Font family-name blocklists are a losing game -- font collections ship
# dozens of single-script fonts (e.g. NotoSansElbasan, an extinct
# Albanian script) whose names don't obviously signal "not general body
# text". Only trust a named allowlist of known general-purpose families,
# verified against their real cmap (not a rendered-bitmap heuristic,
# which a font can satisfy with generic placeholder glyphs).
_TRUSTED_FAMILY_TOKENS = (
    "dejavusans", "dejavuserif", "liberationsans", "liberationserif", "liberationmono",
    "notosans-", "notoserif-", "notomono", "freeserif", "freesans", "freemono",
    "roboto", "opensans", "lato-", "ptsans", "ptserif", "sourcesans", "sourceserif",
    "arial", "times", "verdana", "tahoma", "calibri", "cambria", "georgia", "consolas", "ubuntu",
)
_REQUIRED_CODEPOINTS = tuple(ord(c) for c in "AaZz09АаЯяЁёΑαωω")
_MIN_SCRIPT_CMAP_SIZE = 40


_CMAP_CACHE = {}


def _font_cmap(path: str) -> set:
    cmap = _CMAP_CACHE.get(path)
    if cmap is None:
        try:
            cmap = set(TTFont(path, fontNumber=0, lazy=True).getBestCmap() or ())
        except Exception:
            cmap = set()
        _CMAP_CACHE[path] = cmap
    return cmap


def _is_general_font(path: str) -> bool:
    name = os.path.basename(path).lower()
    if not any(tok in name for tok in _TRUSTED_FAMILY_TOKENS):
        return False
    cmap = _font_cmap(path)
    return all(cp in cmap for cp in _REQUIRED_CODEPOINTS)


def _is_script_font(path: str) -> bool:
    """Any Noto Sans/Serif script-specific family (Bengali, Devanagari,
    Arabic, CJK, ...) with real, non-trivial glyph coverage -- accepted
    without hand-listing every script, since _pick_font checks actual
    text codepoints against each font's own cmap at render time."""
    name = os.path.basename(path).lower()
    if not (name.startswith("notosans") or name.startswith("notoserif")):
        return False
    return len(_font_cmap(path)) >= _MIN_SCRIPT_CMAP_SIZE


@lru_cache(maxsize=4)
def find_fonts(font_dirs: tuple) -> tuple:
    """.ttf/.ttc/.otf files of known families, verified via their real cmap: Latin + Cyrillic + Greek for the
    general body-text families, any real glyph coverage for Noto's script-specific families."""
    dirs = [os.path.expanduser(d) for d in font_dirs]
    candidates = sorted(p for d in dirs if os.path.isdir(d) for ext in ("ttf", "ttc", "otf")
                        for p in glob.glob(f"{d}/**/*.{ext}", recursive=True))
    found = tuple(p for p in candidates if _is_general_font(p) or _is_script_font(p))
    if not found:
        raise FileNotFoundError(f"no supported fonts under {dirs}")
    return found


def _has_letters(word: str) -> bool:
    """Drop pure digit/punctuation tokens (index/footnote/page-number
    listings -- not prose). No script filtering: Latin, Cyrillic, and
    Greek words are all kept -- find_fonts only qualifies fonts covering
    all three."""
    return any(c.isalpha() for c in word)


@lru_cache(maxsize=32)
def _words(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        raw = f.read().split()
    return [w for w in raw if _has_letters(w)]


@lru_cache(maxsize=8)
def book_text(path) -> str:
    """The book's words with letters, space-joined, paragraph breaks (blank lines) kept as newlines."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        pars = re.split(r"\n\s*\n", f.read())
    return "\n".join(p for p in (" ".join(w for w in q.split() if _has_letters(w)) for q in pars) if p)


def book_snippet(path, offset, chars) -> str:
    """`chars` characters of the book from `offset` (wrapping to the start), a word cut at the start dropped."""
    t = book_text(path)
    s = (t[offset:] + "\n" + t)[:chars + 64] if offset else t[:chars + 64]
    return s.split(" ", 1)[-1] if offset else s


@njit(cache=True)
def _blit_glyphs(page, arrs, xs_a, ys_a, xe_a, ye_a, x0, y0):
    """Darken-composite glyph bitmaps onto page (min-blend)."""
    for k in range(len(arrs)):
        arr = arrs[k]
        xs, ys, xe, ye, px0, py0 = xs_a[k], ys_a[k], xe_a[k], ye_a[k], x0[k], y0[k]
        for row in range(ys, ye):
            for col in range(xs, xe):
                v = np.uint8(255) - arr[row - py0, col - px0]
                if v < page[row, col]:
                    page[row, col] = v


_GLYPH_CACHE = {}


def _glyph(font, font_key, ch):
    cache = _GLYPH_CACHE.setdefault(font_key, {})
    entry = cache.get(ch)
    if entry is None:
        mask, offset = font.getmask2(ch, mode="L")
        arr = (np.array(mask).reshape(mask.size[1], mask.size[0])
               if mask.size[0] and mask.size[1] else np.zeros((0, 0), dtype=np.uint8))
        entry = (arr, offset, font.getlength(ch))
        cache[ch] = entry
    return entry


def _font_family_key(path: str) -> str:
    """Family name ignoring weight/width/style suffixes, so a family
    shipping dozens of weight x width files (Noto Sans's ~70+ variants)
    doesn't drown out a family with only one or two (Times, Arial) under
    uniform-random selection."""
    return os.path.basename(path).split("-")[0].split(",")[0].lower()


def _pick_font(rng, font_files: tuple, font_px: int, chars):
    """Random family, then random file within it, among those whose real
    cmap covers every distinct char in this page's text -- find_fonts'
    probe codepoints don't guarantee coverage of every individual
    character (rare glyph gaps), and a script a handful of fonts support
    (CJK, Arabic, Hebrew, Thai, ...) would almost never turn up under
    blind random retries against the whole (mostly Latin/Cyrillic/Greek)
    pool. Selecting by family first also keeps a family's weight/width
    variants (thin, condensed, black, ...) from being over- or
    under-represented relative to single-style families just because it
    happens to ship more files. If no font covers every char (e.g. text
    mixing scripts, or stray presentation-form codepoints outside a
    script font's base cmap), fall back to whichever font(s) cover the
    most of them -- not a uniformly random font from the whole pool, which
    could be entirely unrelated to the text's actual script."""
    codepoints = {ord(c) for c in chars}
    covering = [p for p in font_files if codepoints <= _font_cmap(p)]
    if covering:
        pool = covering
    else:
        scored = [(len(codepoints & _font_cmap(p)), p) for p in font_files]
        best_score = max(s for s, _ in scored)
        pool = [p for s, p in scored if s == best_score]

    families = {}
    for p in pool:
        families.setdefault(_font_family_key(p), []).append(p)
    family_keys = sorted(families)
    chosen_family = family_keys[rng.integers(0, len(family_keys))]
    family_files = families[chosen_family]
    font_path = family_files[rng.integers(0, len(family_files))]
    return ImageFont.truetype(font_path, font_px), font_path


def _wrap_pieces(word: str, glyphs: list, max_w: float):
    """A whitespace-delimited token, split into pieces each narrower than
    max_w -- passes a normal word through unchanged, but breaks an
    unspaced-script token (a whole CJK/Thai/etc. line or paragraph
    collapsed into one `.split()` token) into character-level chunks so it
    still wraps across multiple lines instead of overflowing/clipping."""
    if sum(g[2] for g in glyphs) <= max_w:
        yield word, glyphs
        return
    buf_chars, buf_glyphs, buf_w = [], [], 0.0
    for ch, g in zip(word, glyphs):
        if buf_glyphs and buf_w + g[2] > max_w:
            yield "".join(buf_chars), buf_glyphs
            buf_chars, buf_glyphs, buf_w = [], [], 0.0
        buf_chars.append(ch)
        buf_glyphs.append(g)
        buf_w += g[2]
    if buf_glyphs:
        yield "".join(buf_chars), buf_glyphs


def is_rtl(text):
    letters = [ch for ch in text if ch.isalpha()]
    return bool(letters) and sum("\u0590" <= ch <= "\u08ff" for ch in letters) > 0.3 * len(letters)


def _flow(tokens, k, glyphs, width, height, line_h, space_w, indent_w, rtl, par_gap=False):
    """Lines of one column from tokens[k:] (None = paragraph break) -> (lines, next k). A line is
    (pieces, indent, last line of its paragraph)."""
    lines, line, lw, ind = [], [], 0.0, 0.0
    while k < len(tokens) and (len(lines) + 1) * line_h <= height:
        t = tokens[k]
        if t is None:
            if line:
                lines.append((line, ind, True))
                if par_gap and (len(lines) + 1) * line_h <= height:
                    lines.append(([], 0.0, True))
            line, lw, ind, k = [], indent_w, indent_w, k + 1
            continue
        t = get_display(reshape(t)) if rtl else t
        pieces = list(_wrap_pieces(t, glyphs(t), width - ind))
        done = True
        for j, (pw_, pg) in enumerate(pieces):
            w = sum(g[2] for g in pg)
            if lw + (space_w if line else 0.0) + w <= width:
                line.append((pw_, pg))
                lw += (space_w if line[:-1] else 0.0) + w
                continue
            lines.append((line, ind, False))
            line, lw, ind = [(pw_, pg)], w, 0.0
            if (len(lines) + 1) * line_h > height:
                tokens[k] = "".join(p for p, _ in pieces[j:]) if j else t
                done = False
                break
        k += done
    if line and (len(lines) + 1) * line_h <= height:
        lines.append((line, ind, k >= len(tokens)))
    return lines, k


def _draw(items, lines, x0, y0, width, line_h, space_w, align, rtl):
    for i, (ln, ind, last) in enumerate(lines):
        ln = ln[::-1] if rtl else ln
        natural = sum(g[2] for _, gl in ln for g in gl) + space_w * max(0, len(ln) - 1)
        free, gap = width - ind - natural, space_w
        if align == "justify" and len(ln) > 1 and not last:
            x, gap = (0.0 if rtl else ind), space_w + free / (len(ln) - 1)
        elif align == "right" or rtl:
            x = free + (0.0 if rtl else ind)
        elif align == "center":
            x = ind + free / 2
        else:
            x = ind
        for k, (_, glyphs) in enumerate(ln):
            x += gap if k else 0.0
            for arr, (ox, oy), advance in glyphs:
                if arr.size:
                    items.append((arr, x0 + round(x) + ox, y0 + i * line_h + oy))
                x += advance


def render_text(text: str, rng, w_px: int, h_px: int, px_per_mm: float, font_files: tuple, font_pt, line_spacing,
                f=None):
    """-> uint8 (h_px, w_px) page, 0 ink .. 255 paper, filled to the bottom. Formatting drawn from `f` (a
    TextFormatCfg; None = one column, no headings, no indents): column count and gutter, an optional heading
    across the columns, paragraph indents or blank lines between paragraphs, left / right / justified alignment
    (right or justified for RTL). No kerning."""
    font_px = max(4, round(font_pt * 0.3528 * px_per_mm))
    font, font_path = _pick_font(rng, font_files, font_px, set(text) - {" ", "\n", "\t"})
    rtl = is_rtl(text)

    def glyph_fn(fnt, key):
        return lambda word: [_glyph(fnt, key, ch) for ch in word]
    body = glyph_fn(font, (font_path, font_px))
    space_w = _glyph(font, (font_path, font_px), " ")[2]
    line_h = max(font_px + 1, round(font_px * line_spacing))
    ncol = int(rng.choice([c for c, _ in f.columns], p=_p(f.columns))) if f else 1
    gutter = round(rng.uniform(*f.gutter_mm) * px_per_mm) if f and ncol > 1 else 0
    col_w = (w_px - gutter * (ncol - 1)) // ncol
    if col_w < 14 * font_px:
        ncol, gutter, col_w = 1, 0, w_px
    aligns = ("right", "justify") if rtl else ("left", "right", "justify") if ncol == 1 else ("left", "justify")
    align = rng.choice(aligns)
    indent_w = (rng.choice((0.0, rng.uniform(1.0, 3.0))) * font_px) if f else 0.0
    par_gap = f is not None and indent_w == 0.0 and rng.random() < 0.5
    tokens = []
    for par in text.split("\n"):
        tokens += par.split() + [None]
    items, y0 = [], 0
    if f and rng.random() < f.heading_prob:
        hp = round(font_px * rng.uniform(*f.heading_scale))
        hkey = (font_path, hp)
        hfont = ImageFont.truetype(font_path, hp)
        n = int(rng.integers(2, 9))
        head = [t for t in tokens[:n] if t]
        hl, _ = _flow(head, 0, glyph_fn(hfont, hkey), w_px, 3 * round(hp * 1.2), round(hp * 1.2),
                      _glyph(hfont, hkey, " ")[2], 0.0, rtl)
        _draw(items, hl, 0, 0, w_px, round(hp * 1.2), _glyph(hfont, hkey, " ")[2],
              rng.choice(("left", "center")) if not rtl else "right", rtl)
        y0 = len(hl) * round(hp * 1.2) + line_h
        tokens = tokens[n:]
    k = 0
    for c in range(ncol):
        x0 = (ncol - 1 - c if rtl else c) * (col_w + gutter)
        lines, k = _flow(tokens, k, body, col_w, h_px - y0, line_h, space_w, indent_w, rtl, par_gap)
        _draw(items, lines, x0, y0, col_w, line_h, space_w, align, rtl)
    page = np.full((h_px, w_px), 255, dtype=np.uint8)
    if items:
        x0 = np.array([it[1] for it in items], np.int64)
        y0 = np.array([it[2] for it in items], np.int64)
        gw = np.array([it[0].shape[1] for it in items], np.int64)
        gh = np.array([it[0].shape[0] for it in items], np.int64)
        xs, ys = np.maximum(0, x0), np.maximum(0, y0)
        xe, ye = np.minimum(w_px, x0 + gw), np.minimum(h_px, y0 + gh)
        keep = np.nonzero((xe > xs) & (ye > ys))[0]
        if keep.size:
            _blit_glyphs(page, NumbaList([items[i][0] for i in keep]), xs[keep], ys[keep], xe[keep], ye[keep],
                         x0[keep], y0[keep])
    return page


def _p(table):
    w = np.array([x for _, x in table], float)
    return w / w.sum()
