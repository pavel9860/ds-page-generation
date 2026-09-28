"""Text pages: font discovery by real cmap coverage, snippets from book texts, glyph layout and blitting."""
import glob
import os
from functools import lru_cache

import numpy as np
from fontTools.ttLib import TTFont
from numba import njit
from numba.typed import List as NumbaList
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


@lru_cache(maxsize=1)
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


def char_budget(w_px, h_px, px_per_mm, font_pt, line_spacing):
    """Upper bound of the characters a page holds at its densest packing."""
    font_px = max(4, round(font_pt * 0.3528 * px_per_mm))
    lines = h_px / max(font_px + 1, round(font_px * line_spacing))
    return int(1.3 * lines * w_px / (0.5 * font_px))


def sample_snippet(path, rng, budget) -> str:
    """Contiguous words from a random place of the text file, about budget characters, wrapping around."""
    words = _words(path)
    if not words:
        return ""
    start, out, total = int(rng.integers(len(words))), [], 0
    for i in range(len(words)):
        out.append(words[(start + i) % len(words)])
        total += len(out[-1]) + 1
        if total >= budget:
            break
    return " ".join(out)


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


def render_text(text: str, rng, w_px: int, h_px: int, px_per_mm: float, font_files: tuple, font_pt, line_spacing):
    """-> uint8 (h_px, w_px) page, 0 ink .. 255 paper. Single column filled to the bottom, no kerning;
    left, right or justified alignment."""
    font_px = max(4, round(font_pt * 0.3528 * px_per_mm))
    font, font_path = _pick_font(rng, font_files, font_px, set(text) - {" ", "\n", "\t"})
    rtl = is_rtl(text)
    font_key = (font_path, font_px)
    line_h = max(font_px + 1, round(font_px * line_spacing))
    space_w = _glyph(font, font_key, " ")[2]
    lines, line, line_w, y = [], [], 0.0, 0
    for word in text.split():
        glyphs = [_glyph(font, font_key, ch) for ch in (get_display(word) if rtl else word)]
        for piece_word, piece_glyphs in _wrap_pieces(word, glyphs, w_px):
            piece_w = sum(g[2] for g in piece_glyphs)
            new_w = line_w + (space_w if line else 0.0) + piece_w
            if new_w <= w_px:
                line.append((piece_word, piece_glyphs))
                line_w = new_w
                continue
            lines.append(line)
            y += line_h
            line, line_w = [(piece_word, piece_glyphs)], piece_w
            if y + line_h > h_px:
                break
        if y + line_h > h_px:
            break
    else:
        if line:
            lines.append(line)

    page = np.full((h_px, w_px), 255, dtype=np.uint8)
    items = []
    align = rng.choice(("right", "justify")) if rtl else rng.choice(("left", "right", "justify"))
    for i, ln in enumerate(lines):
        ln = ln[::-1] if rtl else ln
        natural = sum(g[2] for _, gl in ln for g in gl) + space_w * max(0, len(ln) - 1)
        x, gap = 0.0, space_w
        if align == "right":
            x = float(w_px - natural)
        elif align == "justify" and len(ln) > 1 and i < len(lines) - 1:
            gap = space_w + (w_px - natural) / (len(ln) - 1)
        for k, (_, glyphs) in enumerate(ln):
            x += gap if k else 0.0
            for arr, (ox, oy), advance in glyphs:
                if arr.size:
                    items.append((arr, round(x) + ox, i * line_h + oy))
                x += advance
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
