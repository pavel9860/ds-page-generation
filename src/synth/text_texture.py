"""Flat text-page texture: corpus loading + glyph rendering (numba blit).
Feeds the same camera/UV/photo pipeline as texture.py's grid ink, via a
different flat texture."""
import glob
import os

import cv2
import numpy as np
from fontTools.ttLib import TTFont
from numba import njit
from numba.typed import List as NumbaList
from PIL import ImageFont

from . import config as cfg
from .texture import _old_crease_field, make_old_creases

_FONT_DIRS = ("/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"),
              "/kaggle/input")

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

# Script-specific families the corpus needs (CJK, Arabic/Persian, Hebrew,
# Thai) that don't carry Latin+Cyrillic+Greek, checked against their own
# script's codepoints instead of _REQUIRED_CODEPOINTS.
_SCRIPT_FONT_GROUPS = (
    (_TRUSTED_FAMILY_TOKENS, _REQUIRED_CODEPOINTS),
    (("notosanscjk", "notoserifcjk", "sourcehansans", "sourcehanserif", "droidsansfallback", "wqy"),
     tuple(ord(c) for c in "汉字你好一二三四五")),
    (("notosansarabic", "notonaskharabic", "notokufiarabic"),
     tuple(ord(c) for c in "ابجدهوزح")),
    (("notosanshebrew", "notoserifhebrew"),
     tuple(ord(c) for c in "אבגדהוזח")),
    (("notosansthai", "notoserifthai", "notoloopedthai"),
     tuple(ord(c) for c in "กขคงจฉช")),
)


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


def _covers_group(path: str, tokens: tuple, codepoints: tuple) -> bool:
    name = os.path.basename(path).lower()
    if not any(tok in name for tok in tokens):
        return False
    cmap = _font_cmap(path)
    return all(cp in cmap for cp in codepoints)


def find_fonts() -> tuple:
    """.ttf/.ttc/.otf files from known families, verified via real cmap
    against each family group's own script (Latin+Cyrillic+Greek for the
    general body-text families, CJK/Arabic/Hebrew/Thai for their own
    dedicated families)."""
    candidates = sorted(p for d in _FONT_DIRS if os.path.isdir(d)
                        for ext in ("ttf", "ttc", "otf")
                        for p in glob.glob(f"{d}/**/*.{ext}", recursive=True))
    found = tuple(p for p in candidates
                 if any(_covers_group(p, tokens, cps) for tokens, cps in _SCRIPT_FONT_GROUPS))
    if not found:
        raise FileNotFoundError(f"no supported fonts found under {_FONT_DIRS}")
    return found


def load_corpus(texts_dir: str, min_words: int = None) -> list:
    """Excludes files with fewer usable (Latin) words than the snippet
    budget -- below that, sample_snippet's start index always clamps to
    0, so every seed drawing that file gets the identical snippet."""
    paths = sorted(glob.glob(os.path.join(texts_dir, "*", "*.txt")))
    if not paths:
        raise RuntimeError(f"no .txt files under {texts_dir}")
    thresh = min_words if min_words is not None else _SNIPPET_WORD_BUDGET
    usable = [p for p in paths if len(_words(p)) >= thresh]
    if not usable:
        raise RuntimeError(f"no .txt files under {texts_dir} with >= {thresh} usable words")
    return usable


def _has_letters(word: str) -> bool:
    """Drop pure digit/punctuation tokens (index/footnote/page-number
    listings -- not prose). No script filtering: Latin, Cyrillic, and
    Greek words are all kept -- find_fonts only qualifies fonts covering
    all three."""
    return any(c.isalpha() for c in word)


_WORDS_CACHE = {}


def _words(path):
    if path not in _WORDS_CACHE:
        with open(path, encoding="utf-8", errors="ignore") as f:
            raw = f.read().split()
        _WORDS_CACHE[path] = [w for w in raw if _has_letters(w)]
    return _WORDS_CACHE[path]


def _max_words_for_page(page_px=cfg.TEXT_PAGE_PX, page_mm=cfg.TEXT_PAGE_MM,
                        font_pt=min(cfg.TEXT_FONT_PT_RANGE),
                        line_spacing=min(cfg.TEXT_LINE_SPACING_RANGE),
                        margin_mm=cfg.TEXT_MARGIN_MM, avg_word_chars=5.5) -> int:
    """Word count for a full page at the densest packing (smallest font,
    tightest spacing) -- an overestimate is harmless, layout trims it."""
    px_per_mm = page_px / page_mm
    font_px = max(4, round(font_pt * 0.3528 * px_per_mm))
    line_h = max(font_px + 1, round(font_px * line_spacing))
    margin = round(margin_mm * px_per_mm)
    n_lines = max(1, (page_px - 2 * margin) // line_h)
    avg_char_px = font_px * 0.5   # conservative (narrow) average glyph advance
    chars_per_line = max(1.0, (page_px - 2 * margin) / avg_char_px)
    words_per_line = max(1.0, chars_per_line / (avg_word_chars + 1))
    return int(n_lines * words_per_line * 1.3)   # 30% safety margin


_SNIPPET_WORD_BUDGET = _max_words_for_page()


def sample_snippet(paths, rng, n_words=_SNIPPET_WORD_BUDGET) -> str:
    """n_words is an upper bound -- the layout stops at the bottom margin
    regardless of how much text is supplied."""
    words = _words(paths[rng.integers(0, len(paths))])
    start = int(rng.integers(0, max(1, len(words) - n_words)))
    return " ".join(words[start:start + n_words])


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


def _pick_font(rng, font_files: tuple, font_px: int, chars):
    """Random font among those whose real cmap covers every distinct char
    in this page's text -- find_fonts' probe codepoints don't guarantee
    coverage of every individual character (rare glyph gaps), and a script
    a handful of fonts support (CJK, Arabic, Hebrew, Thai) would almost
    never turn up under blind random retries against the whole (mostly
    Latin/Cyrillic/Greek) pool."""
    codepoints = {ord(c) for c in chars}
    covering = [p for p in font_files if codepoints <= _font_cmap(p)]
    pool = covering or font_files
    font_path = pool[rng.integers(0, len(pool))]
    return ImageFont.truetype(font_path, font_px), font_path


def render_flat_text(text: str, rng, page_px: int, page_mm: float, font_files: tuple,
                     font_pt_range=cfg.TEXT_FONT_PT_RANGE, margin_mm=cfg.TEXT_MARGIN_MM,
                     line_spacing_range=cfg.TEXT_LINE_SPACING_RANGE,
                     add_old_creases: bool = True) -> np.ndarray:
    """-> float32 (page_px, page_px) in [INK, PAPER]. Single-column, no
    kerning. Same tone convention and print defects (old-crease ink smear,
    fiber/grain noise) as texture.render_grid_texture. `add_old_creases`:
    the 2D print-stage old-crease ink-smear defect; False skips it
    entirely (changes the rng draw sequence vs. the flag being on)."""
    px_per_mm = page_px / page_mm
    pt = rng.uniform(*font_pt_range)
    font_px = max(4, round(pt * 0.3528 * px_per_mm))   # 1pt = 0.3528mm
    distinct_chars = set(text) - {" ", "\n", "\t"}
    font, font_path = _pick_font(rng, font_files, font_px, distinct_chars)
    font_key = (font_path, font_px)
    margin = round(margin_mm * px_per_mm)
    line_h = max(font_px + 1, round(font_px * rng.uniform(*line_spacing_range)))
    space_w = _glyph(font, font_key, " ")[2]

    w = h = page_px
    y = margin
    max_x, max_y = w - margin, h - margin
    lines, line, line_w = [], [], 0.0
    for word in text.split():
        glyphs = [_glyph(font, font_key, ch) for ch in word]
        word_w = sum(g[2] for g in glyphs)
        new_w = line_w + (space_w if line else 0.0) + word_w
        if new_w > (max_x - margin):
            lines.append(line)
            y += line_h
            line = [(word, glyphs)]
            line_w = word_w
            if y + line_h > max_y:
                line = []
                break
        else:
            line.append((word, glyphs))
            line_w = new_w
    if line and y + line_h <= max_y:
        lines.append(line)

    page = np.full((h, w), 255, dtype=np.uint8)
    n_max = sum(len(word) for ln in lines for word, _ in ln)
    arrs = [None] * n_max
    fx = np.empty(n_max)
    fy = np.empty(n_max, dtype=np.int64)
    fox = np.empty(n_max, dtype=np.int64)
    foy = np.empty(n_max, dtype=np.int64)
    fgw = np.empty(n_max, dtype=np.int64)
    fgh = np.empty(n_max, dtype=np.int64)
    n = 0
    y = margin
    for ln in lines:
        x = float(margin)
        for i, (word, glyphs) in enumerate(ln):
            if i > 0:
                x += space_w
            for arr, (ox, oy), advance in glyphs:
                if arr.size:
                    arrs[n] = arr
                    fx[n], fy[n], fox[n], foy[n] = x, y, ox, oy
                    fgh[n], fgw[n] = arr.shape
                    n += 1
                x += advance
        y += line_h

    if n:
        x0 = np.round(fx[:n]).astype(np.int64) + fox[:n]
        y0 = fy[:n] + foy[:n]
        xs_a = np.maximum(0, x0)
        ys_a = np.maximum(0, y0)
        xe_a = np.minimum(w, x0 + fgw[:n])
        ye_a = np.minimum(h, y0 + fgh[:n])
        keep = np.nonzero((xe_a > xs_a) & (ye_a > ys_a))[0]
        if keep.size:
            _blit_glyphs(page, NumbaList([arrs[i] for i in keep]),
                        xs_a[keep], ys_a[keep], xe_a[keep], ye_a[keep], x0[keep], y0[keep])

    tex = cfg.INK + (cfg.PAPER - cfg.INK) * (page.astype(np.float32) / 255.0)

    old_creases = make_old_creases(rng, page_mm) if add_old_creases else []
    if old_creases:
        mx, my = np.meshgrid(np.arange(w, dtype=np.float32) / px_per_mm,
                             np.arange(h, dtype=np.float32) / px_per_mm)
        sigma = max(px_per_mm * cfg.OLD_CREASE_BLUR_SIGMA_SCALE, cfg.OLD_CREASE_BLUR_SIGMA_MIN)
        blurred = cv2.GaussianBlur(tex, (0, 0), sigma)
        blend = np.zeros_like(tex)
        for c in old_creases:
            blend = np.maximum(blend, _old_crease_field(c, mx, my))
        tex = tex * (1 - blend) + blurred * blend

    fib = cv2.GaussianBlur(rng.standard_normal(tex.shape, dtype=np.float32), (0, 0),
                           cfg.NOISE_FIB_BLUR_SIGMA)
    grain = rng.standard_normal(tex.shape, dtype=np.float32)
    tex = tex + cfg.NOISE_FIB_AMP * fib + cfg.NOISE_GRAIN_AMP * grain
    return np.clip(tex, 0, 1).astype(np.float32)
