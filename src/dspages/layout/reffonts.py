"""Reference fonts: the full font files of fonts that PDFs embed as subsets with a wrong text map (Word's Mangal, Nikosh, Iskoola Pota ...).
A glyph of an embedded subset is found in the full font by its outline (the same drawing: exact, whatever the glyph numbers of the subset), and its
text is the full font's (cmap, and through GSUB the conjuncts and half forms built of mapped glyphs: refont._font_glyph_text). The text is in drawing
order, as a text map gives it (refont puts Indic signs in logical order).

Files: DSPAGES_FONTS_REF (default <UVTM>/fonts_ref), *.ttf / *.otf, known by the family of their name."""

import hashlib
import os
from functools import lru_cache
from pathlib import Path

from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.ttLib import TTFont

from ..config import UVTM
from . import legacy
from .config import MAX_APART  # noqa: F401

DIR = Path(os.environ.get("DSPAGES_FONTS_REF", f"{UVTM}/fonts_ref"))


def _hasher(data):
    """glyph id -> hash of its outline (decomposed, in units of 1/1000 em); None for a glyph without ink."""
    gs, upem, names = legacy._outlines(data)

    def h(g):
        if not 0 <= g < len(names):
            return None
        pen = DecomposingRecordingPen(gs)
        try:
            gs[names[g]].draw(pen)
        except Exception:  # a glyph the program cannot draw: not matched
            return None
        if not pen.value:
            return None
        k = 1000 / upem
        rec = [
            (op, tuple(None if pt is None else (round(pt[0] * k), round(pt[1] * k)) for pt in pts))
            for op, pts in pen.value
        ]  # None: the implied
        # on-curve point of a contour of off-curve points only (qCurveTo)
        return hashlib.md5(repr(rec).encode()).hexdigest()[:16]

    return h


@lru_cache(maxsize=1)
def references():
    """{family key: [({outline hash: (glyph id, text)}, {glyph id: text}, path) per file]} of the reference fonts (one family can have several versions).
    Kept in DIR/.references.pkl, made again when a font file changes (8 s to build, read in 0.1 s)."""
    import pickle

    fs = sorted(DIR.glob("*.[ot]tf")) if DIR.is_dir() else []
    sig = [(f.name, f.stat().st_size, f.stat().st_mtime_ns) for f in fs]
    cache = DIR / ".references.pkl"
    if cache.exists():
        got = pickle.loads(cache.read_bytes())
        if got["sig"] == sig:
            return got["out"]
    out = _references()
    if DIR.is_dir():
        tmp = cache.with_suffix(f".{os.getpid()}")
        tmp.write_bytes(pickle.dumps(dict(sig=sig, out=out)))
        tmp.replace(cache)
    return out


def _references():
    from .refont import _font_glyph_text

    out = {}
    for f in sorted(DIR.glob("*.[ot]tf")) if DIR.is_dir() else ():
        data = f.read_bytes()
        names = TTFont(f, lazy=True)["name"]
        fam = legacy.family(str(names.getDebugName(6) or names.getDebugName(4) or f.stem))
        text, direct, n = _font_glyph_text(data)
        h = _hasher(data)
        by_hash = {}
        for g in sorted(
            text, key=lambda g: (g not in direct, g)
        ):  # an outline drawn by several glyphs: the one the cmap names
            k = h(g) if g < n else None
            if k:
                by_hash.setdefault(k, (g, text[g]))
        out.setdefault(fam, []).append((by_hash, text, str(f)))
    return out


def glyph_texts(name, data):
    """({glyph id: (text, glyph id in the reference)}, reference path) of an embedded program of the font `name`; ({}, None) when there is none. A glyph is found by its
    outline in the version that draws most of the program's glyphs; when those glyphs have the same numbers there (a subset that keeps the full
    font's glyph ids), the glyphs of other outlines (another release, other hinting) are named by their number in that version."""
    refs = references().get(legacy.family(name))
    if not refs or not data:
        return {}, None
    try:
        h = _hasher(data)
        n = TTFont(__import__("io").BytesIO(data), lazy=True)["maxp"].numGlyphs
    except Exception as e:
        print("reference: program unreadable:", name, repr(e)[:100], flush=True)
        return {}, None
    hashes = {g: k for g in range(n) if (k := h(g))}
    by_hash, by_id, path = max(refs, key=lambda r: sum(k in r[0] for k in hashes.values()))
    hit = {g: by_hash[k] for g, k in hashes.items() if k in by_hash}
    out = {g: (t, rg) for g, (rg, t) in hit.items()}
    if hit and sum(rg == g for g, (rg, _) in hit.items()) >= 0.9 * len(hit):
        out.update({g: (by_id[g], g) for g in hashes if g not in out and g in by_id})
    return out, path


@lru_cache(maxsize=16)
def _hb(path):
    import uharfbuzz as hb

    return hb.Font(hb.Face(hb.Blob.from_file_path(path)))


def shaped(path, text):
    """[(glyph id, x of its origin in em)] the reference font draws for a text (HarfBuzz)."""
    import uharfbuzz as hb

    font = _hb(path)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf)
    upem, x, out = font.face.upem, 0, []
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        out.append((info.codepoint, (x + pos.x_offset) / upem))
        x += pos.x_advance
    return out


def word(path, pieces):
    """The Unicode text of a word from its glyphs in drawing order [(text, reference glyph id, x of the origin in em)]: an order of the glyph texts
    that the reference font shapes (HarfBuzz) into exactly these glyphs in this order (the font's own rules decide where a prefix vowel, a reph, a below
    form goes). (text, True) when one is found, else (the Indic reading of the drawing order, False: counted as unverified)."""
    from . import indic

    want = [(g, x) for _, g, x in pieces]
    texts = [t.replace(indic.REPH, "") for t, _, _ in pieces]
    drawn = "".join(t for t, _, _ in pieces)
    first = (indic.to_logical(drawn) if indic.visual_order(drawn) else drawn).replace(indic.REPH, "")
    cands, n = {first: 0, "".join(texts): 0}, len(texts)
    orders = [list(range(n))]
    for moved in range(1, MAX_APART + 1):  # the glyph texts with one, then two neighbours swapped
        orders = [o[:i] + [o[i + 1], o[i]] + o[i + 2 :] for o in orders for i in range(n - 1)]
        for o in orders:
            cands.setdefault("".join(texts[i] for i in o), moved)
    for cand in cands:
        if [g for g, _ in shaped(path, cand)] == [g for g, _ in want]:
            return cand, True
    return first, False
