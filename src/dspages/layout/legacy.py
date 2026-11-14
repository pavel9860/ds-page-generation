"""Text of legacy-font PDFs (Kruti Dev, Preeti, Bijoy, Anmol Lipi, Bamini, old Cyrillic / Armenian / Georgian / Hebrew / Arabic / Thai / Khmer ... fonts,
and Han / kana / Hangul fonts with a private encoding: letters typed in other codes, no Unicode anywhere in the file) read by the SHAPE of the glyphs.
The glyph outlines of the embedded font program are drawn and matched to the glyphs of the Noto font of the script (every glyph of it with its text: the
Unicode cmap and, through GSUB, the half forms and conjuncts; for Han, kana and Hangul every character of Noto Sans CJK rendered). Each glyph keeps its
nearest candidates and the choice among them is fitted on the document's own text: the candidate under which more of the text reads as words of the
language (the lexicon: wordfreq or the real paragraphs of the language). The text of a run is its glyphs' texts in order, with the Indic rules (a half
form + the stem sign makes the full letter, prefix vowel signs and reph into logical order: indic.to_logical). A font whose characters already are
letters of the script (real Unicode) is left alone."""
import collections
import re
import unicodedata
import contextvars
import io
import struct
import math
import os
import pickle
from functools import lru_cache, partial

import numpy as np
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont
from hwsynth.scripts import script as script_of
from PIL import Image, ImageDraw

from . import indic

FONTS = "/usr/share/fonts/truetype/noto"
CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
CACHE = os.path.join(os.environ.get("DSPAGES_LAYOUTS", "/run/media/p/D/ML_DS/UVTM/Layouts"), "archive", "info", "legacy")
# script -> (Noto font, Unicode ranges of its letters, signs and digits, longest text a glyph may stand for)
SPECS = {
    "Devanagari": ("NotoSansDevanagari-Regular.ttf", [(0x0900, 0x097F)], 3), "Bengali": ("NotoSansBengali-Regular.ttf", [(0x0980, 0x09FF)], 3),
    "Gurmukhi": ("NotoSansGurmukhi-Regular.ttf", [(0x0A00, 0x0A7F)], 3), "Gujarati": ("NotoSansGujarati-Regular.ttf", [(0x0A80, 0x0AFF)], 3),
    "Oriya": ("NotoSansOriya-Regular.ttf", [(0x0B00, 0x0B7F)], 3), "Tamil": ("NotoSansTamil-Regular.ttf", [(0x0B80, 0x0BFF)], 3),
    "Telugu": ("NotoSansTelugu-Regular.ttf", [(0x0C00, 0x0C7F)], 3), "Kannada": ("NotoSansKannada-Regular.ttf", [(0x0C80, 0x0CFF)], 3),
    "Malayalam": ("NotoSansMalayalam-Regular.ttf", [(0x0D00, 0x0D7F)], 3), "Sinhala": ("NotoSansSinhala-Regular.ttf", [(0x0D80, 0x0DFF)], 3),
    "Thai": ("NotoSansThai-Regular.ttf", [(0x0E00, 0x0E7F)], 2), "Lao": ("NotoSansLao-Regular.ttf", [(0x0E80, 0x0EFF)], 2),
    "Tibetan": ("NotoSerifTibetan-Regular.ttf", [(0x0F00, 0x0FFF)], 3), "Myanmar": ("NotoSansMyanmar-Regular.ttf", [(0x1000, 0x109F)], 3),
    "Khmer": ("NotoSansKhmer-Regular.ttf", [(0x1780, 0x17FF)], 3), "Ethiopic": ("NotoSansEthiopic-Regular.ttf", [(0x1200, 0x139F)], 1),
    "Armenian": ("NotoSansArmenian-Regular.ttf", [(0x0530, 0x058F)], 1), "Georgian": ("NotoSansGeorgian-Regular.ttf", [(0x10A0, 0x10FF)], 1),
    "Hebrew": ("NotoSansHebrew-Regular.ttf", [(0x0590, 0x05FF)], 1), "Arabic": ("NotoNaskhArabic-Regular.ttf", [(0x0600, 0x06FF)], 2),
    "Cyrillic": ("NotoSans-Regular.ttf", [(0x0400, 0x04FF)], 1), "Greek": ("NotoSans-Regular.ttf", [(0x0370, 0x03FF), (0x1F00, 0x1FFF)], 1),        # + Greek Extended (polytonic)
    "Latin": ("NotoSans-Regular.ttf", [(0x0020, 0x024F)], 1),
}
CJK = {"Han": [(0x4E00, 0x9FFF)], "Kana": [(0x3040, 0x30FF)], "Hangul": [(0xAC00, 0xD7A3)]}
SCRIPTS = dict.fromkeys(SPECS) | dict.fromkeys(CJK)
GRID = 24
GEO = np.array([3.0, 10.0, 10.0], np.float32)      # weights of (log aspect, vertical place, height) against the shape distance
PARAS = "/run/media/p/D/ML_DS/UVTM/handwriting/handwriting_detection/langds/paras"
SIZE = 96


class _Flat(BasePen):
    """Outline of a glyph as polylines."""
    def __init__(self, glyphs):
        super().__init__(glyphs)
        self.contours = []

    def _moveTo(self, p):
        self.contours.append([p])

    def _lineTo(self, p):
        self.contours[-1].append(p)

    def _curveToOne(self, a, b, c):
        p0 = self.contours[-1][-1]
        for t in np.linspace(0, 1, 9)[1:]:
            u = 1 - t
            self.contours[-1].append((u**3 * p0[0] + 3 * u * u * t * a[0] + 3 * u * t * t * b[0] + t**3 * c[0],
                                      u**3 * p0[1] + 3 * u * u * t * a[1] + 3 * u * t * t * b[1] + t**3 * c[1]))

    def _qCurveToOne(self, a, b):
        p0 = self.contours[-1][-1]
        for t in np.linspace(0, 1, 7)[1:]:
            u = 1 - t
            self.contours[-1].append((u * u * p0[0] + 2 * u * t * a[0] + t * t * b[0], u * u * p0[1] + 2 * u * t * a[1] + t * t * b[1]))

    def _closePath(self):
        pass


def ink_feature(ink, base):
    """(vector, geometry) of a glyph's ink (a bool canvas, baseline at row `base`): the ink box cut and fitted to GRID x GRID, its aspect, its height and
    vertical place in the em. None: no ink."""
    ys, xs = np.nonzero(ink)
    if not len(ys):
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    h, w = y1 - y0, x1 - x0
    crop = Image.fromarray((ink[y0:y1, x0:x1] * 255).astype(np.uint8)).resize((GRID, GRID), Image.BILINEAR)
    geo = np.array([math.log(w / h), (base - y1) / SIZE, h / SIZE], np.float32)
    return np.asarray(crop, np.float32).ravel() / 255.0, geo


BASE = SIZE + SIZE // 2                                                 # row of the baseline on the ink canvas


def feature(glyphs, name, upem):
    """Feature of one glyph outline of a font program (see ink_feature); None: no ink or no outline."""
    pen = _Flat(glyphs)
    try:
        glyphs[name].draw(pen)
    except Exception:
        return None
    if not pen.contours:
        return None
    s = SIZE / upem
    ink = np.zeros((2 * SIZE, 2 * SIZE), bool)
    for c in pen.contours:
        if len(c) < 3:
            continue
        m = Image.new("L", ink.shape, 0)
        ImageDraw.Draw(m).polygon([(SIZE // 2 + x * s, BASE - y * s) for x, y in c], fill=1)
        ink ^= np.asarray(m, bool)                                       # even-odd: holes
    return ink_feature(ink, BASE)


EXTRA = {"Devanagari": ["\u094d\u0930"], "Tamil": ["டி", "டீ", "ணி", "ணீ"]}                  # texts no Noto glyph has alone (the sign under a letter: ra after a virama), tried for every glyph by the fit
PUNCT = ",.;:!?()[]{}-\u2013/'\"%&+=*0123456789"
RARE = {0x0900, 0x093A, 0x093B, 0x0946, 0x094A, 0x094E, 0x094F} | set(range(0x0950, 0x0958)) | set(range(0x0971, 0x0980))     # Devanagari signs a legacy font does not have
CJK_LANG = {"zh": (2, ("Han",)), "ja": (0, ("Han", "Kana")), "ko": (1, ("Hangul", "Han"))}         # Noto Sans CJK collection index, character sets
LANG_SCRIPT = {"hi": "Devanagari", "mr": "Devanagari", "ne": "Devanagari", "sa": "Devanagari", "mai": "Devanagari", "bn": "Bengali", "as": "Bengali",
               "pa": "Gurmukhi", "gu": "Gujarati", "or": "Oriya", "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam", "si": "Sinhala",
               "th": "Thai", "lo": "Lao", "km": "Khmer", "my": "Myanmar", "bo": "Tibetan", "dz": "Tibetan", "am": "Ethiopic", "ti": "Ethiopic",
               "hy": "Armenian", "ka": "Georgian", "he": "Hebrew", "yi": "Hebrew", "ar": "Arabic", "fa": "Arabic", "ur": "Arabic", "ps": "Arabic",
               "sd": "Arabic", "ckb": "Arabic", "ug": "Arabic", "el": "Greek", "zh": "CJK-zh", "ja": "CJK-ja", "ko": "CJK-ko"}
for _l in "ru bg mk sr be kk tg ky mn uk".split():
    LANG_SCRIPT[_l] = "Cyrillic"


def ranges_of(script):
    if script.startswith("CJK-"):
        return [r for part in CJK_LANG[script[4:]][1] for r in CJK[part]]
    return SPECS[script][1]


def allowed(text, script):
    """A text a legacy glyph can stand for: at most a few letters of the script's Unicode ranges (no Vedic accents etc.); Latin / punctuation: one character."""
    rg = ranges_of(script)
    n = 1 if script.startswith("CJK-") else SPECS[script][2]
    if len(text) > n or not text.strip():
        return False
    if script == "Devanagari" and any((ord(c) - 0x0900) in RARE for c in text):
        return False
    return all(any(a <= ord(c) <= b for a, b in rg) for c in text)


def _cache_get(script):
    f = os.path.join(CACHE, f"ref_{script}.npz")
    if os.path.exists(f):
        z = np.load(f, allow_pickle=False)
        return str(z["T"]).split("\x00"), z["S"], z["G"]
    return None


def _cache_put(script, T, S, G):
    os.makedirs(CACHE, exist_ok=True)
    tmp = os.path.join(CACHE, f"ref_{script}.{os.getpid()}.tmp.npz")             # written whole, then renamed: processes building it at once never read half a file
    np.savez_compressed(tmp, T=np.array("\x00".join(T)), S=S, G=G)
    os.replace(tmp, os.path.join(CACHE, f"ref_{script}.npz"))


def _doc_cache(doc):
    """The decode decisions of a document kept on disk (CACHE/docs/<file stem>.pkl): {'tables': font_tables' {(font, script): (Table | None, glyphs seen)},
    'table_for': {font: table key | None}}; loaded once per open document, so every document is decoded once in all runs."""
    d = doc.__dict__
    if "_doc_cache" not in d:
        f = os.path.join(CACHE, "docs", os.path.splitext(os.path.basename(doc.name))[0] + ".pkl") if doc.name else None
        d["_doc_cache"] = (f, pickle.load(open(f, "rb")) if f and os.path.exists(f) else {"tables": {}, "table_for": {}})
    return d["_doc_cache"]


@lru_cache(maxsize=None)
def tables_sig(script):
    """A hash of the tables table_for weighs for the script (the decoders that write it, every labelled table): a choice holds only for them."""
    import hashlib
    from . import legacy_tables as LT
    keys = ["score:glotlid+trusted-shapes+charlm", "guard:program"] + sorted(k for k in LT.decoders() if LT.writes(k, script)) + sorted(f"{k}{sorted(v.items())}" for k, v in LT._labelled().items())
    return hashlib.sha1("\n".join(keys).encode()).hexdigest()[:12]


def _doc_save(doc):
    f, c = _doc_cache(doc)
    if f:
        os.makedirs(os.path.dirname(f), exist_ok=True)
        tmp = f"{f}.{os.getpid()}.tmp"                                  # written whole, then renamed
        pickle.dump(c, open(tmp, "wb"))
        os.replace(tmp, f)


def _punct(T, S, G):
    f2 = TTFont(f"{FONTS}/NotoSans-Regular.ttf")                       # punctuation and digits are typed in the same fonts
    cm, gs2, upem2 = f2.getBestCmap(), f2.getGlyphSet(), f2["head"].unitsPerEm
    for ch in PUNCT:
        r = feature(gs2, cm[ord(ch)], upem2) if ord(ch) in cm else None
        if r is not None:
            T.append(ch)
            S.append(r[0])
            G.append(r[1])


def _rendered(script):
    """Reference of Han / kana / Hangul: every character of the language's sets drawn with Noto Sans CJK (the glyph outlines of a collection font are not
    addressed by name here, so the characters are drawn as text)."""
    from PIL import ImageFont
    index, parts = CJK_LANG[script[4:]]
    font = ImageFont.truetype(CJK_FONT, SIZE, index=index)
    T, S, G = [], [], []
    for a, b in ranges_of(script):
        for cp in range(a, b + 1):
            ink = Image.new("L", (2 * SIZE, 2 * SIZE), 0)
            ImageDraw.Draw(ink).text((SIZE // 2, BASE), chr(cp), font=font, fill=1, anchor="ls")
            r = ink_feature(np.asarray(ink, bool), BASE)
            if r is not None:
                T.append(chr(cp))
                S.append(r[0])
                G.append(r[1])
    return T, S, G


@lru_cache(maxsize=24)
def reference(script):
    """(texts, shape matrix, geometry matrix) of the glyphs of the script's Noto font that have ink and a text; cached on disk (building Han takes a minute)."""
    got = _cache_get(script)
    if got:
        return got
    from .refont import _font_glyph_text
    if script.startswith("CJK-"):
        T, S, G = _rendered(script)
    else:
        data = open(f"{FONTS}/{SPECS[script][0]}", "rb").read()
        texts, _, n = _font_glyph_text(data)
        f = TTFont(io.BytesIO(data))
        gs, order, upem = f.getGlyphSet(), f.getGlyphOrder(), f["head"].unitsPerEm
        T, S, G = [], [], []
        for g, t in texts.items():
            if g < len(order) and allowed(t, script):
                r = feature(gs, order[g], upem)
                if r is not None:
                    T.append(t)
                    S.append(r[0])
                    G.append(r[1])
    if script != "Latin":
        _punct(T, S, G)
    S, G = np.array(S, np.float32), np.array(G, np.float32)
    _cache_put(script, T, S, G)
    return T, S, G


def _outlines(data):
    """(glyph set in glyph-id order, em, ids) of an embedded font program: TrueType / OpenType (a damaged cmap or post table is not read: the glyphs are
    addressed by number), bare CFF (Type1C) and Type 1. Raises for anything else."""
    import tempfile
    if data[:4] in (b"\x00\x01\x00\x00", b"true", b"OTTO", b"ttcf"):
        from fontTools.ttLib import TTLibError, newTable
        f = TTFont(io.BytesIO(data), lazy=True)
        n = f["maxp"].numGlyphs
        f.setGlyphOrder([f"glyph{i:05d}" for i in range(n)])
        try:
            f["hmtx"]
        except (TTLibError, KeyError, struct.error):                   # a cut-short hmtx (Word subsets): the outlines need no advances
            f["hmtx"] = newTable("hmtx")
            f["hmtx"].metrics = {g: (0, 0) for g in f.getGlyphOrder()}
        return f.getGlyphSet(), f["head"].unitsPerEm, f.getGlyphOrder()
    if data[:2] == b"\x01\x00":
        from fontTools.cffLib import CFFFontSet
        cff = CFFFontSet()
        cff.decompile(io.BytesIO(data), None)
        top = cff[cff.fontNames[0]]
        cs = top.CharStrings
        names = list(top.charset) if getattr(top, "charset", None) else list(cs.keys())
        return {n: cs[n] for n in names if n in cs}, round(1 / top.FontMatrix[0]) if getattr(top, "FontMatrix", None) else 1000, names
    if data[:2] == b"%!" or data[:1] == b"\x80":
        from fontTools.t1Lib import T1Font
        import re
        from fontTools.misc import eexec, psCharStrings, psLib
        for data in (data.rstrip(b"\x00"), re.sub(rb"(?:0{64}\s*)+cleartomark.*$", b"", data, flags=re.S)):    # the stream whole, else without its trailer
            try:
                with tempfile.NamedTemporaryFile(suffix=".pfa") as t:
                    t.write(data)
                    t.flush()
                    t1 = T1Font(t.name)
                    t1.font = psLib.suckfont(t1.data, t1.encoding)         # T1Font.parse, with no Subrs (optional in Type 1) taken as none
                    subrs = t1.font["Private"].setdefault("Subrs", [])
                    n = t1.font["Private"].get("lenIV", 4)
                    for name, cs in t1.font["CharStrings"].items():
                        t1.font["CharStrings"][name] = psCharStrings.T1CharString(eexec.decrypt(cs, 4330)[0][n:], subrs=subrs)
                    for i in range(len(subrs)):
                        subrs[i] = psCharStrings.T1CharString(eexec.decrypt(subrs[i], 4330)[0][n:], subrs=subrs)
                    del t1.data
                break
            except Exception:
                continue
        else:
            raise ValueError("Type 1 program unreadable")
        return t1.getGlyphSet(), round(1 / t1.font["FontMatrix"][0]), list(t1.font["CharStrings"].keys())
    raise ValueError("not a font program")


_told = set()


def font_features(data, only=None):
    """{glyph id: (shape, geometry)} of the glyphs with ink of a font program (only the glyph ids in `only` when given: the ones the text uses);
    {} for a font program that cannot be read."""
    try:
        gs, upem, names = _outlines(data)
        feats = {g: feature(gs, name, upem) for g, name in enumerate(names) if only is None or g in only}
    except Exception as e:
        if repr(e)[:100] not in _told:                                            # once per reason in a process (the same fonts recur on every page)
            _told.add(repr(e)[:100])
            print("font program unreadable:", repr(e)[:100], flush=True)
        return {}
    return {g: r for g, r in feats.items() if r is not None}


# ------------------------------------------------------------------------------------------------ learned glyph shapes
# Legacy fonts of one family draw identical outlines in every document, so a glyph shape decided once (by the lexicon fit of a document, then confirmed
# by the votes of others) is a better reference than any Noto glyph: the learned set is looked up first, and a glyph it decides needs no text at all.
SAME = 0.9                                      # distance under which two outlines are the same glyph (576-d L2 plus the weighted geometry)
DECIDE_VOTES, DECIDE_SHARE = 3, 0.7             # a learned shape decides alone with this many votes and share
_learned = {}
LEARNED_ITEMS = {}                              # decided glyph shapes collected while learning: script -> [(shape, geometry, text)]


def learned(script):
    """(texts, shape matrix, geometry matrix, votes) of the decided legacy glyph shapes of the script, as {text: count} per entry."""
    if script not in _learned:
        f = os.path.join(CACHE, f"learned_{script}.npz")
        if os.path.exists(f):
            z = np.load(f, allow_pickle=False)
            _learned[script] = [str(z["T"]).split("\x00"), z["S"].astype(np.float32), z["G"].astype(np.float32), z["N"].astype(np.int32)]
        else:
            _learned[script] = [[], np.zeros((0, GRID * GRID), np.float32), np.zeros((0, 3), np.float32), np.zeros(0, np.int32)]
    return _learned[script]


def _dist(V, W, S, G):
    SS = (S ** 2).sum(1)
    return np.sqrt(np.maximum((V ** 2).sum(1)[:, None] + SS[None] - 2 * V @ S.T, 0)) + (np.abs(W[:, None, :] - G[None]) * GEO).sum(-1)


def add_learned(script, items):
    """items: [(shape, geometry, text)] of decided glyphs -> merged into the learned set (same shape + same text: one more vote)."""
    T, S, G, N = learned(script)
    T = list(T)
    for v, w, t in items:
        if len(T):
            d = _dist(v[None], w[None], S, G)[0]
            same = [i for i in np.nonzero(d < SAME)[0] if T[i] == t]
            if same:
                N[same[0]] += 1
                continue
        T.append(t)
        S = np.vstack([S, v[None]])
        G = np.vstack([G, w[None]])
        N = np.append(N, 1).astype(np.int32)
    _learned[script] = [T, S, G, N]


def save_learned():
    os.makedirs(CACHE, exist_ok=True)
    for script, (T, S, G, N) in _learned.items():
        if len(T):
            np.savez_compressed(os.path.join(CACHE, f"learned_{script}.npz"), T=np.array("\x00".join(T)), S=S.astype(np.float16), G=G.astype(np.float16), N=N)


def decide(feats, script):
    """({glyph id: text}, undecided ids): glyphs whose shape is in the learned set with enough votes and a clear winner among the texts near it."""
    T, S, G, N = learned(script)
    out, rest = {}, []
    gids = list(feats)
    if not len(T):
        return out, gids
    for a in range(0, len(gids), 64):
        blk = gids[a:a + 64]
        d = _dist(np.array([feats[g][0] for g in blk], np.float32), np.array([feats[g][1] for g in blk], np.float32), S, G)
        for row, g in enumerate(blk):
            near = np.nonzero(d[row] < SAME)[0]
            votes = collections.Counter()
            for i in near:
                votes[T[i]] += int(N[i])
            tot = sum(votes.values())
            if tot >= DECIDE_VOTES:
                t, n = votes.most_common(1)[0]
                if n >= DECIDE_SHARE * tot:
                    out[g] = t
                    continue
            rest.append(g)
    return out, rest


def candidates(feats, script, k=8):
    """{glyph id: [(text, distance)]}: the k nearest distinct texts of each glyph among the reference of the script (vectorised, in blocks of glyphs)."""
    T, S, G = reference(script)
    gids = list(feats)
    out = {}
    for a in range(0, len(gids), 48):
        blk = gids[a:a + 48]
        d = _dist(np.array([feats[g][0] for g in blk], np.float32), np.array([feats[g][1] for g in blk], np.float32), S, G)
        m = min(4 * k, d.shape[1] - 1)
        for row, g in enumerate(blk):
            part = np.argpartition(d[row], m)[:m + 1]
            seen, cand = set(), []
            for i in part[np.argsort(d[row][part])]:
                if T[i] not in seen:
                    seen.add(T[i])
                    cand.append((T[i], float(d[row][i])))
                    if len(cand) == k:
                        break
            out[g] = cand + [(t, 99.0) for t in EXTRA.get(script, ()) if t not in seen]
    return out


DEVA_JOIN = [("्ा", ""), ("ाे", "ो"), ("ाै", "ौ"), ("ाॅ", "ॉ"), ("अा", "आ"),
             ("अो", "ओ"), ("अौ", "औ"), ("अॉ", "ऑ")]
INDIC = {"Devanagari", "Bengali", "Gurmukhi", "Gujarati", "Oriya", "Tamil", "Telugu", "Kannada", "Malayalam", "Sinhala", "Khmer", "Myanmar"}


@lru_cache(maxsize=1 << 17)
def logical(text, script):
    """Indic rules on the text of glyphs set in order. Devanagari: a half form + the stem sign is the full letter, 'aa' + 'e' is 'o' (the pieces legacy
    fonts build the vowel signs from), a + 'aa' is the vowel letter 'aa'; then prefix vowel signs and the reph into logical order (all Indic scripts)."""
    if script == "Devanagari":
        for a, b in DEVA_JOIN:
            text = text.replace(a, b)
    return indic.to_logical(text) if script in INDIC else text


RTL = {"Arabic", "Hebrew"}


def drawn_order(text, xs, script):
    """The text of glyphs taken in drawing order: for a right-to-left script set left to right (visual order, as most producers draw it) the
    logical order by the Unicode bidirectional algorithm (UAX #9: numbers and Latin inside keep their order)."""
    if script in RTL and len(xs) > 1 and xs[-1] > xs[0]:
        from bidi import get_display
        return get_display(text, base_dir="R")
    return text


PUNCT_STRIP = "\"'“”‘’«»()[]{}<>.,;:!?।॥-–—…*/|%0123456789"
UNSPACED = {"th", "lo", "km", "my"}                      # no spaces between words (and not in wordfreq): the lexicon counts the character pairs of the real paragraphs


@lru_cache(maxsize=48)
def lexicon(lang):
    """(function of a text -> how much of it reads as the language): the sum of the Zipf frequencies of its words (wordfreq where it has the language),
    else the number of its words in the real paragraphs of the language (handwriting_detection langds/paras); Han / kana / Hangul and the unspaced scripts
    count the adjacent character pairs found in those paragraphs (no word boundaries there)."""
    import json
    pool = []
    f = f"{PARAS}/{lang}.json"
    if os.path.exists(f):
        for v in json.load(open(f)).values():
            pool += [t for t in v if isinstance(t, str)]
    if lang in ("zh", "ja", "ko") or lang in UNSPACED:
        n = 3 if lang in UNSPACED else 2                                          # Thai, Lao, Khmer, Burmese: a pair of signs is in too many words, triples tell them apart
        grams = {t[i:i + n] for t in pool for i in range(len(t) - n + 1)}
        return lambda text: sum(text[i:i + n] in grams for i in range(len(text) - n + 1))
    try:
        from wordfreq import available_languages, zipf_frequency
        if lang in available_languages():
            zipf = lru_cache(maxsize=1 << 18)(lambda w: zipf_frequency(w, lang))    # the fit asks the same words again and again
            return lambda text: sum(zipf(w) for w in text.split() if len(w) > 1)
    except ImportError:
        pass
    word = lambda w: w.strip(PUNCT_STRIP)                                         # whole words split at spaces (\w would split at the vowel signs, which are marks)
    for o in sorted(l for l, sc in LANG_SCRIPT.items() if sc == LANG_SCRIPT.get(lang) and l != lang):      # the paragraphs of the other languages of the script too (Nepali shares most words with Hindi)
        if os.path.exists(f"{PARAS}/{o}.json"):
            for v in json.load(open(f"{PARAS}/{o}.json")).values():
                pool += [t for t in v if isinstance(t, str)]
    words = {word(w) for t in pool for w in t.split()}
    return lambda text: sum(3.0 for w in map(word, text.split()) if len(w) > 1 and w in words)


def fit(runs, cand, script, lang, base=None, passes=2, top=150):
    """The table {glyph id: text} of a legacy font from its candidate texts and the glyph-id runs of the document's own text: start with the nearest shape,
    then, glyph by glyph (the `top` most frequent), take the candidate under which more of the text reads as words of the language (coordinate ascent
    on the lexicon score)."""
    table = {**{g: c[0][0] for g, c in cand.items()}, **(base or {})}              # glyphs decided from learned shapes are fixed
    score = lexicon(lang)
    read = lambda run: logical("".join(table.get(g, " ") for g in run), script)
    got = [score(read(r)) for r in runs]                                          # the score is a sum over words: only the runs that hold a glyph are read again
    where = collections.defaultdict(list)
    for i, run in enumerate(runs):
        for g in set(run):
            where[g].append(i)
    best = sum(got)
    freq = collections.Counter(g for run in runs for g in run)
    for _ in range(passes):
        for g, _n in freq.most_common(top):
            if g not in cand or len(cand[g]) < 2:
                continue
            keep, keep_got = table[g], [got[i] for i in where[g]]
            for t, _d in cand[g]:
                if t == keep:
                    continue
                table[g] = t
                new = [score(read(runs[i])) for i in where[g]]
                if sum(new) > sum(keep_got):
                    best += sum(new) - sum(keep_got)
                    keep, keep_got = t, new
            table[g] = keep
            for i, v in zip(where[g], keep_got):
                got[i] = v
    return table, best


# ---------------------------------------------------------------------------------------------------------- pages
LANG = contextvars.ContextVar("legacy_lang", default=None)         # the language of the document being read (set by language())
FIT_PAGES = 3                                                         # pages of the document whose text the table of a font is fitted on
NATIVE = 0.3                                                          # a font whose characters are this share in the script's ranges is real Unicode text


class language:
    """with legacy.language("hi"): ... the pages read inside get the text of their legacy-font runs by shape."""
    def __init__(self, lang):
        self.lang = lang

    def __enter__(self):
        self.token = LANG.set(self.lang)

    def __exit__(self, *a):
        LANG.reset(self.token)


@lru_cache(maxsize=None)
def _kind(c, script):
    """1 a letter or mark of the script, 0 one of another script, None no letter (dot leaders, dashes, digits and spaces say nothing)."""
    if not (c.isalpha() or unicodedata.category(c)[0] == "M"):
        return None
    if script.startswith("CJK-"):
        return int(any(a <= ord(c) <= b for a, b in ranges_of(script)))
    sc = script_of(c)                                                  # the Unicode Script property: presentation forms and extension blocks are the script's too
    return None if sc in ("Inherited", "Common") else int(sc == script)


def native_share(chars, script):
    """Share of the letters and marks that already are of the script (Unicode Script property)."""
    k = [v for c in chars if (v := _kind(c, script)) is not None]
    return sum(k) / max(len(k), 1)


def nearest_mean(feats, script):
    """Mean distance of the glyphs to their nearest glyph of the script's reference."""
    T, S, G = reference(script)
    gids = list(feats)
    tot = 0.0
    for a in range(0, len(gids), 64):
        blk = gids[a:a + 64]
        tot += _dist(np.array([feats[g][0] for g in blk], np.float32), np.array([feats[g][1] for g in blk], np.float32), S, G).min(1).sum()
    return tot / max(len(gids), 1)


DRAWN = 0.95                                                          # drawn_scripts: a script's glyph distance under this share of Latin's


def drawn_scripts(page, fonts=2):
    """The scripts the most used fonts of the page draw (the script reference nearest their glyph shapes), Latin left out: what a page shows whatever its
    language label says."""
    gids, used = collections.defaultdict(set), collections.Counter()
    for tr in page.get_texttrace():
        for u, g, _, _ in tr["chars"]:
            if g > 0 and not chr(u).isspace():
                gids[tr["font"]].add(g)
                used[tr["font"]] += 1
    out = []
    for name, _ in used.most_common(fonts):
        xs = [x for x, _, _, base, *_ in page.get_fonts() if base.split("+")[-1][:24] == name and x > 0]
        feats = max((font_features(page.parent.extract_font(x)[3], gids[name]) for x in xs), key=len, default={})
        if feats:
            near = {sc: nearest_mean(feats, sc) for sc in SPECS}
            script = min(near, key=near.get)
            if near[script] < DRAWN * near["Latin"] and script not in out:   # clearly nearer than Latin (a Latin font is about as near Cyrillic or Greek)
                out.append(script)
    return out


def is_legacy(feats, script):
    """The font draws the script's letters, not Latin ones: its glyphs sit nearer to the script's glyphs than to Latin letters."""
    return nearest_mean(feats, script) < nearest_mean(feats, "Latin")


STORE_FILE = os.path.join(CACHE, "families.json")
_store = {}                                                          # script -> family -> code char -> {text: count}: what the fitted fonts of many documents say


def family(name):
    """The font family of a font name: subset tag, style words and the like dropped (KOGEBG+KrutiDev010Bold -> krutidev010)."""
    import re
    n = name.split("+")[-1].lower()
    n = re.sub(r"[^a-z0-9]", "", n)
    return re.sub(r"(bold|italic|oblique|regular|normal|narrow|wide|plain|light|medium|black|mt|ps)+$", "", n) or n


def store(script):
    if script not in _store:
        try:
            import json
            _store[script] = json.load(open(STORE_FILE)).get(script, {})
        except (OSError, ValueError):
            _store[script] = {}
    return _store[script]


def vote(script, fam, pairs):
    """Count (code char, text) pairs of a fitted font into the family's votes."""
    d = store(script).setdefault(fam, {})
    for c, t in pairs:
        d.setdefault(c, {})
        d[c][t] = d[c].get(t, 0) + 1


def consensus(script, fam, min_votes=3, share=0.6):
    """{code char: text} the fits of the family agree on (at least min_votes of them, the winner with `share` of the votes)."""
    out = {}
    for c, v in store(script).get(fam, {}).items():
        tot = sum(v.values())
        t, n = max(v.items(), key=lambda x: x[1])
        if tot >= min_votes and n >= share * tot:
            out[c] = t
    return out


def save_store(extra=None):
    """Merge the votes of this process (and `extra` votes of other processes) into the stored file."""
    import json
    cur = {}
    try:
        cur = json.load(open(STORE_FILE))
    except (OSError, ValueError):
        pass
    for src in (_store, extra or {}):
        for script, fams in src.items():
            tgt = cur.setdefault(script, {})
            for fam, codes in fams.items():
                for c, v in codes.items():
                    for t, n in v.items():
                        d = tgt.setdefault(fam, {}).setdefault(c, {})
                        d[t] = d.get(t, 0) + n if src is not _store else max(d.get(t, 0), n)
    os.makedirs(CACHE, exist_ok=True)
    json.dump(cur, open(STORE_FILE, "w"), ensure_ascii=False)


class Table:
    """The text of the glyphs of one font: by glyph id (own fit) and by code character (the family's votes, for fonts without a readable program)."""
    only = False                                                       # the glyphs of gid only: the rest of the font's text map stands

    def __init__(self, gid, code):
        self.gid, self.code = gid, code

    def text(self, g, u):
        return self.gid.get(g) or self.code.get(chr(u)) or " "


MAC_PUNCT = {"\xd0": "\u2013", "\xd1": "\u2014", "\xd2": "\u201c", "\xd3": "\u201d", "\xd4": "\u2018", "\xd5": "\u2019"}
CODEPAGE = {"Cyrillic": "cp1251", "Greek": "cp1253", "Hebrew": "cp1255", "Arabic": "cp1256", "Thai": "cp874"}


def mixed(word, script):
    """A word that holds letters of the script: a Latin letter in it is a code a table left, not an English word."""
    return any(unicodedata.category(c)[0] in "LM" and _kind(c, script) > 0 for c in word)


def stray(c, script):
    """A character of a wrong text map inside a Unicode font of the script: private use, or a letter / mark of another script that is not ASCII.
    U+FFFD is no wrong map: the producer says the text is unknown (refont drops it where the glyphs' text is there)."""
    if c.isascii() or c.isspace() or c == "\ufffd":
        return False
    cat = unicodedata.category(c)
    return cat in ("Co", "Cn", "Cs") or (cat[0] in "LM" and _kind(c, script) == 0)


def font_tables(doc, page, script, lang, learn=False):
    """{font name: Table | None} of the fonts the page uses; None for a font that is not a legacy one (real Unicode, a Latin font) or that cannot be read.
    Fitted once per document and font (again when a page brings glyphs the fit has not seen); the family's votes replace a glyph's text where the fits of
    many documents agree on another. A font whose text is Unicode of the script keeps its own text map."""
    cache = _doc_cache(doc)[1]["tables"]
    before = {k: v[1] for k, v in cache.items()}
    traces = typed_traces(page)
    used = {tr["font"] for tr in traces}
    xrefs = collections.defaultdict(list)
    for xref, ext, kind, base, *_ in page.get_fonts():
        name = base if base in used else base.split("+")[-1]          # 'CIDFont+F4' is a name, 'ABCDEF+Kaumudi' a subset of one
        if name in used:
            xrefs[name].append(xref)
    for name, xs in xrefs.items():                                     # two subsets of one font on a page share its name in the text trace: the program that has the glyphs
        chars, bad = {}, set()
        for tr in traces:
            if tr["font"] == name:
                for u, g, o, _ in tr["chars"]:
                    if g > 0:
                        c = chr(u) if u != 0xFFFD else shown(page).get((name, round(o[0], 1), round(o[1], 1)), "\ufffd")
                        chars.setdefault(g, c)
                        if misread(c, script) or c == "\ufffd":       # a broken code, or no text in the map nor in the text extraction
                            bad.add(g)
        if chars and native_share(list(chars.values()), script) >= NATIVE:     # its text is Unicode of the script: the font's own text map stands,
            if bad - cache.get((name, script, MISREAD_RULE), (None, set()))[1]:      # but for the glyphs it maps to broken codes (a conjunct of a subset): read by shape
                cache[(name, script, MISREAD_RULE)] = (_misread_table(doc, page, xs, name, chars, bad, script, lang), bad | cache.get((name, script, MISREAD_RULE), (None, set()))[1])
            continue
        seen = cache.get((name, script), (None, None))[1]
        if seen is not None and set(chars) <= seen:
            continue
        cache[(name, script)] = (None, set(chars) | (seen or set()))
        if not chars:
            continue
        fam = family(name)
        try:
            gids = set(chars)
            feats = max(({} if x <= 0 else font_features(doc.extract_font(x)[3], gids) for x in xs), key=len)
            cons = consensus(script, fam)
            if not feats:                                              # no program (system font, Type 1): the family's votes by code
                if cons:
                    cache[(name, script)] = (Table({}, cons), gids | (seen or set()))
                else:
                    cache.pop((name, script))                          # another subset of the name may have the program
                continue
            if not is_legacy(feats, script):
                continue
            gid, rest = decide(feats, script)                          # shapes already decided in other documents: no text needed
            if rest:
                cs = candidates({g: feats[g] for g in rest}, script)
                runs = []
                for pn in sorted(set(range(min(len(doc), FIT_PAGES))) | {page.number}):      # the page being read too: its glyphs may not be on the first pages
                    runs += [[g for _, g, _, _ in tr["chars"] if g > 0] for tr in doc[pn].get_texttrace() if tr["font"] == name]
                gid = fit(runs, cs, script, lang, base=gid)[0]
                if learn:
                    cnt = collections.Counter(g for run in runs for g in run)
                    LEARNED_ITEMS.setdefault(script, []).extend((feats[g][0], feats[g][1], gid[g]) for g in rest if cnt[g] >= 5 and g in gid)
            code = {}
            for u, g in {(ord(c), g) for g, c in chars.items()}:
                if g in gid:
                    code[chr(u)] = gid[g]
            if learn:
                vote(script, fam, code.items())
            cache[(name, script)] = (Table(gid, {**cons, **code}), gids | (seen or set()))
        except Exception:
            cache[(name, script)] = (None, gids | (seen or set()))
    if not learn and {k: v[1] for k, v in cache.items()} != before:
        _doc_save(doc)
    return {n: cache.get((n, script), (None,))[0] or cache.get((n, script, MISREAD_RULE), (None,))[0] for n in used}


MISREAD_RULE = "misread-shapes-8"                                   # _misread_table's rule: a cached table of another rule is made again


def shown(page):
    """{(font, x, y): character} of the page as the text extraction gives it (where the glyph trace has U+FFFD, it may have the text, or the code typed)."""
    d = page.parent.__dict__.setdefault("_shown", {})
    if page.number not in d:
        d.clear()
        d[page.number] = {(sp["font"], round(ch["origin"][0], 1), round(ch["origin"][1], 1)): ch["c"]
                          for b in page.get_text("rawdict")["blocks"] for l in b.get("lines", []) for sp in l["spans"] for ch in sp["chars"]}
    return d[page.number]


def misread(c, script):
    """A code a text map gives a glyph that is no text: a control, unassigned, private-use or surrogate code (a letter of any script is valid Unicode:
    the map stands)."""
    return unicodedata.category(c) in ("Cc", "Cn", "Co", "Cs") and not c.isspace()


def _misread_table(doc, page, xs, name, chars, bad, script, lang):
    """Table (only: these glyphs) of the misread glyphs of a Unicode font, read by shape: the nearest glyphs of the script, the one under which the text
    reads best, with the text map of the other glyphs fixed; None when the program has no outlines for them."""
    feats = max(({} if x <= 0 else font_features(doc.extract_font(x)[3], bad) for x in xs), key=len, default={})
    if not feats:
        return None
    gid, rest = decide(feats, script)
    if rest:
        runs = [[g for _, g, _, _ in tr["chars"] if g > 0] for pn in sorted(set(range(min(len(doc), FIT_PAGES))) | {page.number})
                for tr in doc[pn].get_texttrace() if tr["font"] == name]
        base = {**{g: c for g, c in chars.items() if g not in bad}, **gid}
        gid = fit(runs, candidates({g: feats[g] for g in rest}, script), script, lang, base=base)[0]
    t = Table({g: gid[g] for g in feats if g in gid}, {})
    t.only = True
    return t


def word_share(text, lang):
    """Share of the words of a text that the language's word list has (wordfreq, with its CJK extra for zh / ja / ko; 1.0 for a language it lacks:
    no evidence against)."""
    from wordfreq import available_languages, zipf_frequency
    if lang not in available_languages():
        return 1.0
    ws = [w.strip(PUNCT_STRIP) for w in text.split()]
    ws = [w for w in ws if len(w) > 1 and any(c.isalpha() for c in w)]
    return sum(zipf_frequency(w, lang) > 0 for w in ws) / max(len(ws), 1)


@lru_cache(maxsize=1)
def _glotlid():
    """GlotLID (fastText, 1600+ languages) through corpus_collecting's numpy reader (ftnp)."""
    import sys
    from ..ingest.paths import CC_ROOT, LANG_MODELS
    sys.path.insert(0, str(CC_ROOT))
    from corpus import ftnp
    return ftnp.Model(LANG_MODELS / "glotlid.bin")


def glot(text, script=None):
    """GlotLID's probability that the text is a language (written in `script` when given): how surely a decode reads as real text (character n-grams,
    the charset-detection practice; no word list)."""
    model = _glotlid()                                                 # puts corpus_collecting on the path
    from corpus.langstack import iso
    labels, probs = model.predict(" ".join(text.split())[:4000], k=5)
    return max((p for lb, p in zip(labels, probs) if script is None or LANG_SCRIPT.get(iso(lb)) == script), default=0.0)


SCREEN = 600                                                          # characters a candidate table must turn into the script before it is scored
COVER = 0.9                                                           # share of the sample's letters a labelled table must know (it leaves the others as typed)


TRUST = 2 / 3                                                         # share of codes the glyph-shape fit must read as some candidate table does to count


def _safe(f):
    """f, if it can run on one code (a table that cannot read the codes is no candidate)."""
    try:
        f("a")
        return True
    except Exception:
        return False


def table_for(doc, font, traces, script, lang, shapes=None):
    """The table a legacy font of the document is decoded with, detected as charset detectors do: the font's text on the page decoded by the table its
    name routes to and by every published and labelled table, each scored by the language model (lexicon); the best one (a font named for one encoding
    set in another: ML-Revathi in the ML-TT codes; a font no name routes); None when no table gives the script. Once per document and font."""
    from . import legacy_tables as LT
    cache, sig = _doc_cache(doc)[1]["table_for"], tables_sig(script)
    if not (isinstance(cache.get(font), tuple) and cache[font][1] == sig):   # decided among the script's present tables (an older entry: decide again)
        sample = " ".join(LT.raw((u for u, g, _, _ in tr["chars"] if g > 0), font) for tr in traces if tr["font"] == font)[:4000]
        from .legacy_fit import char_lm
        lm = char_lm(lang)
        codes = sorted({c for c in sample if not c.isspace() and c in (shapes or {})})
        against = lambda f: -sum(f(c) not in (c, shapes[c]) for c in codes)   # codes the table maps otherwise than the font's glyph shapes read them
        agree = lambda f: sum(f(c) == shapes[c] for c in codes)
        score = lambda t, f: (round(glot(t, script), 2), against(f), lm.mean(t))   # GlotLID saturates (tables a few codes apart both read 1.0): then the
        # table the glyph shapes contradict least (a code it leaves as typed the shapes still fill), then the language's character trigram
        typed = [c for c in sample if c.isalpha()]
        lab = LT._labelled()                                           # a labelled table leaves the codes it lacks as typed: it must know most letters of the sample
        cands = {**({"route": partial(LT.decode, font)} if LT.known(font) else {}), **{k: f for k, f in LT.decoders().items() if LT.writes(k, script)},
                 **{f"labelled:{f}": partial(LT.run, "labelled", f) for f, m in lab.items() if sum(c in m for c in typed) >= COVER * len(typed)}}
        out = {}
        trusted = bool(codes) and max((agree(f) for f in cands.values() if _safe(f)), default=0) >= TRUST * len(codes)   # the fit reads this font: it
        if not trusted:                                                # matches some table on most codes (a fit that matches none says nothing)
            against = lambda f: 0
        for k, f in cands.items():                                     # the script test on the first SCREEN characters, the language score on the whole sample
            try:
                ok = (t := f(sample[:SCREEN])) and native_share(t, script) >= 0.9 and (t := f(sample)) and native_share(t, script) >= 0.9
            except Exception:                                          # a table that cannot read these codes is no candidate
                continue
            if ok:
                out[k] = score(t, f)
        if "route" in cands and not any(c.isalpha() for c in sample):      # digits and signs only: nothing to tell the tables apart by
            out = {"route": (0, 0, 0)}
        if "route" not in cands and out:                                   # a font no name routes: decoded only when that reads better than as typed
            best = max(out, key=out.get)
            if out[best][0] <= glot(sample):                             # GlotLID reads the typed text as some language at least as surely
                out = {}
        best = max(out, key=out.get, default=None)
        cache[font] = (best, sig)
        _doc_save(doc)
        if best and best != "route":
            print("table detected:", font, "->", best, flush=True)
    best = cache[font][0]
    return None if best is None else partial(LT.decode, font) if best == "route" else \
        partial(LT.run, "labelled", best.split(":", 1)[1]) if best.startswith("labelled:") else LT.decoders()[best]


CODE_NAME = re.compile(r"[A-Za-z]{1,2}([0-9A-Fa-f]{2,3})")                # a glyph named by the code typed (G2B, G43, MT133)


def _no_text(u):
    return u == 0xFFFD or (u < 32 and chr(u) not in "\t\n\r") or 0x7F <= u < 0xA0


def _code_names(doc):
    """{trace font name: {glyph id: code}} of the font programs of the document whose glyph names all carry the code typed; the codes are hex where a
    name of the document has a hex letter (the Windows PostScript driver: G2B), else decimal (G43, MT133). Read once per document."""
    d = doc.__dict__
    if "_code_names" not in d:
        names, hexa = {}, False
        for x, ext, kind, base, *_ in {f for pn in range(len(doc)) for f in doc.get_page_fonts(pn)}:
            if x <= 0 or ext in ("ttf", "otf"):
                continue
            try:
                ids = _outlines(doc.extract_font(x)[3])[2]
            except Exception:
                continue
            m = [CODE_NAME.fullmatch(n) for n in ids[1:]]
            if m and all(m):
                names[base.split("+")[-1][:24]] = {g: k.group(1) for g, k in enumerate(m, 1)}
                hexa |= any(not c.isdigit() for k in m for c in k.group(1))
        d["_code_names"] = {f: {g: int(c, 16 if hexa else 10) for g, c in v.items()} for f, v in names.items()}
    return d["_code_names"]


def typed_traces(page):
    """page.get_texttrace() with the code typed where the text map gives a glyph no text (a control code, U+FFFD) and its glyph name has it (_code_names):
    the byte as a character, as an 8-bit text map gives it (the code page or legacy table reads it). Each trace's 'typed': the indexes put back."""
    tr = page.get_texttrace()
    if not any(g > 0 and _no_text(u) for t in tr for u, g, _, _ in t["chars"]):
        return tr
    names = _code_names(page.parent)
    for t in tr:
        nm, chars, typed = names.get(t["font"], {}), [], set()
        for i, (u, g, o, b) in enumerate(t["chars"]):
            code = nm.get(g) if g > 0 and _no_text(u) else None
            if code is not None and 32 <= code < 256 and code != 0x7F:
                u = code
                typed.add(i)
            chars.append((u, g, o, b))
        t["chars"], t["typed"] = tuple(chars), typed
    return tr


class Fixed(dict):
    """fixed()'s result; fitted: the keys read by the glyph-shape fit (a guess: a font program's own glyph text goes before it)."""
    def __init__(self):
        super().__init__()
        self.fitted, self.guess = set(), set()                         # guess: shape reads of glyphs the text map gives a broken code


LATIN_RULE = "no-program-is-latin"                                    # latin_drawn's rule: a cached decision of another rule is made again


def latin_drawn(page, font, script):
    """True when the font's program draws Latin letters, not the script's (is_legacy on its glyph shapes): a Latin font such as ArialMT typing a URL on
    a Punjabi page gets no legacy table; so does a font whose program cannot be read (a system Times typing English: no evidence of a legacy font, and a
font a table name routes is decoded by that name). Once per document and font."""
    cache = _doc_cache(page.parent)[1].setdefault("latin", {})
    if not (isinstance(cache.get(font), tuple) and cache[font][1] == LATIN_RULE):   # decided by this rule (an older entry: decide again)
        gids = {g for tr in page.get_texttrace() if tr["font"] == font for _, g, _, _ in tr["chars"] if g > 0}
        xs = [x for x, _, _, base, *_ in page.get_fonts() if base == font or base.split("+")[-1] == font]
        feats = max(({} if x <= 0 else font_features(page.parent.extract_font(x)[3], gids) for x in xs), key=len, default={})
        cache[font] = (not feats or not is_legacy(feats, script), LATIN_RULE)   # no program to show the script's shapes: no evidence of a legacy font
        _doc_save(page.parent)
    return cache[font][0]


def fixed(page, lang, learn=False):  # tables first (legacy_tables), then the shape fit
    """{(font, x, y): text} of the legacy-font runs of a page in the form refont's extract_segments takes: the text of a run on its first glyph, '' on the
    others (the rules join glyphs: half form + stem, prefix signs); {} for a language without a script here or a page without legacy fonts."""
    from . import legacy_tables as LT
    script = LANG_SCRIPT.get(lang)
    if script is None:
        return Fixed()
    out, traces = Fixed(), typed_traces(page)
    shape = font_tables(page.parent, page, script, lang, learn)
    typed = collections.defaultdict(list)
    for tr in traces:
        typed[tr["font"]] += [chr(u) for u, g, _, _ in tr["chars"] if g > 0]
    legacy_font = {n: native_share(c, script) < NATIVE or (LT.route(n) or ("",))[0] in LT.NATIVE_CODED for n, c in typed.items()}     # per font over the page: a run of spaces or punctuation is no sign of a legacy font; a font typed in the script's own code points (Zawgyi) by its name
    tables = {f: d for f in typed if legacy_font[f] and (LT.known(f) or not latin_drawn(page, f, script)) and (d := table_for(page.parent, f, traces, script, lang, shape[f].code if shape.get(f) else None))}
    for tr in traces:                                              # fonts with a table (by name, or found by the text it gives): exact
        if not (tr["chars"] and tr["font"] in tables):
            continue
        lines = [[]]
        for ch in tr["chars"]:                                     # a run set over several lines (a paragraph in one text object): each line decoded on its own first glyph
            if lines[-1] and abs(ch[2][1] - lines[-1][-1][2][1]) > 0.5 * tr["size"]:
                lines.append([])
            lines[-1].append(ch)
        dec = tables[tr["font"]]
        for chars in lines:
            text = dec(LT.raw((u for u, g, _, _ in chars if g > 0), tr["font"]))
            if text is not None and shape.get(tr["font"]):              # codes the published table leaves as typed: read from the glyph shapes
                code = shape[tr["font"]].code
                left = lambda c, w: dec(c) == c and (stray(c, script) or (ord(c) > 127 and not c.isalpha()) or (c.isascii() and c.isalpha() and mixed(w, script)))
                text = "".join("".join(code.get(c, c) if left(c, w) else c for c in w) for w in re.split(r"(\s+)", text))   # codes the table in use leaves
                # as typed (a Latin letter inside a word of the script, not an English word)
            if text is not None:
                text = "".join(MAC_PUNCT.get(c, c) if stray(c, script) else c for c in text)     # the quotes and dashes the table does not know stay at their Mac Roman codes
                text = drawn_order(text, [o[0] for _, _, o, _ in chars], script)
                if script == "Khmer":                                  # the Khmer converter tables map code by code: a coeng ro typed before its consonant
                    text = logical(text, script)                       # (as drawn) goes after it
                for i, (_, _, o, _) in enumerate(chars):
                    out[(tr["font"], round(o[0], 1), round(o[1], 1))] = text if i == 0 else ""
    tabs = {n: t for n, t in shape.items() if t and n not in tables}
    page_cp = CODEPAGE.get(script)
    for tr in traces:                                              # 8-bit code page fonts (Cyrillic, Greek, Hebrew, Arabic, Thai letters at the Latin-1 positions)
        if page_cp and tr["chars"] and tr["font"] not in tables and all(u < 256 for u, g, _, _ in tr["chars"] if g > 0):   # Latin-1 codes only (a real letter is no code)
            t = LT.raw(u for u, g, _, _ in tr["chars"] if g > 0)
            if t.strip() and all(ord(c) < 256 for c in t) and sum(chr(u).isalpha() for u, g, _, _ in tr["chars"]) >= 0.5 * len(t.strip()):      # a run of dot leaders or dashes is no code-page text (the Mac code of '…' is a letter)
                d = t.encode("latin-1").decode(page_cp, "replace")
                L = [c for c in d if c.isalpha()]
                if L and sum(script_of(c) == script for c in L) >= 0.9 * len(L) and sum(c.isascii() for c in t if c.isalpha()) < 0.5 * len(L):
                    for i, (_, _, o, _) in enumerate(tr["chars"]):
                        out[(tr["font"], round(o[0], 1), round(o[1], 1))] = d if i == 0 else ""
    for tr in traces:
        tab = tabs.get(tr["font"])
        if not tab or not tr["chars"] or (tr["font"], round(tr["chars"][0][2][0], 1), round(tr["chars"][0][2][1], 1)) in out:   # decoded exactly (a code page)
            continue
        if tab.only:                                                   # a Unicode font: its misread glyphs only, in logical order as the map is
            for u, g, o, _ in tr["chars"]:
                if g in tab.gid:
                    out[(tr["font"], round(o[0], 1), round(o[1], 1))] = tab.gid[g]
                    out.guess.add((tr["font"], round(o[0], 1), round(o[1], 1)))      # the last resort: only where the text stays broken (refont)
            continue
        text = logical("".join(tab.text(g, u) for u, g, _, _ in tr["chars"] if g > 0), script)
        text = drawn_order(text, [o[0] for _, _, o, _ in tr["chars"]], script)
        for i, (_, _, o, _) in enumerate(tr["chars"]):
            out[(tr["font"], round(o[0], 1), round(o[1], 1))] = text if i == 0 else ""
            out.fitted.add((tr["font"], round(o[0], 1), round(o[1], 1)))
    return out


def _learn_doc(args):
    """(path, lang) -> votes of the fonts of the document (first pages)."""
    import pymupdf
    path, lang = args
    _store.clear()
    LEARNED_ITEMS.clear()
    try:
        doc = pymupdf.open(path)
        for pn in range(min(len(doc), 3)):
            fixed(doc[pn], lang, learn=True)
    except Exception as e:
        print("unreadable:", path, repr(e)[:100], flush=True)
    return {k: v for k, v in _store.items()}, {k: list(v) for k, v in LEARNED_ITEMS.items()}


def learn(docs, workers=6):
    """docs: [(path, lang)]. Fits the legacy fonts of the documents and stores the votes per font family."""
    from concurrent.futures import ProcessPoolExecutor
    total = {}
    with ProcessPoolExecutor(workers) as ex:
        for i, (votes, items) in enumerate(ex.map(_learn_doc, docs, chunksize=2), 1):
            for script, its in items.items():
                add_learned(script, its)
            for script, fams in votes.items():
                for fam, codes in fams.items():
                    for c, v in codes.items():
                        for t, n in v.items():
                            d = total.setdefault(script, {}).setdefault(fam, {}).setdefault(c, {})
                            d[t] = d.get(t, 0) + n
            if i % 20 == 0:
                print(i, "of", len(docs), flush=True)
    save_store(total)
    save_learned()


# ---------------------------------------------------------------------------------------------------------- text drawn as paths
LINE_OVERLAP, CHAR_MARGIN = 0.5, 2.0                                  # pdfminer's LAParams: a glyph joins a line it overlaps by half the smaller height, within 2 widths of its neighbour
PATH_GLYPHS = 50                                                      # a page with this many glyph outlines sets (some of) its text as outlines
PATH_READ = os.path.join(CACHE, "tables", "paths_read.json")          # {"<stem>_<page>": {"glyphs", "texts"}}: the agents' decided readings of the path lines (legacy_evidence)


def path_glyphs(page):
    """Lines of the text a page draws as filled paths (a path a glyph, a word or a line): [[rect]] top to bottom, grouped as pdfminer groups characters
    (LINE_OVERLAP, CHAR_MARGIN). A glyph outline (pdf-inspector's vector-text check): a filled path of more than 4 line / curve segments, off the text
    layer (a bold or outline effect under real text), of text height; [] for a page with fewer than PATH_GLYPHS."""
    import pymupdf
    text = [pymupdf.Rect(w[:4]) for w in page.get_text("words")]
    ds = [d for d in page.get_drawings() if d.get("fill") is not None and d["type"] == "f" and sum(it[0] in "lc" for it in d["items"]) > 4
          and not any(d["rect"].intersects(t) for t in text)]
    if ds:
        h = np.median([d["rect"].height for d in ds])
        ds = [d for d in ds if 0.2 * h <= d["rect"].height <= 3 * h]
    if len(ds) < PATH_GLYPHS:
        return []
    rows = []
    for d in sorted(ds, key=lambda d: ((d["rect"].y0 + d["rect"].y1) / 2, d["rect"].x0)):
        r = d["rect"]
        if rows and min(r.y1, rows[-1][1]) - max(r.y0, rows[-1][0]) > LINE_OVERLAP * min(r.height, rows[-1][1] - rows[-1][0]):
            rows[-1][0], rows[-1][1] = min(rows[-1][0], r.y0), max(rows[-1][1], r.y1)
            rows[-1][2].append(r)
        else:
            rows.append([r.y0, r.y1, [r]])
    out = []
    for _, _, row in rows:
        out.append([])
        for r in sorted(row, key=lambda r: r.x0):
            if out[-1] and r.x0 - out[-1][-1].x1 > CHAR_MARGIN * max(r.width, out[-1][-1].width):
                out.append([])
            out[-1].append(r)
    return out


@lru_cache(maxsize=1)
def path_read():
    import json
    return json.load(open(PATH_READ, encoding="utf-8")) if os.path.exists(PATH_READ) else {}


def mark_no_text(pdf):
    """<pdf>.paths.json: the glyph outlines left on each page of a pdf we made by replacing all its text (a translated form: the outlines the source's
    reading has as no text, a picture) read as no text, so path_lines reads the pdf."""
    import json
    import pymupdf
    with pymupdf.open(pdf) as d:
        got = {str(p.number): dict(glyphs=[len(l) for l in ls], texts=[""] * len(ls)) for p in d if (ls := path_glyphs(p))}
    json.dump(got, open(f"{pdf}.paths.json", "w", encoding="utf-8"))                # {} for a pdf without outlines: marked once all the same


def path_lines(page):
    """rawdict-like lines of the text a page draws as paths, as the agents read it (path_read): a read line's characters on its outlines where they match
    one to one, else over its box evenly; a pdf we made reads from its <pdf>.paths.json (mark_no_text). A page with path text nobody read (or that reads
    other lines now) raises: its text is not known."""
    import json
    import pymupdf
    lines = path_glyphs(page)
    if not lines:
        return []
    key = f"{os.path.splitext(os.path.basename(page.parent.name or ''))[0]}_{page.number}"
    read = path_read().get(key) if page.parent.name else None
    side = f"{page.parent.name}.paths.json"                                    # a page we made (a translated form): its leftover outlines read as no text
    if not read and page.parent.name and os.path.exists(side):
        read = json.load(open(side, encoding="utf-8")).get(str(page.number))
    if not read or read["glyphs"] != [len(l) for l in lines]:
        raise LookupError(f"text drawn as paths not read: {key}")
    out = []
    for line, text in zip(lines, read["texts"]):
        if not text:
            continue
        box = pymupdf.Rect(line[0])
        for r in line:
            box |= r
        w = box.width / len(text)
        boxes = [tuple(r) for r in line] if len(text) == len(line) else [(box.x0 + i * w, box.y0, box.x0 + (i + 1) * w, box.y1) for i in range(len(text))]
        boxes = [(*b[:2], max(b[2], n[0]), b[3]) for b, n in zip(boxes, boxes[1:])] + boxes[-1:]    # each reaches to the next: the reading has the spaces
        chars = [dict(c=c, bbox=b, origin=(b[0], box.y1)) for c, b in zip(text, boxes)]
        span = dict(font="(paths)", size=max(r.height for r in line), flags=0, color=0, ascender=1.0, descender=0.0, alpha=255, origin=(box.x0, box.y1),
                    bbox=tuple(box), chars=chars)
        out.append(dict(bbox=tuple(box), dir=(1.0, 0.0), wmode=0, spans=[span]))
    return out
