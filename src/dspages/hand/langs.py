"""Languages and scripts of the text sources: which script a language is written in, and checks that a passage is real text of it."""

import os
import unicodedata
from collections import Counter
from functools import cache

from hwsynth.scripts import script

EXPECT = {
    "el": "Greek",
    "he": "Hebrew",
    "ar": "Arabic",
    "fa": "Arabic",
    "ur": "Arabic",
    "ps": "Arabic",
    "sd": "Arabic",
    "ckb": "Arabic",
    "ku": "Arabic",
    "hi": "Devanagari",
    "mr": "Devanagari",
    "ne": "Devanagari",
    "sa": "Devanagari",
    "mai": "Devanagari",
    "bn": "Bengali",
    "as": "Bengali",
    "th": "Thai",
    "ka": "Georgian",
    "hy": "Armenian",
    "ta": "Tamil",
    "te": "Telugu",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Gurmukhi",
    "or": "Oriya",
    "si": "Sinhala",
    "km": "Khmer",
    "lo": "Lao",
    "my": "Myanmar",
    "am": "Ethiopic",
    "bo": "Tibetan",
    "zh": "Han",
    "ja": "Han",
    "ko": "Hangul",
    "ru": "Cyrillic",
    "uk": "Cyrillic",
    "bg": "Cyrillic",
    "mk": "Cyrillic",
    "be": "Cyrillic",
    "kk": "Cyrillic",
    "ky": "Cyrillic",
    "tg": "Cyrillic",
    "mn": "Cyrillic",
    "sr": "Cyrillic",
    "dz": "Tibetan",
    "ti": "Ethiopic",
    "ug": "Arabic",
    "yi": "Hebrew",
    "mni": "Meetei_Mayek",
}  # languages not listed are Latin
NO_TEXT_LAYER = {
    "ur",
    "ps",
    "ku",
    "ckb",
    "sd",
}  # their PDF text layers are in visual order or scrambled: no usable text from them
MERGE = {"Hiragana": "Han", "Katakana": "Han"}
SKIP_SCRIPTS = {"Thaana", "Common", "Inherited", "Unknown"}
GROUP = {"Latin": "latin", "Cyrillic": "cyrillic", "Han": "cjk", "Hiragana": "cjk", "Katakana": "cjk", "Hangul": "cjk"}


def group_of(scr):
    """Script group of the language shares: latin, cyrillic, cjk or other."""
    return GROUP.get(scr, "other")


def script_ok(text, lang, share=None):
    """True when most letters of the text are in the script of the language (Latin when the language has no entry); for a language of another
    script the Latin letters do not count (English terms, the English half of a bilingual page): a text of no letter of the script (romanised, or
    legacy codes decoded to Latin) still fails, the Latin gibberish of a partly decoded page is garbled()'s."""
    want = EXPECT.get(lang, "Latin")
    c = Counter(
        s for s in (MERGE.get(script(ch), script(ch)) for ch in text if ch.isalpha()) if want == "Latin" or s != "Latin"
    )
    n = sum(c.values())
    return n > 0 and c[want] >= (share or (0.97 if want == "Latin" else 0.8)) * n


PRINT_HANDS = (
    "round_print",
    "compact_print",
    "print_script",
    "ruled_neat",
    "rounded_neat",
    "bubbly_print",
    "soft_upright",
    "block_capitals",
)
PRINT_ONLY_NOT = {
    "Latin",
    "Greek",
}  # every other script: its letters become scribbles or Latin shapes under cursive and fitted writers (checked by eye)
LEGIBLE_STYLES = ("casual", "formal", "accurate", "professional", "educated")
ILLEGIBLE_TYPES = ("sloppy_fast", "micro", "child", "thready")


def broken(c):
    """A code point a broken font map gives: unpaired surrogate, private use, unassigned, U+FFFD."""
    return unicodedata.category(c) in ("Cs", "Co", "Cn") or c == "\ufffd"


from ..paths import INFO

DECODED = str(
    INFO / "ds100k/shift/pairs/tr"
)  # pages of the corpus read by the restyling reader: <stem>_<lang>_<page>_<tl>.json
A4_MM, MARGIN_MM, BODY_PT = (
    (210.0, 297.0),
    25.4,
    12.0,
)  # ISO 216 A4, 1 inch margins, 12 pt body type: a standard typed page
FORM_SENTENCES, TEXT_SHARE = (
    3,
    0.1,
)  # a form: three sentences of a medium length; a text page: a tenth of a typed A4 page


def _paras(lang):
    """Real prose of the language: the paragraphs of its corpus pages as the restyling reader decoded them (DECODED: legacy fonts and drawn text
    read) and its curated paragraphs (langds paras); headings and list items left out."""
    import glob
    import json

    from ..layout.legacy import PARAS

    got, seen = [], set()
    for f in sorted(glob.glob(os.path.join(DECODED, f"*_{lang}_*_*.json"))):
        page = f.rsplit("_", 1)[0]
        if page not in seen:
            seen.add(page)
            got += [b[1] for b in json.load(open(f, encoding="utf-8"))["blocks"] or [] if b[0] == "p"]
    f = os.path.join(PARAS, f"{lang}.json")
    return got + (
        [t for t in json.load(open(f, encoding="utf-8")).get("p", []) if isinstance(t, str)]
        if os.path.exists(f)
        else []
    )


@cache
def a4_chars(lang):
    """Characters of the language a typed A4 page holds (A4_MM, MARGIN_MM, BODY_PT): the lines of the script's Noto font (its own line height) times the
    characters a line holds (the language's real paragraphs shaped by HarfBuzz: their advance per character)."""
    import uharfbuzz as hb

    from ..layout import legacy as L

    scr = L.LANG_SCRIPT.get(lang, "Latin")
    path, index = (L.CJK_FONT, L.CJK_LANG[lang][0]) if scr.startswith("CJK-") else (f"{L.FONTS}/{L.SPECS[scr][0]}", 0)
    face = hb.Face(open(path, "rb").read(), index)
    font = hb.Font(face)
    upem = face.upem
    ext = font.get_font_extents("ltr")
    line = (ext.ascender - ext.descender + ext.line_gap) / upem * BODY_PT
    text = " ".join(_paras(lang))
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf)
    adv = sum(p.x_advance for p in buf.glyph_positions) / upem * BODY_PT / len(text)  # points per character
    mm = 72 / 25.4
    width, height = (A4_MM[0] - 2 * MARGIN_MM) * mm, (A4_MM[1] - 2 * MARGIN_MM) * mm
    return int(height // line) * width / adv


def _group(lang):
    return "cjk" if group_of(EXPECT.get(lang, "Latin")) == "cjk" else "rest"


@cache
def min_chars(lang, kind):
    """Least characters of a source page of the language: the limit of its script group, CJK or all the rest (Latin, Cyrillic and the other
    scripts come out alike: 200 to 237 characters a form, 301 to 328 a text page; CJK 135 and 175). A form: FORM_SENTENCES sentences of the
    median length of the group's prose (cut after the Unicode sentence terminators); a text page: TEXT_SHARE of the median of its languages' typed A4 pages."""
    import statistics

    import regex

    from ..layout.legacy import LANG_SCRIPT

    langs = [l for l in set(EXPECT) | set(LANG_SCRIPT) if _group(l) == _group(lang) and _paras(l)]
    if kind == "form":
        return FORM_SENTENCES * statistics.median(
            len(x.strip())
            for l in langs
            for t in _paras(l)
            for x in regex.split(r"(?<=\p{STerm})(?!\p{N})\s*", t)
            if regex.search(r"\p{L}", x)
        )
    return TEXT_SHARE * statistics.median(a4_chars(l) for l in langs)
