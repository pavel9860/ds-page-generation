"""Text of real documents for handwriting"""
import random
import unicodedata
from collections import Counter

SCRIPTS = {
    "Arabic": "ar fa ur ps ku ckb sd ug", "Hebrew": "he yi", "Oriya": "or", "Thaana": "dv", "Meetei_Mayek": "mni", "Greek": "el", "Cyrillic": "ru be bg mk kk kz ky mn tg tj uk",
    "Devanagari": "hi mr ne sa mai", "Bengali": "bn as", "Gujarati": "gu", "Gurmukhi": "pa", "Tamil": "ta", "Telugu": "te", "Kannada": "kn",
    "Malayalam": "ml", "Sinhala": "si", "Khmer": "km", "Lao": "lo", "Myanmar": "my", "Thai": "th", "Tibetan": "bo dz",
    "Ethiopic": "am ti", "Armenian": "hy", "Georgian": "ka", "Hangul": "ko", "Han": "zh", "Hiragana": "ja", "Katakana": "ja",
}
ALSO = {"ku": {"Latin"}, "mni": {"Bengali"}, "sr": {"Cyrillic", "Latin"}, "uz": {"Latin", "Cyrillic"}, "ja": {"Han", "Hiragana", "Katakana"}}


def scripts_of(lang):
    """The scripts a language is written in (Latin unless listed)."""
    own = {s for s, langs in SCRIPTS.items() if lang in langs.split()} | ALSO.get(lang, set())
    return own or {"Latin"}


def _script(c):
    return unicodedata.name(c, "?").split()[0]


_VOWELS = {"A", "AA", "I", "II", "U", "UU", "E", "EE", "AI", "O", "OO", "AU"}


def _invalid(word):
    """Orthographically impossible spellings"""
    prev = None
    scripts = set()
    for c in word:
        cat = unicodedata.category(c)
        if cat in ("Cn", "Co", "Cs"):
            return True
        name = unicodedata.name(c, "")
        if prev is None and cat.startswith("M"):
            return True
        if "VOWEL SIGN" in name and prev is not None:
            pname = unicodedata.name(prev, "")
            if prev == c or (pname.split(" LETTER ")[-1] in _VOWELS and " LETTER " in pname):
                return True
        if c.isalpha():
            scripts.add(name.split()[0] if name else "?")
        prev = c
    return len(scripts) > 1


def well_formed(text, max_bad=0.03):
    """False for text a PDF extraction scrambled (legacy font encodings, visual order)"""
    words = text.split()
    letters = [c for c in text if c.isalpha()]
    if not words or not letters:
        return False
    main = Counter(_script(c) for c in letters).most_common(1)[0][0]
    bad = sum(_invalid(w) or (main != "LATIN" and any("\u00c0" <= c <= "\u024f" for c in w)) for w in words)
    return bad / len(words) <= max_bad


def sample_pages(files, n, seed):
    """n pages [(path, page)] drawn uniformly from all pages of the files [{path, pages}]."""
    allp = [(f["path"], p) for f in files for p in range(f["pages"])]
    return random.Random(seed).sample(allp, min(n, len(allp)))
