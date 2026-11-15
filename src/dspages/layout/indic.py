"""Logical order of Indic text from the glyph order of a PDF: producers that do not shape-map their text write a vowel sign
that stands before its consonant cluster (i, e, ai, o...) and the reph (ra + virama) after it, as the glyphs sit on the line;
Unicode stores the sign after the cluster and the reph before it."""

import unicodedata

import regex

from .config import NUKTA, PREFIX, RA, VIRAMA  # noqa: F401

_INSC = regex.compile(r"\p{InSC=Consonant}|\p{InSC=Consonant_Dead}")

KHMER_RO = "\u17d2\u179a"  # coeng ro (subscript ro)
KHMER_SPLIT = regex.compile(
    "\u17c1(?=[\u17be\u17bf\u17c0\u17c4\u17c5])"
)  # Khmer e drawn as the left piece of oe, ya, ie, oo, au
REPH = "\ue0f0"  # marks the text of a reph glyph (GSUB rphf), drawn after its syllable


def _base(c):
    return unicodedata.category(c) == "Lo" and ord(c) not in PREFIX


def _cluster_end(t, i):
    """End (exclusive) of the consonant cluster that starts at t[i]: base, nukta, then (virama, base)* ."""
    j = i + 1
    while j < len(t):
        if ord(t[j]) in NUKTA:
            j += 1
        elif ord(t[j]) in VIRAMA and j + 1 < len(t) and _base(t[j + 1]):
            j += 2
        else:
            break
    return j


def _consonant(c):
    """A consonant by the Unicode property Indic_Syllabic_Category (an independent vowel takes no vowel sign)."""
    return bool(_INSC.match(c))


def _prefix_first(text):
    """A prefix vowel sign where it cannot be in logical order: at the start of a word or after anything but a consonant cluster; or a Khmer coeng
    ro at the start of a word (in logical order it follows its consonant)."""
    return any(
        ord(c) in PREFIX
        and (i == 0 or not (_consonant(text[i - 1]) or ord(text[i - 1]) in NUKTA or ord(text[i - 1]) in VIRAMA))
        or text.startswith(KHMER_RO, i)
        and (i == 0 or text[i - 1].isspace())
        for i, c in enumerate(text)
    )


def visual_order(text):
    """Whether text has a sign in drawing order: a prefix vowel sign before its cluster (_prefix_first), or a reph glyph (REPH: its GSUB rphf text)."""
    return _prefix_first(text) or REPH in text


def to_logical(text):
    """Prefix vowel signs moved after the cluster that follows them (when the text has them first: _prefix_first), a reph (ra + virama) at the end
    of a word moved before the cluster it stands over; two-part vowels (e + aa) recomposed."""
    out, i, first = [], 0, _prefix_first(text)
    while i < len(text):
        c = text[i]
        if first and ord(c) in PREFIX and i + 1 < len(text) and _base(text[i + 1]):
            j = _cluster_end(text, i + 1)
            out.append(text[i + 1 : j] + c)
            i = j
        elif (
            first and text.startswith(KHMER_RO, i) and i + 2 < len(text) and _base(text[i + 2])
        ):  # Khmer coeng ro, drawn left of its consonant
            j = _cluster_end(text, i + 2)
            out.append(text[i + 2 : j] + KHMER_RO)
            i = j
        else:
            out.append(c)
            i += 1
    t = "".join(out)
    t = (
        KHMER_SPLIT.sub("", t).replace("\u17c1\u17b8", "\u17be").replace("\u17c1\u17b6", "\u17c4")
    )  # Khmer two-part vowels have no decomposition
    # (NFC keeps them apart): the left piece e before oe / ya / ie / oo / au goes, e + ii is oe, e + aa is oo
    t = t.replace(REPH, "")
    words = []
    for w in t.split(" "):
        if len(w) >= 3 and ord(w[-1]) in VIRAMA and ord(w[-2]) in RA:
            k = len(w) - 2
            start = k - 1
            while start > 0 and (
                ord(w[start]) in NUKTA
                or ord(w[start]) in VIRAMA
                or unicodedata.category(w[start]).startswith("M")
                or (start > 0 and ord(w[start - 1]) in VIRAMA)
            ):
                start -= 1
            w = w[:start] + w[k:] + w[start:k]
        words.append(w)
    return unicodedata.normalize("NFC", " ".join(words))
