"""Logical order of Indic text from the glyph order of a PDF: producers that do not shape-map their text write a vowel sign
that stands before its consonant cluster (i, e, ai, o...) and the reph (ra + virama) after it, as the glyphs sit on the line;
Unicode stores the sign after the cluster and the reph before it."""
import unicodedata

PREFIX = {0x093F, 0x09BF, 0x09C7, 0x09C8, 0x0A3F, 0x0ABF, 0x0BC6, 0x0BC7, 0x0BC8, 0x0D46, 0x0D47, 0x0D48, 0x0DD9, 0x0DDA,
          0x0DDB, 0x17C1, 0x17C2, 0x17C3, 0x1031}
RA = {0x0930, 0x09B0, 0x0AB0, 0x0A30}
VIRAMA = {0x094D, 0x09CD, 0x0ACD, 0x0A4D, 0x0BCD, 0x0C4D, 0x0CCD, 0x0D4D, 0x0DCA, 0x17D2, 0x1039}
NUKTA = {0x093C, 0x09BC, 0x0ABC, 0x0A3C}


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


def visual_order(text):
    """Whether text has a prefix vowel sign where it cannot be in logical order: at the start of a word or after anything
    but a consonant cluster (its sign is stored after the consonant)."""
    for i, c in enumerate(text):
        if ord(c) in PREFIX and (i == 0 or not (_base(text[i - 1]) or ord(text[i - 1]) in NUKTA or ord(text[i - 1]) in VIRAMA)):
            return True
    return False


def to_logical(text):
    """Prefix vowel signs moved after the cluster that follows them, a reph (ra + virama) at the end of a word moved before
    the cluster it stands over; two-part vowels (e + aa) recomposed."""
    out, i = [], 0
    while i < len(text):
        c = text[i]
        if ord(c) in PREFIX and i + 1 < len(text) and _base(text[i + 1]):
            j = _cluster_end(text, i + 1)
            out.append(text[i + 1:j] + c)
            i = j
        else:
            out.append(c)
            i += 1
    t = "".join(out)
    words = []
    for w in t.split(" "):
        if len(w) >= 3 and ord(w[-1]) in VIRAMA and ord(w[-2]) in RA:
            k = len(w) - 2
            start = k - 1
            while start > 0 and (ord(w[start]) in NUKTA or ord(w[start]) in VIRAMA or unicodedata.category(w[start]).startswith("M")
                                 or (start > 0 and ord(w[start - 1]) in VIRAMA)):
                start -= 1
            w = w[:start] + w[k:] + w[start:k]
        words.append(w)
    return unicodedata.normalize("NFC", " ".join(words))
