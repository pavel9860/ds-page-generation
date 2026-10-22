"""Text helpers of the page writer: cleaning, fonts that cover the text, item trees (h / p / li / v / table / cols)."""
import re
import unicodedata

from hwsynth.scripts import script as char_script

CHAR_W = {"Latin": 1.8, "Cyrillic": 2.0, "Greek": 2.2, "Han": 3.6, "Hiragana": 3.6, "Katakana": 3.6, "Hangul": 3.8, "Arabic": 2.5,
          "Devanagari": 2.6, "Bengali": 2.4, "Tamil": 2.3, "Telugu": 1.95, "Thai": 1.4}   # x-heights per character, measured
MIN_XH, MAX_XH = 1.0, 3.4                              # x-height of the writing [mm]
MAX_GLYPH_MM = 15.0                                    # tallest word of a page (ascender to descender)
LEADERS = re.compile(r"[.…·_\-–—=~*|]{3,}")
COMMON = ("Common", "Inherited", "Unknown", None)


def clean(text):
    text = LEADERS.sub(" ", unicodedata.normalize("NFC", text))          # dotted leaders and rules of forms
    return re.sub(r"\s+", " ", "".join(c for c in text if c.isprintable() or c == " ")).strip()


def usable(t):
    n = sum(not c.isspace() for c in t)
    return n >= 1 and sum(unicodedata.category(c)[0] in "LMN" for c in t) >= 0.6 * n       # letters, vowel signs, digits


def trim(text, chars):
    if len(text) <= chars:
        return text
    cut = text[:chars]
    sp = cut.rfind(" ")
    return cut[:sp] if sp > 0.7 * chars else cut


def script_of(ch):
    s = char_script(ch)
    return "Latin" if s in COMMON else s


def walk_text(items):
    for i in items:
        if i["k"] == "cols":
            for c in i["cols"]:
                yield from walk_text(c["items"])
        elif i["k"] == "table":
            for r in i["rows"]:
                yield from r
        else:
            yield i["t"]


def flat_items(items):
    for i in items:
        if i["k"] == "cols":
            for c in i["cols"]:
                yield from flat_items(c["items"])
        else:
            yield i


def map_text(items, fn):
    out = []
    for i in items:
        if i["k"] == "cols":
            out.append(dict(i, cols=[dict(c, items=map_text(c["items"], fn)) for c in i["cols"]]))
        elif i["k"] == "table":
            out.append(dict(i, rows=[[fn(c) for c in r] for r in i["rows"]]))
        else:
            out.append(dict(i, t=fn(i["t"])))
    return out


def pick_fonts(w, text, prefer=None):
    """One font per script that has every character of the page: the writer's own when it covers the text, else another handwriting font
    of the script (a font asked for by name goes first). No mixing of fonts inside a word."""
    by = {}
    for ch in set(text):
        if not ch.isspace():
            by.setdefault(script_of(ch), set()).add(ch)
    mapping = {}
    for sc, chars in by.items():
        own = w.font(sc)
        cands = ([own] if own else []) + list(w.fonts.candidates(sc, w.font_class)) + list(w.fonts.candidates(sc, "any"))
        name = (prefer or {}).get(sc)
        if name:
            cands = [f for f in cands if str(f).endswith(name)] + cands
        need = "".join(chars)
        mapping[sc] = next((f for f in cands if f and w.fonts.covers(f, need)), None)
    return {k: v for k, v in mapping.items() if v}


def strip_missing(w, text):
    """The text without the characters the writer's fonts lack (they would be empty boxes) and without orphaned vowel signs."""
    out, start = [], True
    for ch in text:
        if ch.isspace():
            out.append(ch)
            start = True
            continue
        cm = w.fonts.cmap.get(w.font(script_of(ch)), ())
        if (unicodedata.category(ch)[0] == "M" and (start or unicodedata.combining(out[-1]) == 9)) or (cm and ord(ch) not in cm):
            start = True
            continue
        out.append(ch)
        start = False
    return "".join(out)


def uncovered(w, text):
    """Characters of the text that the writer's font of their script does not have."""
    n = 0
    for ch in set(text):
        if not ch.isspace():
            cm = w.fonts.cmap.get(w.font(script_of(ch)), ())
            n += bool(cm) and ord(ch) not in cm
    return n
