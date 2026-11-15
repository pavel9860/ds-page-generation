"""Text helpers of the page writer: cleaning, fonts that cover the text, item trees (h / p / li / v / table / cols)."""

import re
import unicodedata

from hwsynth.scripts import script as char_script
from hwsynth.scripts import text_script

from .config import CHAR_W, MAX_GLYPH_MM, MIN_XH_SCRIPT  # noqa: F401

MIN_XH, MAX_XH = (
    3.0,
    3.4,
)  # x-height of the writing [mm]; 3.0 mm is about 2.1% of the page height word height at 1024 px (hwdet rejects pages under 14)
LEADERS = re.compile(r"[.…·_\-–—=~*|]{3,}")
COMMON = ("Common", "Inherited", "Unknown", None)


def clean(text):
    text = LEADERS.sub(" ", unicodedata.normalize("NFC", text))  # dotted leaders and rules of forms
    return re.sub(r"\s+", " ", "".join(c for c in text if c.isprintable() or c == " ")).strip()


def usable(t):
    n = sum(not c.isspace() for c in t)
    return n >= 1 and sum(unicodedata.category(c)[0] in "LMN" for c in t) >= 0.6 * n  # letters, vowel signs, digits


def trim(text, chars):
    if len(text) <= chars:
        return text
    cut = text[:chars]
    sp = cut.rfind(" ")
    return cut[:sp] if sp > 0.7 * chars else cut


def char_scripts(text):
    """(char, script, shared) of the characters of the text as hwsynth.writer draws them: a word in the font of the script of its letters, Latin for a word
    without letters; shared = punctuation / digits / symbols, which the writer takes from any hand font when that font lacks them (Fonts.for_unit)."""
    for wd in text.split():
        ws = text_script(wd)
        ws = "Latin" if ws in COMMON else ws
        for ch in wd:
            s = char_script(ch)
            yield ch, ws if s in COMMON else s, s == "Common"


def lacks(w, ch, sc, shared):
    """The writer draws no glyph for ch: its font for the script lacks it, and for a shared character no hand font has it either."""
    cm = w.fonts.cmap.get(w.font(sc), ())
    return bool(cm) and ord(ch) not in cm and not (shared and w.fonts.for_unit(None, ch))


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
    of the script (a font asked for by name goes first), else the hand font of any script with most of them. No mixing of fonts inside a word."""
    by = {}
    for ch, sc, shared in char_scripts(text):
        if not shared:
            by.setdefault(sc, set()).add(ch)
    mapping = {}
    for sc, chars in by.items():
        own = w.font(sc)
        cands = (
            ([own] if own else []) + list(w.fonts.candidates(sc, w.font_class)) + list(w.fonts.candidates(sc, "any"))
        )
        name = (prefer or {}).get(sc)
        if name:
            cands = [f for f in cands if str(f).endswith(name)] + cands
        need = "".join(chars)
        mapping[sc] = next((f for f in cands if f and w.fonts.covers(f, need)), None)
        if (
            mapping[sc] is None
        ):  # no font has every character: the one with most of them (the rest is stripped, and a page that lost too much is rejected)
            cov = lambda f: sum(ord(c) in w.fonts.cmap.get(f, ()) for c in need)
            best = max((f for f in cands if f), key=cov, default=None)
            if best is not None and cov(best) >= 0.9 * len(need):
                mapping[sc] = best
    for (
        sc,
        chars,
    ) in by.items():  # a script no font of its own covers: the hand font with most of its characters (the kana font of a Japanese page has its kanji)
        if mapping[sc] is None:
            cov = lambda f: sum(ord(c) in w.fonts.cmap.get(f, ()) for c in chars)
            mapping[sc] = next(
                (
                    f
                    for fs in (
                        [f for f in mapping.values() if f],
                        w.fonts.hand,
                    )  # the page's own fonts first: one hand on the page
                    for f in sorted(fs, key=cov, reverse=True)[:1]
                    if cov(f) >= 0.9 * len(chars)
                ),
                None,
            )
    return {k: v for k, v in mapping.items() if v}


def strip_missing(w, text):
    """The text without the characters the writer's fonts lack (they would be empty boxes) and without orphaned vowel signs."""
    sc = iter(char_scripts(text))
    out, start = [], True
    for ch in text:
        if ch.isspace():
            out.append(ch)
            start = True
            continue
        _, s, shared = next(sc)
        if (unicodedata.category(ch)[0] == "M" and (start or unicodedata.combining(out[-1]) == 9)) or lacks(
            w, ch, s, shared
        ):
            start = True
            continue
        out.append(ch)
        start = False
    return "".join(out)


def uncovered(w, text):
    """Characters of the text that the writer cannot draw (lacks)."""
    return len({c for c, sc, shared in char_scripts(text) if lacks(w, c, sc, shared)})
