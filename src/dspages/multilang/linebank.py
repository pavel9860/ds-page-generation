"""Short written lines per language"""
import json
import re
from bisect import bisect_left, bisect_right
from functools import lru_cache
from pathlib import Path

import regex

from .config import LINEBANK as C
from .paths import LANGDS

BANK = LANGDS / "linebank"
TRANSLATED = LANGDS / "linebank_tr"
_SENT, _CLAUSE = re.compile(C.sentence), re.compile(C.clause)


def key(line):
    return re.sub(r"[\W_]+", " ", line.casefold()).strip()


def stem(line):
    return " ".join(key(line).split()[:C.stem_words])


def _graphemes(text):
    return regex.findall(r"\X", text)


def _pieces(part, lang):
    """A clause as it is, or cut into equal pieces of at most `max_words` words (unspaced scripts: of that many words' worth of graphemes)."""
    if lang in C.unspaced and len(part.split()) < 3:
        g, k = _graphemes(part), C.max_words * C.unit.get(lang, C.unit_default)
        return ["".join(g[i:i + k]).strip(C.edge) for i in range(0, len(g), k)]
    words = part.split()
    size = -(-len(words) // -(-len(words) // C.max_words))
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def phrases(text, lang):
    """The clauses of a paragraph, cut at sentence and clause marks; nothing is dropped."""
    parts = (part.strip(C.edge) for sent in _SENT.split(text) for part in _CLAUSE.split(sent))
    return [piece for part in parts if part for piece in _pieces(part, lang) if piece]


class Bank:
    """The lines of one language sorted by length."""

    def __init__(self, lang, lines):
        self.lang = lang
        self.lines = sorted(set(lines), key=lambda s: (len(s), s))
        self.lens = [len(s) for s in self.lines]

    def __len__(self):
        return len(self.lines)

    @staticmethod
    def _free(line, used, stems):
        return key(line) not in used and (stems is None or stem(line) not in stems)

    @staticmethod
    def _take(line, used, stems):
        used.add(key(line))
        if stems is not None:
            stems.add(stem(line))
        return line

    def pick(self, cap, rng, used, stems=None, digits=True):
        """A line of at most `cap` characters whose length is near a random share of the field (digits=False: a line without digits)"""
        hi = bisect_right(self.lens, cap)
        if cap < C.min_cap or not hi:
            return None
        share = C.share if cap > C.short_cap else C.short_share
        for t in range(C.tries):
            want = max(C.min_cap, cap * rng.uniform(*share))
            lo = bisect_left(self.lens, want * (C.lo_ratio - C.lo_step * t))
            top = min(hi, bisect_right(self.lens, want * (C.hi_ratio + C.hi_step * t)))
            if top <= lo:
                lo, top = max(0, hi - C.fallback), hi
            line = self.lines[rng.randrange(lo, top)]
            if self._free(line, used, stems) and (digits or not any(c.isdigit() for c in line)):
                return self._take(line, used, stems)
        return None

    def words(self, n, rng):
        """n single words (names for a language without a Faker locale)."""
        one = [s for s in self.lines if " " not in s]
        one = [s for s in one if s[:1].isupper()] or one
        return [rng.choice(one) for _ in range(n)] if one else []


@lru_cache(maxsize=None)
def load(lang, root=None):
    """The Bank of a language with its translated lines, None when it was not built."""
    f = Path(root or BANK) / f"{lang}.json"
    if not f.exists():
        return None
    t = Path(root or BANK).parent / TRANSLATED.name / f"{lang}.json"
    return Bank(lang, json.loads(f.read_text())["w"] + (json.loads(t.read_text())["w"] if t.exists() else []))
