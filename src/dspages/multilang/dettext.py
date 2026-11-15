"""Text written by hand in the detection set, kept to what a person writes on a document of the language"""

import json
import random
import re
import unicodedata
from functools import cache, lru_cache

from .config import VALUES as V
from .paths import DATA as D

UNSPACED = {"th", "lo", "km", "my", "bo", "dz", "zh", "ja"}
NATIVE_DIGITS = {
    "ar": "Arabic",
    "bn": "Bengali",
    "as": "Bengali",
    "my": "Myanmar",
    "km": "Khmer",
    "lo": "Lao",
    "ne": "Devanagari",
    "mr": "Devanagari",
    "bo": "Tibetan",
    "dz": "Tibetan",
    "th": "Thai",
    "fa": "Persian",
    "ur": "Persian",
    "ps": "Persian",
}
INITIALS = {"Latin", "Cyrillic", "Greek"}
CITE = re.compile(
    r"(\bet al\b|\bpp?\.\s*\d|\bIn:|\bISBN\b|\bISSN\b|https?://|www\.|\bdoi\b|@|\(\d{4}[a-z]?\)|\b(19|20)\d\d[a-z]?\)\s*[.,:]"
    r"|\bvol\.\s*\d|\beds?\.\s|\bno\.\s*\d|©|\[\d+\])",
    re.I,
)
GROUPS = [
    {"hr", "sr", "bs", "sl"},
    {"id", "ms"},
    {"hi", "mr", "ne"},
    {"no", "da", "nb", "nn", "sv"},
    {"cs", "sk"},
    {"bg", "mk"},
    {"ru", "uk", "be", "kk", "ky", "mn", "tg"},
    {"fa", "ur", "ps"},
    {"es", "gl", "pt", "ca"},
    {"af", "nl"},
    {"zh", "ja"},
    {"ms", "tl"},
]


@lru_cache(maxsize=1)
def _detector():
    from lingua import Language, LanguageDetectorBuilder

    return LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build(), {
        l.iso_code_639_1.name.lower(): l for l in Language.all()
    }


def same_language(text, lang):
    """False only when Lingua is confident (>= 0.7) that the text is a language outside the group of `lang`."""
    det, iso = _detector()
    if lang not in iso or len(text) < 20:
        return True
    conf = det.compute_language_confidence_values(text)
    if not conf:
        return True
    top = conf[0]
    code = top.language.iso_code_639_1.name.lower()
    if code == lang or top.value < 0.7:
        return True
    return any(lang in g and code in g for g in GROUPS)


def script_of(text):
    from hwsynth.scripts import text_script

    return text_script(text)


@cache
def clean_pool(lang, script=None):
    """Texts of the language fit to be written by hand (see the module doc); script"""
    f = D / "langds/paras" / f"{lang}.json"
    if not f.exists():
        return ()
    d = json.loads(f.read_text())
    texts = [" ".join(t.split()) for t in d.get("p", []) + d.get("li", []) + d.get("h", [])]
    texts = [t for t in texts if len(t) >= 3 and not CITE.search(t)]
    if not texts:
        return ()
    if script is None:
        from collections import Counter

        script = Counter(script_of(t) for t in texts[:200]).most_common(1)[0][0]
    return tuple(t for t in texts if script_of(t) == script and same_language(t, lang))


def cut(text, n, lang):
    """About n words from the start of text (a sentence end inside is kept as the end)."""
    if lang in UNSPACED and len(text.split()) < 3:
        k = {"zh": 2, "ja": 3}.get(lang, 5) * n
        j = min(len(text), k)
        while j < len(text) and (unicodedata.category(text[j]).startswith("M") or text[j - 1] in "្ฺ္"):
            j += 1
        return text[:j].strip()
    return " ".join(text.split()[:n]).rstrip(",;:(")


_CONS, _VOW = "bcdfghjklmnprstvwz", "aeiou"


def _handle(rng):
    """A pronounceable ASCII handle: names of a script that has no Latin form get one of these in an e-mail address."""
    return "".join(rng.choice(_CONS) + rng.choice(_VOW) for _ in range(rng.randint(*V.handle)))


class Values:
    """Short values a person writes by hand in a language"""

    def __init__(self, lang, rng: random.Random):
        self.lang, self.rng = lang, rng
        self.fake = None
        self.cast = []
        from faker import Faker  # required: a missing package fails here, not as forms without names
        from faker.config import AVAILABLE_LOCALES

        loc = [l for l in AVAILABLE_LOCALES if l == lang or l.startswith(lang + "_")]
        if loc:  # a language Faker has no locale for: names from the bank (person)
            self.fake = Faker(rng.choice(sorted(loc)) if len(loc) < 6 else sorted(loc, key=len)[0])
            self.fake.seed_instance(rng.getrandbits(32))

    def _digits(self, s):
        sc = NATIVE_DIGITS.get(self.lang)
        if sc and self.rng.random() < 0.5:
            from dspages.pagegen.canvas import native_digits

            m = {str(i): chr(0x06F0 + i) for i in range(10)} if sc == "Persian" else native_digits(sc)
            return "".join(m.get(c, c) for c in s)
        return s

    def date(self, lo=V.years[0], hi=V.years[1]):
        """A date in one of the formats people write; with a month name when the locale has them."""
        r = self.rng
        d, m, y = r.randint(1, 28), r.randint(1, 12), r.randint(lo, hi)
        f = r.choice(
            [
                "{d:02d}.{m:02d}.{y}",
                "{d}/{m}/{y}",
                "{d:02d}/{m:02d}/{yy:02d}",
                "{y}-{m:02d}-{d:02d}",
                "{d}.{m}.{yy:02d}",
                "{d:02d}-{m:02d}-{y}",
                "{d}.{m}.{y}",
                "{m:02d}/{d:02d}/{y}",
            ]
        )
        if self.fake is not None and self.lang in V.month_names and r.random() < V.p_month_name:
            f = r.choice(["{d} {M} {y}", "{M} {d}, {y}", "{d}-{M}-{yy:02d}"])
            return f.format(d=d, M=self.fake.month_name(), y=y, yy=y % 100)
        if self.lang in ("zh", "ja"):
            f = r.choice(["{y}年{m}月{d}日", "{m}月{d}日", "{y}/{m}/{d}"])
        elif self.lang == "ko":
            f = r.choice(["{y}.{m}.{d}", "{y}년 {m}월 {d}일"])
        return self._digits(f.format(d=d, m=m, y=y, yy=y % 100))

    def digits(self, n):
        """An identifier of n digits (no leading zero), sometimes grouped."""
        r = self.rng
        s = str(r.randint(1, 9)) + "".join(r.choice("0123456789") for _ in range(max(n - 1, 0)))
        if n >= 7 and r.random() < V.p_group_digits:
            g = r.choice([3, 4])
            s = " ".join(s[i : i + g] for i in range(0, len(s), g))
        return self._digits(s)

    def number(self, cap=None):
        r = self.rng
        opts = [
            str(r.randint(1, 99)),
            str(r.randint(100, 99999)),
            f"{r.randint(1, 999)},{r.randint(0, 99):02d}",
            f"{r.randint(1, 99)}%",
            f"№ {r.randint(1, 999)}",
            f"{r.randint(1, 9)}.{r.randint(1, 9)}",
            f"x{r.randint(2, 12)}",
            str(r.randint(1, 12)),
            f"{r.randint(10, 999)}/{r.randint(1, 99)}",
            f"{r.choice('ABCDEFKMPRST')}-{r.randint(100, 9999)}",
        ]
        s = r.choice(opts)
        return self._digits(s[:cap] if cap else s)

    def phone(self):
        if (
            self.fake is not None and hasattr(self.fake, "phone_number") and self.rng.random() < 0.7
        ):  # a locale without phone numbers: digits
            return self._digits(self.fake.phone_number())
        r = self.rng
        return self._digits(r.choice(["+", "", "0"]) + self.digits(r.choice([9, 10, 11])))

    def _name_parts(self):
        """(first, last) of the locale of one sex (a surname that has female and male forms agrees with the first name)"""
        f, sex = self.fake, self.rng.choice(("female", "male"))
        if f is None:
            return None
        first = getattr(f, f"first_name_{sex}", f.first_name)()
        last = getattr(f, f"last_name_{sex}", f.last_name)()
        return first.strip(), last.strip()

    def person(self, script=None, bank=None):
        """A new person (first, last, full) not given before by this Values"""
        for _ in range(30):
            p = self._name_parts()
            if p is None and bank is not None:
                w = bank.words(2, self.rng)
                p = (w[0], w[1]) if len(w) == 2 else None
            if p is None or (script is not None and script_of(p[0] + p[1]) != script):
                continue
            first, last = p
            a, b = (last, first) if self.lang in V.last_first else (first, last)
            full = (a + b) if self.lang in V.joined else f"{a} {b}"
            if full not in {c[2] for c in self.cast}:
                self.cast.append((first, last, full))
                return self.cast[-1]
        return None

    def name(self, script=None):
        """A person's name of the locale, None when there is no locale or its names are not in `script`."""
        p = self.person(script)
        return p[2] if p else None

    def email(self, person=None):
        r = self.rng
        first, last = (person[0], person[1]) if person else (None, None)
        if first and (first + last).isascii():
            a, b = re.sub(r"[^a-z]", "", first.lower()), re.sub(r"[^a-z]", "", last.lower())
        else:
            a, b = _handle(r), _handle(r)
        user = r.choice(
            [
                f"{a}.{b}",
                f"{a[:1]}{b}",
                f"{a}{b}",
                f"{a}_{b}",
                f"{b}.{a}",
                f"{a}{r.randint(1, 99)}",
                f"{a[:1]}.{b}{r.randint(1, 9)}",
            ]
        )
        return f"{user}@{r.choice(V.domains)}"
