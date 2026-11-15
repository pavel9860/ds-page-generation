"""Answers for form fields, no model"""

import random
import re
from collections import Counter
from functools import cache

from . import linebank as LB
from .config import FILL as C
from .config import LANGDS as L
from .dettext import INITIALS, Values

_KEYS = [(kind, re.compile(rx)) for kind, rx in C.keys]
_NUM = re.compile(C.numeric)


@cache
def script_of(lang):
    from hwsynth.scripts import text_script

    bank = LB.load(L.alias.get(lang, lang))
    votes = Counter(s for t in bank.lines if (s := text_script(t)) != "Common")  # digits and punctuation tell no script
    return votes.most_common(1)[0][0] if votes else None


def _person(kind, n, V, bank, script, rng):
    p = V.person(script, bank)
    if p is None:
        return None
    text = p[2] if len(p[2]) <= n else (f"{p[0][:1]}. {p[1]}" if script in INITIALS else p[1])
    return text, "faker" if V.fake is not None else "bank-words"


def _faker(kind, V):
    method = getattr(V.fake, {"address": "street_address"}.get(kind, kind), None)
    return str(method()).replace("\n", ", ") if method else None


def other(kind, n, V, bank, script, rng):
    """(value, source) of a non-line kind, None when there is none."""
    if kind == "name":
        return _person(kind, n, V, bank, script, rng)
    simple = {"date": V.date, "phone": V.phone, "email": lambda: V.email(V.cast[-1] if V.cast else None)}
    if kind in simple:
        return simple[kind](), "faker"
    if kind in ("postcode", "city", "company", "job", "country", "address") and V.fake is not None:
        text = _faker(kind, V)
        return (V._digits(text), "faker") if text else None
    if kind == "postcode":
        return V.digits(rng.choice((4, 5, 6))), "faker"
    return None


def _number(n, V, rng):
    """A number of at most n digits (at most 3)."""
    return V._digits(str(rng.randint(1, 10 ** min(n, 3) - 1)))


def kind_of(label_en):
    """The kind of value a field's label asks for (its English translation against C.keys), "line" (free text) for any other."""
    return next((k for k, rx in _KEYS if rx.search(label_en.casefold())), "line")


def fill_form(item, lang, used):
    """({field: answer}, {field: [kind, source, key]}) of one form: each field the kind its label names (box label_en, fill_forms), free text
    without digits for the others (numbers only where the label asks for one); a field nothing fits stays empty."""
    rng = random.Random(item["form_id"])
    V = Values(lang, rng)
    bank = LB.load(L.alias.get(lang, lang)) or LB.Bank(lang, [])
    script = script_of(lang)
    stems, answers, prov = set(), {}, {}
    for box in item["boxes"]:
        n, kind = box["chars"], kind_of(box.get("label_en", ""))
        for _ in range(C.tries):  # a value that fits the field
            if kind == "line":
                got = (bank.pick(n, rng, used, stems, digits=False), "bank") if len(bank) else None
            elif kind == "number":
                got = _number(n, V, rng), "faker"
            else:
                got = other(kind, n, V, bank, script, rng)
            if got and got[0] and len(got[0]) <= n:
                answers[str(box["i"])] = got[0]
                prov[str(box["i"])] = [kind, got[1], LB.key(got[0])]
                break
    return answers, prov


def labels_en(items):
    """Each box's label translated into English (box label_en), one request per form: the field kinds are read in one language."""
    from . import gtranslate as GT

    for item in items:
        todo = [b for b in item["boxes"] if "label_en" not in b]
        if todo:
            got = (
                GT.lines([b.get("label") or "-" for b in todo], "en")
                if any(b.get("label") for b in todo)
                else [""] * len(todo)
            )
            for b, t in zip(todo, got):
                b["label_en"] = (t or "") if b.get("label") else ""
    return items


def fill_forms(items, meta):
    """(answers, provenance) of every item, in order; a bank line is written once in the run"""
    used, answers, prov = set(), {}, {}
    for item in labels_en(items):
        answers[item["form_id"]], prov[item["form_id"]] = fill_form(item, meta[item["form_id"]]["lang"], used)
    return answers, prov


def repeats(answers):
    """Extra copies of a value inside a form, answers sharing their opening, and text values used in several forms."""
    seen, within, total, openings = Counter(), 0, 0, 0
    for d in answers.values():
        keys = Counter(LB.key(v) for v in d.values() if v)
        total += sum(keys.values())
        within += sum(n - 1 for n in keys.values())
        openings += sum(n - 1 for n in Counter(LB.stem(v) for v in d.values() if v and len(v.split()) > 1).values())
        seen.update(keys)
    cross = sum(1 for v, n in seen.items() if n > 1 and len(v) > 3 and not _NUM.match(v))  # a word, not a short code
    return dict(answers=total, extra_copies=within, same_opening=openings, cross_form=cross)
