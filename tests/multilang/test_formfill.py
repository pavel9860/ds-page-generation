"""detset"""

import random
import re

import pytest

from dspages.multilang import formfill as FF
from dspages.multilang import linebank as LB


@pytest.fixture
def bank(monkeypatch):
    rng = random.Random(0)
    words = [f"{chr(97 + i % 26)}{chr(97 + i // 26 % 26)}{chr(97 + i // 676)}x" for i in range(900)]
    lines = [" ".join(rng.sample(words, rng.randint(1, 10))) for _ in range(3000)]
    b = LB.Bank("en", lines)
    monkeypatch.setattr(LB, "load", lambda lang, root=None: b if lang == "en" else None)
    FF.script_of.cache_clear()
    return b


def item(fid, caps, labels=None):
    lab = lambda i: (labels or {}).get(i, "Description of the work")
    return dict(form_id=fid, boxes=[dict(i=i, chars=c, label=lab(i), label_en=lab(i)) for i, c in enumerate(caps)])


def test_answers_fit_and_free_text_has_no_digits(bank):
    caps = [2, 3, 5, 9, 14, 20, 30, 45, 60, 90] * 4
    answers, prov = FF.fill_form(item("f1", caps), "en", set())
    assert len(answers) >= 0.8 * len(caps) and set(answers) == set(prov)
    assert all(len(answers[k]) <= caps[int(k)] for k in answers)
    assert not any(
        re.search(r"\d", v) for k, v in answers.items() if prov[k][0] == "line"
    )  # numbers only where the label asks


def test_nothing_repeats_inside_or_across_forms(bank):
    used, caps = set(), [12, 18, 25, 40, 60] * 4
    a, _ = FF.fill_form(item("f1", caps), "en", used)
    b, _ = FF.fill_form(item("f2", caps), "en", used)
    r = FF.repeats({"f1": a, "f2": b})
    assert r["extra_copies"] == 0 and r["cross_form"] == 0


def test_label_names_the_kind(bank):
    caps = [20, 20, 20]
    _, prov = FF.fill_form(item("f3", caps, {0: "Post code", 1: "E-mail", 2: "Date of birth"}), "en", set())
    assert [prov[str(i)][0] for i in range(3)] == ["postcode", "email", "date"]
    assert FF.kind_of("Quantity") == "number" and FF.kind_of("Signature") == "name" and FF.kind_of("Remarks") == "line"
