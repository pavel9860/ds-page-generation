"""Translation of bank lines without a browser: planning, batching, cutting, raw kept."""

import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from dspages.multilang import gtranslate

sys.path.insert(0, str(Path(__file__).parents[2] / "tools" / "multilang" / "translation"))
gt = importlib.import_module("gt_lines")
PAR = "Please open the file first, then read the first page. It holds the rules, which every member must know. Ask the clerk if unsure."
LONG = " ".join(f"w{i}" for i in range(37))


@pytest.fixture(autouse=True)
def plain_lines(monkeypatch):
    """gtranslate.lines as shipped: tools/seed_texts.py replaces it at import (tagged texts) for the rest of the process."""
    spec = importlib.util.spec_from_file_location("dspages.multilang._gtranslate_fresh", gtranslate.__file__)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    f = fresh.lines
    monkeypatch.setattr(
        gtranslate, "lines", types.FunctionType(f.__code__, vars(gtranslate), f.__name__, f.__defaults__, f.__closure__)
    )


def test_cutting_loses_no_word_and_makes_short_lines():
    for text in (PAR, LONG, LONG + ", " + PAR):
        lines = gt.LB.phrases(text, "en")
        assert " ".join(lines).split() == text.replace(",", "").replace(".", "").split()
        assert all(len(line.split()) <= gt.LB.C.max_words for line in lines)


def test_unspaced_text_is_cut_into_graphemes_without_loss():
    text = "这是一个很长的句子没有任何标点符号所以需要被切开成几个较短的部分" * 2
    assert "".join(gt.LB.phrases(text, "zh")) == text


def test_translate_all_keeps_every_paragraph_in_order_and_marks_failures():
    jobs = [("aa", "fr", PAR), ("aa", "fr", PAR + " Another one."), ("bb", "de", PAR)]
    out = gt.translate_all(lambda l, sl, p: p.upper() if sl == "fr" else None, jobs, workers=3)
    assert (
        [r["src"] for r in out["aa"]] == [PAR, PAR + " Another one."]
        and out["bb"][0]["tr"] is None
        and out["aa"][0]["from"] == "fr"
    )


def test_a_429_is_waited_out_then_the_text_comes(monkeypatch):
    answers = iter([429, 429, 200])

    class R:
        def __init__(self, code):
            self.status_code = code

        def json(self):
            return [[["ok", "x"]]]

    monkeypatch.setattr(gtranslate.requests, "post", lambda *a, **k: R(next(answers)))
    monkeypatch.setattr(gtranslate.time, "sleep", lambda s: None)
    assert gtranslate.google("x", "ur") == "ok"
    monkeypatch.setattr(gtranslate.requests, "post", lambda *a, **k: R(500))
    assert gtranslate.google("x", "ur") is None


def test_lines_are_unique_and_none_translation_gives_none():
    raw = [
        {"from": "fr", "src": "a", "tr": PAR},
        {"from": "fr", "src": "b", "tr": PAR},
        {"from": "de", "src": "c", "tr": None},
    ]
    lines = gt.lines_of(raw, "en")
    assert len(lines) == len({gt.LB.key(x) for x in lines}) and "Ask the clerk if unsure" in lines


def test_plan_uses_every_paragraph_once_and_covers_the_need(monkeypatch):
    src = {"en": [PAR + f" {i}" for i in range(60)], "fr": [PAR + f" f{i}" for i in range(60)]}
    monkeypatch.setattr(gt, "missing", lambda l: 30 if l in ("aa", "bb") else 0)
    plan = gt.plan(["aa", "bb", "cc"], src)
    used = [p for v in plan.values() for p in v]
    assert set(plan) == {"aa", "bb"} and len(used) == len(set(used))
    assert all(sum(len(gt.LB.phrases(p, s)) for s, p in v) >= 30 for v in plan.values())


def test_missing_counts_translated_lines_and_translated_paragraphs_are_not_planned_again(monkeypatch, tmp_path):
    monkeypatch.setattr(gt.LB, "BANK", tmp_path / "bank")
    monkeypatch.setattr(gt.LB, "TRANSLATED", tmp_path / "tr")
    (tmp_path / "bank").mkdir()
    (tmp_path / "tr").mkdir()
    (tmp_path / "bank" / "aa.json").write_text('{"w": ["x"]}')
    (tmp_path / "tr" / "aa.json").write_text(
        json.dumps({"raw": [{"from": "en", "src": "p1", "tr": "t"}], "w": ["a", "b"]})
    )
    assert gt.missing("aa") == gt.C.min_lines - 3
    monkeypatch.setattr(gt, "PARAS", tmp_path)
    (tmp_path / "en.json").write_text('{"p": ["p1", "p2"]}')
    assert gt.sources(["aa"]) == {"en": ["p2"]}


def test_page_lines_come_back_one_per_line_also_when_google_merges_lines(monkeypatch):
    monkeypatch.setattr(gtranslate.time, "sleep", lambda s: None)
    replies = {"a\nb": "A B", "a": "A", "b": "B"}
    monkeypatch.setattr(gtranslate, "google", lambda text, tl, sl="auto": replies[text])
    assert gtranslate.lines(["a", "b"], "ur") == ["A", "B"]
    monkeypatch.setattr(gtranslate, "google", lambda text, tl, sl="auto": {"a\nb": "A\nB"}.get(text))
    assert gtranslate.lines(["a", "b"], "ur") == ["A", "B"]
