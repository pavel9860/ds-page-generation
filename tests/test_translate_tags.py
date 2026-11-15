"""Whole-page translation with <p id> tags (tools/seed_texts.keyed) and the garbled gate (hand.rare.translated: more than LOST of the tags lost)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).parents[2] / "handwriting_detection" / "src"))
seed_texts = pytest.importorskip("seed_texts")
from dspages.hand import rare  # noqa: E402


def test_keyed_all_tags():
    assert seed_texts.keyed('<p id="0">one</p>\n<p id="1">two words</p>', 2) == ["one", "two words"]


def test_keyed_tag_as_google_mangles_it():                                       # seen: <pid="9">, no </p>, spaces in the tag
    assert seed_texts.keyed('<p id="0">a</p>\n<pid="1">b\n< p id = 2 >c</p>', 3) == ["a", "b", "c"]


def test_keyed_lost_tag_is_none():                                               # seen: the opening tag gone, its </p> left
    assert seed_texts.keyed('<p id="0">a</p>\n</p>\n<p id="2">c</p>', 3) == ["a", None, "c"]


def test_keyed_ignores_ids_out_of_range_and_repeats():
    assert seed_texts.keyed('<p id="0">a</p><p id="0">x</p><p id="7">y</p>', 1) == ["a"]


@pytest.mark.parametrize("lost, garbled", [(0, False), (5, False), (6, True), (10, True)])
def test_gate_more_than_half_lost(monkeypatch, lost, garbled):
    from dspages.multilang import gtranslate as GT
    blocks = [("p", f"text {i}", (0, i, 10, i + 1)) for i in range(10)]
    monkeypatch.setattr(GT, "lines", lambda src, tl, sl="auto": [None if i < lost else f"t{i}" for i in range(len(src))])
    out = rare.translated(blocks, "hi", "en")
    assert (out is None) == garbled
    if out:
        assert [b[1] for b in out][lost:] == [f"t{i}" for i in range(lost, 10)]


@pytest.mark.parametrize("answer, got", [('<p id="0">один</p>\n<p id="1">два</p>', ["один", "два"]), ("", None)])
def test_whole_one_call_tagged(monkeypatch, answer, got):
    calls = []
    monkeypatch.setattr(seed_texts, "_google", lambda text, tl, sl: (calls.append(text), answer)[1])
    assert seed_texts.whole(["one", "two"], "ru", "en") == got and calls == ['<p id="0">one</p>\n<p id="1">two</p>']


def test_no_answer_stops_the_run(monkeypatch):
    monkeypatch.setattr(seed_texts, "_google", lambda text, tl, sl: None)
    with pytest.raises(RuntimeError, match="no answer"):
        seed_texts.tagged_lines(["one"], "ru", "en")

