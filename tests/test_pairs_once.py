"""Every job of the pairs pipeline (tools/shift_test.py cmd_pairs) is done once: the page text read once (gate and translation share it), a form's
fields found once and its labels translated once (gate, filling), the hand pages written once (no rerender), a gated page not gated again, a
translation not made again, one process pool per size for the whole run."""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
ST = pytest.importorskip("shift_test")
from conftest import needs_fonts, needs_tables

from dspages import paths as _P  # noqa: E402
from dspages.hand import rare  # noqa: E402
from dspages.multilang import filledforms as FF  # noqa: E402
from dspages.multilang import formfill as FO  # noqa: E402

pytestmark = [needs_fonts, needs_tables]

TEXT = [["p", "word " * 200, [0, 0, 100, 10], 10.0, {}]]


@pytest.fixture
def pairs(tmp_path, monkeypatch):
    monkeypatch.setattr(ST, "PAIRS", tmp_path)
    monkeypatch.setattr(ST, "IMG", tmp_path / "img")
    import pymupdf

    pdf = tmp_path / "doc_ru.pdf"
    with pymupdf.open() as d:
        for _ in range(3):
            d.new_page()
        d.save(pdf)
    return tmp_path, str(pdf)


def counter(monkeypatch, obj, name, ret):
    n = []
    monkeypatch.setattr(obj, name, lambda *a, **k: (n.append(a), ret(*a))[1])
    return n


def test_page_text_read_once(pairs, monkeypatch):
    _, pdf = pairs
    reads = counter(monkeypatch, rare, "page_blocks", lambda *a: (TEXT, "text layer"))
    monkeypatch.setattr(rare, "translated", lambda b, tl, sl: b)
    assert ST.clean(pdf, 0, "ru", __import__("collections").Counter(), "text")  # the gate
    ST._translate((7, pdf, 0, "ru", "hi"))  # the translation
    assert ST.clean(pdf, 0, "ru", __import__("collections").Counter(), "text")
    assert len(reads) == 1


def test_form_fields_once(pairs, monkeypatch):
    _, pdf = pairs
    monkeypatch.setattr(rare, "page_blocks", lambda *a: (TEXT, "text layer"))
    item = dict(form_id="f1", boxes=[dict(i=0, chars=8, label="Имя")])
    preps = counter(
        monkeypatch,
        FF,
        "prep_one",
        lambda job: (json.loads(json.dumps(item)), dict(fid="f1", path=job[0], page=job[1], lang=job[2])),
    )
    assert ST.clean(pdf, 0, "ru", __import__("collections").Counter(), "form")  # the gate finds the fields
    for _ in range(2):  # the filling (and a fill run again)
        ST._prep((pdf, 0, "ru"))
    assert len(preps) == 1


def test_form_labels_translated_once(pairs, monkeypatch):
    _, pdf = pairs
    item = dict(form_id="f1", boxes=[dict(i=0, chars=8, label="Имя")])
    monkeypatch.setattr(
        FF,
        "prep_one",
        lambda job: (json.loads(json.dumps(item)), dict(fid="f1", path=job[0], page=job[1], lang=job[2])),
    )
    labels = counter(
        monkeypatch,
        FO,
        "labels_en",
        lambda items: [b.update(label_en="Name") for it in items for b in it["boxes"]] and items,
    )
    for _ in range(2):  # the filling, and a fill run again
        assert ST._labels((pdf, 0, "ru"))["boxes"][0]["label_en"] == "Name"
    assert len(labels) == 1


def test_form_prepped_again_when_pdf_newer(pairs, monkeypatch):
    _, pdf = pairs
    preps = counter(monkeypatch, FF, "prep_one", lambda job: (dict(form_id="f", boxes=[]), dict(fid="f")))
    ST.prepped(pdf, 0, "ru")
    t = time.time() - 60
    os.utime(ST._prepfile(pdf, 0, "ru"), (t, t))  # the translation made again after its fields were found
    ST.prepped(pdf, 0, "ru")
    ST.prepped(pdf, 0, "ru")
    assert len(preps) == 2


def test_hand_pages_written_once(pairs, monkeypatch):
    _, pdf = pairs
    ST._write(ST.trfile(dict(path=pdf, page=0, tl="hi")), dict(blocks=TEXT, tr=TEXT))
    monkeypatch.setattr(ST, "have", lambda i, e, r: e == "reflow")
    monkeypatch.setattr(ST, "printed", lambda r, tr: ({}, b""))
    plan = dict(persona="a", style="b", pen="c", paper="d", fmt="e", dense=False)
    writes = counter(
        monkeypatch,
        rare,
        "write",
        lambda *a: (np.zeros((4, 4), np.uint8), dict(lines=1, lines_want=10), plan, None, "text cut short"),
    )
    i, out = ST.render((3, dict(kind="text", path=pdf, page=0, lang="ru", tl="hi", pages=1), None))
    assert len(writes) == 2 and [e for e, _, _ in out] == [
        "tr_print",
        "hand",
        "tr_hand",
    ]  # a short page is kept, not written again


def test_pages_from_the_estimate_nothing_written(pairs, monkeypatch):
    _, pdf = pairs
    from dspages.hand import page

    reads = counter(monkeypatch, rare, "page_blocks", lambda *a: (TEXT, "text layer"))
    monkeypatch.setattr(page, "fills", lambda plan, items: len(items) >= 2)  # two pages of text fill the hand page
    monkeypatch.setattr(page, "make", lambda *a: pytest.fail("a page written to count the pages"))
    r = dict(kind="text", path=pdf, page=0, lang="ru", tl="hi")
    assert ST._pages((3, r)) == (3, 2)
    assert ST._pages((3, r)) == (3, 2) and len(reads) == 2  # each page read once


def test_pages_stack_below(pairs):
    a = [["p", "a", [0, 0, 10, 50], 10.0, {}]]
    assert ST.stack(a, [["p", "b", [0, 0, 10, 5], 10.0, {}]])[1][2] == [0, 70, 10, 75]


def test_picked_page_not_gated_again(pairs, monkeypatch):
    _, pdf = pairs
    ST.gate_out({(pdf, 0)}, "text layer")  # layouts() records what clean() passed
    monkeypatch.setattr(ST, "pooled", lambda fn, jobs, w: iter(()) if not jobs else pytest.fail(f"gated again: {jobs}"))
    assert ST.regate(pd.DataFrame(dict(path=[pdf], page=[0], lang=["ru"], kind=["text"])), 2) == set()


def test_translation_not_made_again(pairs, monkeypatch):
    _, pdf = pairs
    ST._write(ST.trfile(dict(path=pdf, page=0, tl="hi")), dict(blocks=TEXT, tr=TEXT))
    monkeypatch.setattr(
        ST, "pooled", lambda fn, jobs, w: iter(()) if not jobs else pytest.fail(f"translated again: {jobs}")
    )
    ST.translate_texts([(1, pdf, 0, "ru", "hi")], 2)


def test_one_pool_per_size():
    ST._pools.clear()
    assert dict(ST.pooled(abs, {0: -1}, 2)) == {0: 1}
    ex = ST._pools[2]
    assert dict(ST.pooled(abs, {1: -2}, 2)) == {1: 2}
    assert ST._pools == {2: ex}
    ex.shutdown()
    ST._pools.clear()


def test_cached_text_as_the_reader_gives_it(pairs, monkeypatch):
    _, pdf = pairs
    table = ("t", "a b", (0, 0, 10, 10), 9.0, {"rows": [["a", "b"]]})
    monkeypatch.setattr(rare, "page_blocks", lambda *a: ([table], "text layer"))
    first, again = ST.page_text(pdf, 0, "ru"), ST.page_text(pdf, 0, "ru")  # made, then read from the cache
    assert first == again and type(again[0][0]) is tuple
    from dspages.multilang import gtranslate as GT

    monkeypatch.setattr(GT, "lines", lambda src, tl, sl="auto": [s.upper() for s in src])
    assert rare.translated(again[0], "hi", "ru")[0][4]["rows"] == [
        ["A", "B"]
    ]  # a table block from the cache translates


def test_translated_form_outlines_read_as_no_text(tmp_path):
    """A translated form keeps the source's outline pictures (read as no text there): marked once, its text layer reads (no LookupError)."""
    import shutil

    import pymupdf

    from dspages.layout import legacy

    src = Path(f"{_P.ROOT}/tests/pairs_steps/forms_tr/bg_cbf9f5a77353_zh_0.pdf")  # a colouring-contest form
    if not src.exists():
        pytest.skip("sample form not here")
    pdf = tmp_path / src.name
    shutil.copy(src, pdf)
    with pymupdf.open(pdf) as d, pytest.raises(LookupError):
        legacy.path_lines(d[0])
    legacy.mark_no_text(str(pdf))
    with pymupdf.open(pdf) as d:
        assert legacy.path_glyphs(d[0]) and legacy.path_lines(d[0]) == []


def test_form_page_read_once_for_its_target_languages(tmp_path, monkeypatch):
    import pymupdf
    import seed_texts

    from dspages.layout import legacy
    from dspages.multilang import gtranslate as GT
    from dspages.multilang import translate as TR

    pdf = tmp_path / "f_ru.pdf"
    with pymupdf.open() as d:
        d.new_page()
        d.save(pdf)
    reads = counter(monkeypatch, TR, "lines", lambda page, src: [dict(text="Имя", bbox=[0, 0, 9, 9], size=9)])
    monkeypatch.setattr(TR, "render", lambda *a, **k: None)
    monkeypatch.setattr(TR, "fonts_for", lambda *a: ["font"])
    monkeypatch.setattr(legacy, "path_lines", lambda page: [])
    monkeypatch.setattr(legacy, "mark_no_text", lambda pdf: None)
    monkeypatch.setattr(GT, "lines", lambda src, tl, sl="auto": [f"{tl}:{s}" for s in src])
    for tl in ("hi", "bg", "ka", "km", "am"):
        assert seed_texts.form((str(pdf), 0, tl, "ru", str(tmp_path)))
    assert len(reads) == 1


def test_targets_balanced_by_family_of_look():
    """Each target of another family than the source's; each family one share however many scripts and languages it has; in a family the script
    used least, in it the language used least."""
    pool = [
        "ru",
        "uk",
        "bg",
        "en",
        "el",
        "ko",
        "he",
        "am",
        "ka",
        "hy",
        "th",
        "lo",
        "hi",
        "bn",
        "pa",
        "ta",
        "te",
        "kn",
        "mni",
    ]
    L = pd.DataFrame(columns=["kind", "lang", "tl"])
    for _ in range(30):
        for tl in ST.targets("Devanagari", L, pool):
            L.loc[len(L)] = ["text", "hi", tl]
    f = L.tl.map(ST.script).map(ST.FAMILY_OF.get)
    assert f.notna().all() and "Devanagari-Bengali-Gurmukhi-Meetei" not in set(f) and "mni" not in set(L.tl)
    n = f.value_counts()
    assert n.max() - n.min() <= 1  # Cyrillic (ru uk bg) one share as Hangul (ko)
    cy = L[f == "Cyrillic"].tl.value_counts()
    assert cy.max() - cy.min() <= 1  # its languages in turn
    sa = L[f == "Thai-Lao-Khmer-Gujarati"].tl.map(ST.script).value_counts()
    assert set(sa.index) == {"Thai", "Lao"} and sa.max() - sa.min() <= 1  # its scripts in turn
