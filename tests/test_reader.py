"""One page reader for every tool: legacy fonts are decoded wherever a page's text is read (dspages.hand.texts, hwdet.translate, hwdet render_form)."""
from pathlib import Path

import pymupdf
import pytest
from hwsynth.scripts import text_script

LEGACY = Path("/run/media/p/D/ML_DS/UVTM/Layouts/forms_clean/77a22ccab8e9_mr.pdf"), 10, "mr"     # Marathi set in a DVBW-TT legacy font
pytestmark = pytest.mark.skipif(not LEGACY[0].exists(), reason="corpus not mounted")


def deva(text):
    return sum(text_script(w) == "Devanagari" for w in text.split())


def test_page_lines_decode_legacy_fonts():
    from dspages.hand import texts as T
    with pymupdf.open(LEGACY[0]) as doc:
        raw = doc[LEGACY[1]].get_text()
        lines = T.page_lines(doc[LEGACY[1]], LEGACY[2])
    text = " ".join(l["text"] for l in lines)
    assert deva(raw) == 0 and deva(text) >= 8                      # the raw layer is Latin-1 codes; the reader gives Devanagari
    assert "D-0509" in text and "P.T.O." in text                   # header and footer kept


def test_words_carry_the_decoded_text():
    from dspages.hand import texts as T
    with pymupdf.open(LEGACY[0]) as doc:
        pg = doc[LEGACY[1]]
        words = T.page_words(pg, LEGACY[2])
        r = pg.rect
    assert deva(" ".join(w[4] for w in words)) >= 8
    assert all(0 <= w[0] <= w[2] <= r.width + 1 and 0 <= w[1] <= w[3] <= r.height + 1 for w in words)


def test_hwdet_reads_through_the_one_reader():
    from hwdet import translate as TR
    with pymupdf.open(LEGACY[0]) as doc:
        ls = TR.lines(doc[LEGACY[1]], LEGACY[2])
        ps = TR.paragraphs(doc[LEGACY[1]], lang=LEGACY[2])
    assert deva(" ".join(l["text"] for l in ls)) >= 8 and deva(" ".join(p["text"] for p in ps)) >= 8
    assert {"text", "bbox", "size", "bold", "role"} <= set(ps[0])


def test_form_labels_are_decoded():
    from dataclasses import replace

    from hwdet.config import Config
    from hwdet.pagegen import forms
    f = forms.render_form(str(LEGACY[0]), LEGACY[1], replace(Config(), long_side=1754), lang=LEGACY[2])
    assert f.spots and not any(c in s.get("label", "") for s in f.spots for c in "³ÖŸßµ¸ü")      # no legacy codes in a label


def test_no_tool_reads_raw_text_for_its_words():
    """The tools that read a page's words go through dspages.hand.texts (no get_text words / dict of their own)."""
    hw = Path(__file__).parents[2] / "handwriting_detection" / "src" / "hwdet"
    for f in (hw / "translate.py", hw / "pagegen" / "forms.py", Path(__file__).parents[1] / "tools" / "seed_texts.py"):
        s = f.read_text()
        assert 'get_text("words"' not in s and 'get_text("text"' not in s and "def _para(" not in s, f


def test_every_table_set_on_disk_is_a_decoder():
    """Khmer converter fonts (khm), glibc charsets (iconv) decode; a font of the Khmer database is routed by its name."""
    from dspages.layout import legacy_tables as LT
    d = LT.decoders()
    assert any(k.startswith("khm:") for k in d) and {f"iconv:{n}" for n in LT.ICONV} <= set(d)
    assert LT.route("Limon S1")[0] == "khm" and LT.run(*LT.route("Limon S1"), "Gñk").startswith("អ្នក")
    assert LT.run("iconv", "ARMSCII-8", "²ÜÐ²î²Î²Ü") == "ԱՆՀԱՏԱԿԱՆ"


def test_cached_choices_follow_the_tables():
    """A table choice cached among another set of the script's tables is decided again; one of another script's tables is not touched."""
    from dspages.layout import legacy as L
    assert L.tables_sig("Devanagari") != L.tables_sig("Khmer")
    assert L.tables_sig("Devanagari") == L.tables_sig("Devanagari")


def test_aliases_and_zawgyi_decode():
    """Every alias reaches a table run() can read; a Zawgyi font is routed by name to ICU's Zawgyi-my transform."""
    from dspages.layout import legacy_tables as LT
    for k in LT._aliases():
        LT.run(*LT.route(k), "abc")
    assert LT.route("Zawgyi-One") == ("icu", "Zawgyi-my") and LT.run("icu", "Zawgyi-my", "ျမန္မာ") == "မြန်မာ"
