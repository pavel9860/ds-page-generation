"""PDF reading and legacy decoding: the rules found on real pages (Priyaanka Telugu, runs over several lines, ornaments, short column groups, polytonic
Greek, conjuncts named by a reference font, a table found by its text). Page cases by font: tests/test_decode_cases.py. Corpus tests are skipped where
the corpus is not mounted."""
import json
from pathlib import Path

import pytest

from dspages.layout import legacy_tables as LT

TE = Path("/run/media/p/D/ML_DS/UVTM/Layouts/text/8759dd7000d8_te.pdf")
corpus = pytest.mark.skipif(not TE.exists(), reason="corpus not mounted")


def test_priya_names_from_the_full_glyph_list():
    assert LT._priya_decode("u≤ãTμμ") == "బాబు’’"                       # Greek mu reported for the micro sign: the table's 'mu'
    assert "�" not in LT._priya_decode("|üfiTfl")                    # fi / fl ligatures: names only the full AGL list has


def test_raw_types_nothing_for_unmapped_glyphs():
    assert LT.raw([0xFFFD, ord("a")]) == "a"


def test_short_column_groups_join_a_column():
    from dspages.hand.structure import to_items
    body = "x " * 60
    bl = [("p", body, (0, 100, 200, 300), 10.0), ("p", body, (300, 100, 500, 300), 10.0), ("h", "Title", (420, 20, 480, 40), 14.0),
          ("l", "page 61", (330, 320, 400, 330), 8.0)]
    text = json.dumps(to_items(bl), ensure_ascii=False)
    assert "Title" in text and "page 61" in text


@corpus
def test_priyaanka_page_reads_clean_in_paragraphs():
    from dspages.hand import rare
    bl, how = rare.page_blocks(TE, 60, "te")
    text = " ".join(b[1] for b in bl)
    assert how == "text layer", how
    assert "�" not in text and "రాచపూటి రమేష్" in text and "బాబు’’" in text
    assert sum(b[0] == "p" for b in bl) >= 8                              # the body in paragraphs, not one line block


def test_bijoy_reph_and_windows_characters():
    assert LT.raw([0x2021]) == "‡" and LT.raw([0x87]) == "‡"                                  # a cp1252 character and its C1 byte: the table's code


def test_arabic_letters_and_lam_alef_with_hamza():
    from dspages.hand.texts import _word
    assert _word("الشھر", "ar") == "الشهر"
    assert _word("األولى", "ar") == "الأولى"


def test_bamini_reads_by_its_own_table():
    assert LT.decode("Bamini", "Nghu;j;Jf;Nfau;fs; aho;g;ghdj;ij") == "போர்த்துக்கேயர்கள் யாழ்ப்பானத்தை"


@corpus
def test_polytonic_greek_letters_are_not_code_page_bytes():
    from dspages.hand import rare
    bl, _ = rare.page_blocks(Path("/run/media/p/D/ML_DS/UVTM/Layouts/untyped_tmp/edadc5e6a4b4_el.pdf"), 1, "el")
    text = " ".join(b[1] for b in bl)
    assert text.startswith("Ὑπὲρ τοῦ ῥυσθῆναι ἡμᾶς ἀπὸ") and "Ή" not in text


@corpus
def test_reference_font_names_the_conjuncts_of_a_wrong_map():
    from dspages.layout import reffonts
    if "nikosh" not in reffonts.references():
        pytest.skip("no Nikosh reference font")
    from dspages.hand import rare
    bl, how = rare.page_blocks(Path("/run/media/p/D/ML_DS/UVTM/Layouts/forms_clean/9b2f223eb577_bn.pdf"), 3, "bn")
    text = " ".join(b[1] for b in bl)
    assert how == "text layer" and all(w in text for w in ("ব্যবসা", "প্রকাশ্য", "বিক্রি", "সংগ্রহ", "হস্তান্তর", "প্রমাণাদি", "প্রদানের"))


@corpus
def test_table_found_by_the_text_it_gives():
    from dspages.hand import rare
    bl, how = rare.page_blocks(Path("/run/media/p/D/ML_DS/UVTM/Layouts/untyped_tmp/f3736ddd9105_mk.pdf"), 0, "mk")      # MACCTimes: no table by its name
    assert how == "text layer", how
