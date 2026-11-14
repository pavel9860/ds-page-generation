"""Decoding on real corpus pages, by case: the text the reader (dspages.hand.texts.page_lines) gives for a page must hold words checked by hand against the
page, and must not hold the garbage a wrong decode gives. Known defects are strict xfails (a fix turns them into failures to be moved up). Pages that
must not be decoded: they keep their typed codes, or the reader stops (LookupError), or the page reads as no text / garbled (hand.rare.page_blocks).
Skipped where the corpus is not mounted; the garbled cases ask Google Translate (dspages.ingest.lang.garbled)."""
import json
from pathlib import Path

import pymupdf
import pytest

L = Path("/run/media/p/D/ML_DS/UVTM/Layouts")
pytestmark = pytest.mark.skipif(not L.exists(), reason="corpus not mounted")


def read(rel, page, lang):
    from dspages.hand import texts as T
    with pymupdf.open(L / rel) as d:
        return " ".join(" ".join(l["text"] for l in T.page_lines(d[page], lang)).split())


# (file, page, lang, words the page shows, garbage a wrong decode gives)
DECODED = [
    ("archive/rejected/Rare_scripts/fill/bf6cc3cd9db6_km.pdf", 1, "km", ["ក្រមព្រហ្មទណ្ឌ"], ["្រក", "μ"]),                       # Limon: its own table (khm:56), the coeng ro after its consonant
    ("archive/rejected/Rare_scripts/fill/3f475f95b2d1_pa.pdf", 10, "pa", ["ਸ਼ੋਭਾ", "ਬਾਹਰੀ"], ["Fੋਭਾ", "-ੋਭਾ"]),                  # GurbaniLipi: Asees (the table the glyph shapes contradict least), its F from the shapes
    ("text/2c98d3b5895c_ta.pdf", 2, "ta", ["இதயத்தோடு", "போது"], ["இதயத்சூதாடு", "சூபாது"]),                                # Baamini: the Bamini codes (not Padma's Amudham)
    ("tables/9601ac53d343_ne.pdf", 0, "ne", ["प्रगति", "प्रतिवेदन"], ["पगति", "पतिवेदन"]),                                    # Nirmala UI: a conjunct glyph (pa+ra) its subset names no text keeps the map's whole cluster
    ("tables/9601ac53d343_ne.pdf", 0, "ne", ["दोस्रो चौमासिक", "मंगलसेन नगरपालिका"], []),                                # Preeti
    ("books_pdf/b2a02af519c5_ne.pdf", 7, "ne", ["बस्ती र घरपरिवार विवरण", "बेदकोट नगरपालिका"], []),                         # Kantipur
    ("archive/rejected/Rare_scripts/pictures/e3c80fee0674_hi.pdf", 0, "hi", ["गणतन्त्र दिवस समारोह", "विश्वविद्यालय"], []),   # Kruti Dev
    ("forms_clean/2e1adf369775_si.pdf", 0, "si", ["ශ්‍රී ලංකා", "සමාජවාදී ජනරජයේ"], []),                                # FM Abhaya
    ("text/c5c2d12abc83_kn.pdf", 8, "kn", ["ಕನಿಷ್ಕ", "ಕುಶಾಣರ ಮನೆತನವು"], []),                                             # Shree-Dev
    ("bureaucracy/b882e41b3b0d_hy.pdf", 0, "hy", ["Ոսկեհասկ համայնքը"], []),                                               # Arial Armenian
    ("books_pdf/1cc0098d1820_my.pdf", 0, "my", ["ကျောင်းသာ", "မန်းလေး"], []),                                             # Zawgyi (ICU)
    ("books_pdf/fbe9035a6db0_fa.pdf", 9, "fa", ["زمین کردن"], ["٠٠٠", "ګ؈دن"]),                                         # Unicode, presentation forms
    ("text/4d26be2bbb65_fa.pdf", 12, "fa", ["ادامه پیدا کرد", "صندوق قبور"], ["ٽڂات"]),                                   # Unicode, presentation forms
    ("text/dca6923b1c6e_km.pdf", 0, "km", ["មេរៀនទី", "ព្រះយេស៊ូវ", "មើល", "ហើយ"], ["មេរេៀន", "្ពះ", "ƛ", "Ƈ"]),        # Khmer OS: visual order, coeng ro, wrong maps (reference font)
    ("text/607b0e15a313_pa.pdf", 1, "pa", ["https://australiainstitute", "ਆਸਟ੍ਰੇਲੀਆ ਦੀ ਗੈਸ ਨੀਤੀ"], ["ਹਟਟਪਸ"]),            # ArialMT URL on a Punjabi page
    ("text/fa97d7de6695_bn.pdf", 0, "bn", ["জাদুবাস্তবতা"], ["ম্নজ্রল্বঙ্খ"]),                                            # system Times on a Bengali page
    ("books_pdf/9effcefb5c95_kk.pdf", 75, "kk", ["Елім нұрлы Қазақстан"], []),                                             # Unicode Cyrillic
]

DEFECTS = [   # decoded, but wrong: the words the page shows
    ("archive/rejected/Rare_scripts/fill/3f475f95b2d1_pa.pdf", 10, "pa", ["ਵਧਾਉਂਦਾ"], "GurbaniLipi: code I (Asees ਜ਼, the glyph shapes ੱ, the page ਂ)"),
]


@pytest.mark.parametrize("rel,page,lang,words,garbage", DECODED, ids=[c[0].split("/")[-1] for c in DECODED])
def test_page_decodes(rel, page, lang, words, garbage):
    from unicodedata import normalize
    t = normalize("NFC", read(rel, page, lang))                                 # compared in NFC (precomposed ਸ਼ and ਸ + nukta are one letter)
    assert all(normalize("NFC", w) in t for w in words), t[:300]
    assert not any(normalize("NFC", g) in t for g in garbage), t[:300]


@pytest.mark.parametrize("rel,page,lang,words,why", DEFECTS, ids=[c[4].split(":")[0] for c in DEFECTS])
def test_known_defect(rel, page, lang, words, why):
    """A defect still there: xfail. Its words read right: fixed, fails until the case moves to DECODED."""
    if not all(w in read(rel, page, lang) for w in words):
        pytest.xfail(why)
    pytest.fail(f"fixed ({why}): move the case to DECODED")


def test_latin_font_typing_armscii_is_not_decoded():
    """The page shows Latin-1 letters (its Times draws them; the document lost its Armenian font): the codes stay, no table makes Armenian of them."""
    t = read("tables/9cda07bb59f2_hy.pdf", 0, "hy")
    assert "²ÜÐ²î²Î²Ü" in t and "ԱՆՀԱՏԱԿԱՆ" not in t


def test_text_drawn_as_unread_outlines_stops_the_reader():
    with pytest.raises(LookupError):
        read("text/fecdaa77333b_sr.pdf", 0, "sr")
    with pytest.raises(LookupError):
        read("forms_clean/16e6dacca12b_ja.pdf", 0, "ja")


def test_scanned_page_has_no_text():
    from dspages.hand import rare
    assert rare.page_blocks(L / "forms_clean/58269bf59986_zh.pdf", 3, "zh")[1] == "no text"


MISSED = pytest.mark.xfail(strict=True, reason="garbled (Google Translate, word counts): Google transliterates the garbage, the word counts stay comparable")


@pytest.mark.parametrize("rel,page,lang", [pytest.param("forms_clean/77a22ccab8e9_mr.pdf", 10, "mr", id="surekh_mr", marks=MISSED),
                                           pytest.param("magazines/365bfafe8278_as.pdf", 0, "as", id="garbage_as", marks=MISSED),
                                           pytest.param("tables/9cda07bb59f2_hy.pdf", 0, "hy", id="armscii_in_times_hy", marks=MISSED)])
def test_garbled_page_is_rejected(rel, page, lang):
    from dspages.hand import rare
    assert rare.page_blocks(L / rel, page, lang)[1] == "garbled layer"


def test_decision_of_an_older_rule_is_made_again(tmp_path, monkeypatch):
    """A document's cached decode decisions made by older code are not reused: a 'latin' entry without the current rule (here a stale False that
    sent a system Times typing English through a Bengali table) is decided again."""
    import pickle
    from dspages.layout import legacy
    monkeypatch.setattr(legacy, "CACHE", str(tmp_path))
    (tmp_path / "docs").mkdir()
    stale = {"v": 2, "tables": {}, "table_for": {"TimesNewRomanPSMT": ("padma:aabpbengalix", "old")}, "latin": {"TimesNewRomanPSMT": False}}
    pickle.dump(stale, open(tmp_path / "docs" / "fa97d7de6695_bn.pkl", "wb"))
    t = read("text/fa97d7de6695_bn.pdf", 0, "bn")
    assert "Prantik Gabeshana Patrika" in t and "ম্নজ্রল্বঙ্খ" not in t


def test_baamini_is_bamini():
    """Baamini types the Bamini codes (on 41 of 42 corpus pages Padma's Amudham table turns its e-kar into சூ): both names decode alike."""
    from dspages.layout import legacy_tables as LT
    assert LT.route("Baamini") == LT.route("Bamini") and LT.decode("Baamini", ",jaj;NjhL") == LT.decode("Bamini", ",jaj;NjhL")


def test_close_tables_are_told_apart_by_the_language_model(tmp_path, monkeypatch):
    """Two tables a few codes apart both read 1.0 by GlotLID: the character trigram picks the right one (Bamini codes, not Padma's Amudham), with no
    alias to name it."""
    from dspages.layout import legacy, legacy_tables as LT
    monkeypatch.setattr(legacy, "CACHE", str(tmp_path))
    (tmp_path / "docs").mkdir()
    monkeypatch.setattr(LT, "_aliases", lambda: {k: v for k, v in json.load(open(LT.os.path.join(LT.TABLES, "aliases.json"))).items() if k != "baamini"})
    t = read("text/2c98d3b5895c_ta.pdf", 2, "ta")
    assert "இதயத்தோடு" in t and "சூபாது" not in t


@pytest.mark.parametrize("drawn, logical", [("្រកម្រពហ្មទណ្ឌ", "ក្រមព្រហ្មទណ្ឌ"), ("កម្រព", "កម្រព"), ("ក្រុម", "ក្រុម"), ("ខ្ញុំ ្រក", "ខ្ញុំ ក្រ")])
def test_khmer_coeng_ro_to_logical_order(drawn, logical):
    """A coeng ro at the start of a word is drawing order (logical order puts it after its consonant): moved; text in logical order stays."""
    from dspages.layout import indic
    assert indic.to_logical(drawn) == logical
