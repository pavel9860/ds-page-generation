import numpy as np
import pytest

pytest.importorskip("torch", reason="needs the corpus extra")

from dspages.acquire import urlmodel
from dspages.ingest import litefeat


def test_url_tokens_ignore_scheme_and_www():
    a = urlmodel.tokens("https://www.example.rs/docs/2025/form-12.pdf")
    assert a == urlmodel.tokens("http://example.rs/docs/2025/form-12.pdf")
    assert "tld:rs" in a and "h:example.rs" in a and "form" in a and "#4" in a


def test_encode_offsets_match_tokens():
    urls = ["https://a.com/x.pdf", "https://b.org/y/z.pdf"]
    ids, off = urlmodel.encode(urls)
    assert off.tolist() == [0, len(urlmodel.tokens(urls[0]))]
    assert len(ids) == sum(map(len, map(urlmodel.tokens, urls)))
    assert urlmodel.Net(dim=8)(ids, off).shape == (2, 16)


def test_farthest_first_takes_the_outlier_first():
    V = np.array([[0, 0], [0.1, 0], [5, 5]], np.float32)
    assert litefeat.farthest_first(V, [V[0]], 1) == [2]


def test_legacy_devanagari_rules():
    from dspages.layout import legacy

    assert legacy.logical("भ्ाारत", "Devanagari") == "भारत"  # half bha + stem + aa = bhaarat
    assert legacy.logical("काे", "Devanagari") == "को"  # aa + e = o


def test_published_legacy_tables():
    from dspages.layout import legacy_tables as L

    assert L.decode("Kruti Dev 010", "gSa") == "हैं"
    assert L.decode("Preeti", "s") == "क"
    assert L.decode("ML-TTRevathi", "DÕmlw") == "ഉത്സാഹം"
    import unicodedata

    assert unicodedata.normalize("NFC", L.decode("SutonnyMJ", "Avwg evsjvq Mvb MvB")) == unicodedata.normalize(
        "NFC", "আমি বাংলায় গান গাই"
    )
    assert L.decode("Calibri", "abc") is None


def test_aliased_and_new_tables():
    """Close variants of a published font (aliases.json) and the tables added from the converters of Punjabi, Georgian, Tibetan, Hebrew."""
    from dspages.layout import legacy_tables as L

    assert L.decode("DevLys 010", "fdlku") == "किसान"
    assert L.decode("Chatrik,Bold", "Kfilsqfn") == "ਖਾਲਿਸਤਾਨ"
    assert L.decode("LitNusx", "gamarjoba") == "გამარჯობა"
    assert L.decode("TCRCYoutso", "xÛ-¾ô-") == "ཕྱི་ལོ་"
    assert L.decode("RamatGan", "àáâ") == "אבג"


def test_labelled_table():
    from dspages.layout import legacy_tables as LT

    assert (
        LT.decode("Geetanjalilight", "\u00f8") == "\u09bf" and LT.label_source("Geetanjalilight", "\u00f8") == "label"
    )
    assert LT.label_source(
        "Geetanjalilight",
        "\u00ab",
    ) in (None, "label") or LT.label_source("Geetanjalilight", "\u00ab").startswith("guess")
    assert LT.decode("Brahma", "\u00ea") == "\u0c3e"
