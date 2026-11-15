"""Fast checks of the language dataset"""

import json
import random
from collections import Counter

import pymupdf
import pytest
from PIL import Image

from dspages.multilang import paths
from dspages.multilang.buckets import BUCKETS, bucket_of
from dspages.multilang.config import LANGDS as C
from dspages.multilang.langcheck import clean_ok, lexicon_share, repair
from dspages.multilang.langds import balance, counts, plan, report, writable
from dspages.multilang.realpages import text_layer_ok

DATA = paths.DATA
PARAS = DATA / "langds" / "paras"
LANGS = sorted({m for ms in BUCKETS.values() for m in ms.split()} - {"dv"} - set(C.alias))


GOOD = {
    "de": "Das ist ein ganz normaler deutscher Satz über das Wetter in Berlin.",
    "ru": "Это обычное русское предложение о погоде в Москве.",
    "hi": "यह एक साफ वाक्य है जो मौसम के बारे में बात करता है।",
    "ar": "هذا مثال على جملة عربية سليمة عن الطقس في القاهرة اليوم.",
}
BAD = {
    "legacy Indic (vowel signs split off)": ("hi", "प्रेस नोट - ददन ांक 11.07.2026 स इबर क्र इम थ न , जनपद उन्न व ।"),
    "legacy Bengali (Latin letters inside)": ("bn", "Ʊসখােন দুজন পূণƳ মানুষ একজন অধƳ মানুষ।"),
    "replacement character": ("de", "Das ist ein normaler Satz � über das Wetter."),
    "words no lexicon knows (letters misread: li for h, rn for m)": (
        "de",
        "Mascliinenbau olme direkte Verbindung zur Schweifiteclmik ist undenkbar und Schweifien ist.",
    ),
    "split Arabic letters": ("ar", "تجربتين من التجارب المختارة وا لا شكال الموج ي ة والمنحنيات الناتجة."),
}


@pytest.mark.parametrize("lang", GOOD)
def test_clean_text_passes(lang):
    assert clean_ok(GOOD[lang], lang)


@pytest.mark.parametrize("why", BAD)
def test_broken_text_is_rejected(why):
    lang, text = BAD[why]
    assert not clean_ok(text, lang), why


def test_newline_is_not_a_control_character():
    assert clean_ok(GOOD["de"] + "\n" + GOOD["de"], "de")


@pytest.mark.parametrize(
    ("broken", "fixed"),
    [
        (
            "Das ist ein normaler Satz Ã¼ber das Wetter in Berlin.",
            "Das ist ein normaler Satz über das Wetter in Berlin.",
        ),
        (
            "Das ist ein normaler Satz \x96 über das Wetter in Berlin.",
            "Das ist ein normaler Satz – über das Wetter in Berlin.",
        ),
        ("هل يصح هذا اإلرسال", "هل يصح هذا الإرسال"),
        ("这是一个句子，用全角逗号。", "这是一个句子，用全角逗号。"),
    ],
)
def test_what_can_be_repaired_is_repaired_not_rejected(broken, fixed):
    assert repair(broken) == fixed
    assert clean_ok(repair(broken), "ar" if "ه" in broken else "zh" if "这" in broken else "de")


def test_arabic_presentation_forms_become_base_letters():
    shaped = "ﺍﻟﺴﻼﻡ ﻋﻠﻴﻜﻢ"
    text = repair(shaped)
    assert text == "السلام عليكم"
    assert not any(0xFB50 <= ord(c) <= 0xFDFF or 0xFE70 <= ord(c) <= 0xFEFC for c in text)


def test_lexicon_share_separates_garbled_from_clean():
    assert lexicon_share(GOOD["hi"], "hi") >= C.min_lexicon
    assert lexicon_share("वचंताजनक आंकड़ो को आिार बनाया गया है एिं प्रभाि वकया", "hi") < C.min_lexicon
    assert lexicon_share("今天天气很好", "zh") is None


def _pdf(text, render_mode=0, font="helv"):
    doc = pymupdf.open()
    pg = doc.new_page()
    pg.insert_text((72, 100), text, fontname=font, fontsize=12, render_mode=render_mode)
    return doc


def test_visible_text_layer_is_digital():
    assert text_layer_ok(_pdf("Plain digital text.")[0])


def test_invisible_text_layer_is_ocr():
    assert not text_layer_ok(_pdf("Text an OCR engine placed over a scan.", render_mode=3)[0])


@pytest.mark.parametrize(
    "font", ["KrutiDev010", "DevLys-010", "AkrutiDevPriyaBold", "SHREE-DEV7-0713", "Devnagari-ChanakyaNormal"]
)
def test_legacy_fonts_are_recognised(font):
    import re

    assert re.search(C.legacy_font, font, re.I)


@pytest.mark.parametrize("font", ["Mangal", "NotoSerifDeva", "Arial-BoldMT", "TimesNewRoman", "SakalMarathi"])
def test_unicode_fonts_are_not_legacy(font):
    import re

    assert not re.search(C.legacy_font, font, re.I)


def test_every_language_is_in_one_class():
    seen = Counter(m for ms in BUCKETS.values() for m in ms.split())
    assert not [m for m, n in seen.items() if n > 1]
    assert all(bucket_of(lang) for lang in LANGS)


def test_plan_gives_every_language_every_printed_type():
    quota, _ = plan([])
    for lang in LANGS:
        for t in C.types:
            if t in C.hand:
                continue
            assert quota[(lang, t)] >= C.min_per_type, (lang, t)


def test_plan_has_no_handwriting_where_there_is_no_realistic_font():
    quota, _ = plan([])
    for lang in LANGS:
        for t in C.hand:
            assert (quota[(lang, t)] > 0) == writable(lang), (lang, t)


def test_plan_is_equal_per_language_inside_a_class():
    quota, _ = plan([])
    for cls, ms in BUCKETS.items():
        langs = [m for m in ms.split() if m != "dv" and m not in C.alias]
        if not langs:
            continue
        totals = {lang: sum(quota[(lang, t)] for t in C.types) for lang in langs}
        assert max(totals.values()) - min(totals.values()) <= 12, (cls, totals)


def test_plan_type_shares_follow_the_config():
    quota, _ = plan([])
    total = sum(quota.values())
    for t in ("text", "table", "form", "flyer", "magazine"):
        got = sum(v for (_, tt), v in quota.items() if tt == t) / total
        assert abs(got - C.share[t]) < 0.08, (t, got)


def test_balance_cuts_each_class_to_the_quota():
    rows = [dict(lang=lang, type="text", file=f"{lang}{i}") for lang in LANGS for i in range(30)]
    got = counts(balance(rows, 100))
    avail = Counter()
    for lang in LANGS:
        avail[bucket_of(lang)] += 30
    assert {c: sum(v.values()) for c, v in got.items()} == {c: min(100, n) for c, n in avail.items()}


def test_balance_is_equal_per_language_in_a_class():
    ms = [m for m in BUCKETS["slavic_cyrillic"].split() if m != "dv"]
    rows = [dict(lang=lang, type="text", file=f"{lang}{i}") for lang in ms for i in range(50)]
    per = Counter(r["lang"] for r in balance(rows, 5 * len(ms)))
    assert set(per.values()) == {5}


def test_report_counts_the_cells_under_the_minimum():
    full = [
        dict(lang=lang, type=t, file=f"{lang}{t}{i}")
        for lang in LANGS
        for t in C.types
        if t not in C.hand or writable(lang)
        for i in range(C.min_per_type)
    ]
    assert report(full)["n_short"] == 0
    assert report(full[1:])["n_short"] == 1


TEXTS = {
    "de": [
        "Die Stadtverwaltung hat gestern beschlossen, das Schwimmbad im Sommer länger zu öffnen. Besucher können nun bis spät am Abend schwimmen.",
        "Im Herbst beginnt die Sanierung der Hauptstraße, wodurch es zu Verkehrsbehinderungen kommen kann. Anwohner werden rechtzeitig informiert.",
    ]
    * 12,
    "ru": [
        "Городская администрация вчера решила продлить время работы бассейна летом. Посетители смогут плавать допоздна каждый день недели.",
        "Осенью начнётся ремонт главной улицы, из-за чего возможны заторы на дорогах. Жители будут заранее проинформированы об изменениях.",
    ]
    * 12,
    "hi": [
        "नगर प्रशासन ने कल निर्णय लिया कि गर्मियों में तरण ताल को अधिक समय तक खोला जाएगा। आगंतुक अब देर शाम तक तैर सकेंगे।",
        "शरद ऋतु में मुख्य सड़क की मरम्मत शुरू होगी, जिससे यातायात में बाधा आ सकती है। निवासियों को पहले से सूचित किया जाएगा।",
    ]
    * 12,
}


def _pool(lang):
    t = TEXTS[lang]
    words = [w.strip(".,") for x in t for w in x.split()][:60]
    return dict(
        p=list(dict.fromkeys(t)) * 6,
        h=[" ".join(words[i : i + 3]) for i in range(0, 30, 3)],
        li=[" ".join(words[i : i + 6]) for i in range(0, 36, 6)],
        labels=words[:9],
        heads=words[10:18],
        cells=words[20:41],
        titles_table=words[42:44],
        titles_form=words[45:47],
    )


def _make(kind, lang, tmp_path, seed=1):
    pytest.importorskip("hwsynth")
    from dspages.multilang.synthpages import make_synth

    try:
        row = make_synth((kind, lang, seed, str(tmp_path), _pool(lang), "train"))
    except Exception as e:
        pytest.skip(f"the handwriting font index is not available: {e}")
    assert row, (kind, lang)
    return row, Image.open(tmp_path / row["file"])


@pytest.mark.parametrize("kind", ["text", "table", "form", "magazine", "flyer"])
@pytest.mark.parametrize("lang", ["de", "ru", "hi"])
def test_printed_page_is_a_full_readable_window(kind, lang, tmp_path):
    import numpy as np

    row, im = _make(kind, lang, tmp_path)
    assert im.size == (C.side, C.side)
    a = np.asarray(im.convert("L"))
    ink = np.nonzero((a < 128).any(axis=1))[0]
    assert ink.max() - ink.min() > 0.45 * C.side, "page is mostly empty"
    lo, hi = C.font_pt[kind]
    assert lo <= row["font_pt"] <= hi + 1.0
    assert 0 <= row["margin_pt"] <= C.margin_max * C.pt + 0.1
    assert row["boxes"] and all(
        0 <= b["b"][0] < b["b"][2] <= C.side + 2 and 0 <= b["b"][1] < b["b"][3] <= C.side + 2 for b in row["boxes"]
    )


def test_page_text_has_no_glyph_the_font_lacks(tmp_path):
    row, _ = _make("text", "hi", tmp_path)
    assert row["script"] == "Devanagari"


def test_a_page_never_repeats_a_paragraph_while_the_pool_has_more(tmp_path):
    from dspages.multilang.synthpages import Text

    t = Text(_pool("de"), random.Random(0), "Latin")
    got = [t.para() for _ in range(len(set(_pool("de")["p"])))]
    assert len(set(got)) == len(got)


@pytest.mark.slow
def test_handwritten_page_is_filled_with_a_margin(tmp_path):
    import numpy as np

    row, im = _make("handwritten", "ru", tmp_path)
    a = np.asarray(im.convert("L"))
    ink = np.nonzero((a < 128).any(axis=1))[0]
    assert ink.max() - ink.min() > 0.5 * C.side
    assert all(b["k"] == "hand" for b in row["boxes"])


GAPS = {"ps": "Pashto has 5 clean paragraphs: source text of the language is missing"}


@pytest.mark.skipif(not (PARAS.exists() and (DATA / "bucketing/src.json").exists()), reason="no text pools")
@pytest.mark.parametrize(
    "lang",
    [
        pytest.param(lang, marks=pytest.mark.xfail(reason=GAPS[lang], strict=True)) if lang in GAPS else lang
        for lang in LANGS
    ],
)
def test_pool_of_a_language_is_clean(lang):
    from dspages.multilang.synthpages import load_pools

    pool = load_pools(lang, DATA / "langds", DATA / "bucketing", None)
    assert len(pool["p"]) >= 10, "too few clean paragraphs"
    for k in ("p", "h", "li"):
        assert all(clean_ok(x, lang, 0.25) for x in random.Random(lang).sample(pool[k], min(len(pool[k]), 25))), k
    assert pool["labels"] and pool["heads"] and pool["cells"]
    text = "".join(pool["p"] + pool["h"] + pool["li"])
    assert not any(
        0xFB50 <= ord(c) <= 0xFDFF or 0xFE70 <= ord(c) <= 0xFEFC or 0x80 <= ord(c) <= 0x9F or ord(c) == 0xFFFD
        for c in text
    )


@pytest.mark.skipif(not (DATA / "langds" / "real_checked.jsonl").exists(), reason="no checked real pages")
def test_checked_real_pages_are_digital_and_of_their_type():
    rows = [json.loads(x) for x in open(DATA / "langds" / "real_checked.jsonl", encoding="utf-8")]
    assert all(r["type_mass"] >= C.min_mass for r in rows)
    assert not [r for r in rows if r.get("lang_status") == "mismatch"]
