"""Handwritten pages: structure, text helpers, plan balance, paper geometry, one page end to end (skipped without hwsynth fonts)."""

import numpy as np
import pytest

pytest.importorskip("hwsynth")
from dspages.hand import check, langs, page, paper, plan, structure, util


def blk(k, t, x0, y0, x1, y1, size=10.0):
    return (k, t, (x0, y0, x1, y1), size)


def test_structure_flow_gaps_and_heading_scale():
    items = structure.to_items(
        [
            blk("h", "Title", 0, 0, 600, 30, 16),
            blk("p", "a" * 200, 0, 35, 600, 120),
            blk("p", "b" * 200, 0, 190, 600, 260),
        ]
    )
    assert [i["k"] for i in items] == ["h", "p", "p"]
    assert items[0]["scale"] == 1.25 and items[0]["gap"] == 0
    assert items[1]["gap"] == 0 and items[2]["gap"] >= 1  # the blank lines of the source


def test_structure_columns_and_full_width_bands():
    left = [blk("p", "l" * 300, 0, 100, 290, 200), blk("p", "l" * 300, 0, 220, 290, 320)]
    right = [blk("p", "r" * 300, 310, 100, 600, 200), blk("p", "r" * 300, 310, 220, 600, 320)]
    items = structure.to_items(
        [blk("h", "Head", 0, 0, 600, 30, 16)] + left + right + [blk("p", "f" * 300, 0, 340, 600, 400)]
    )
    assert [i["k"] for i in items] == ["h", "cols", "p"]
    cols = items[1]["cols"]
    assert len(cols) == 2 and cols[0]["x"] == 0 and cols[1]["x"] > 0.4 and all(len(c["items"]) == 2 for c in cols)


def test_util_text_helpers():
    assert util.clean("a ......... b\n c") == "a b c"
    assert util.usable("abc 1") and not util.usable("...")
    assert util.trim("alpha beta gamma delta", 14) == "alpha beta"
    tree = [
        dict(k="h", t="H"),
        dict(k="cols", cols=[dict(items=[dict(k="p", t="x")])]),
        dict(k="table", rows=[["a", "b"]]),
    ]
    assert sorted(util.walk_text(tree)) == ["H", "a", "b", "x"]
    assert [i["t"] for i in util.map_text(tree[:2], str.lower)[1]["cols"][0]["items"]] == ["x"]


def test_char_scripts_shared_characters_take_the_script_of_their_word():
    got = list(util.char_scripts("قال، نعم 。 abc,"))
    assert ("،", "Arabic", True) in got and ("。", "Latin", True) in got and (",", "Latin", True) in got
    assert all(sc == "Arabic" for c, sc, _ in got if c in "قالنعم")


def test_arabic_and_cjk_punctuation_are_covered(has_fonts):
    import hwsynth as H

    w = H.sample_writer("print_script", "casual", seed=310)
    text = "قال، الطالب؛ نعم ـ 我们。「好」 ２０２４"
    w.font = lambda scr, _m=util.pick_fonts(w, text), _o=w.font: _m.get(scr) or _o(scr)
    assert util.uncovered(w, text) == 0 and util.strip_missing(w, text) == text


def test_kanji_of_a_japanese_page_take_its_kana_font(has_fonts):
    import hwsynth as H

    w = H.sample_writer("print_script", "casual", seed=310)
    text = "開発の資料を確認してください。農業と環境"
    m = util.pick_fonts(w, text)
    assert m.get("Han") and m["Han"] == m.get("Hiragana")


def test_traditional_chinese_takes_a_hand_font_of_any_script(has_fonts):
    import hwsynth as H

    w = H.sample_writer("print_script", "casual", seed=310)
    text = "並且亞洲倫理債務價格內則區域協會"
    w.font = lambda scr, _m=util.pick_fonts(w, text), _o=w.font: _m.get(scr) or _o(scr)
    assert util.uncovered(w, text) == 0


def test_est_lines_columns_take_the_longest():
    one = [dict(k="p", t="x" * 100)]
    two = [dict(k="cols", cols=[dict(w=0.5, items=one), dict(w=0.5, items=one)])]
    assert page.est_lines(two, 50) == pytest.approx(page.est_lines(one, 25))


def test_balanced_exact_shares():
    out = plan.balanced(["a", "b", "c"], [1, 1, 2], 100, np.random.default_rng(0))
    assert sorted((v, out.count(v)) for v in "abc") == [("a", 25), ("b", 25), ("c", 50)]


def test_cramers_v_independence():
    rng = np.random.default_rng(0)
    a = list(rng.integers(0, 4, 4000))
    assert check.cramers_v(a, list(rng.integers(0, 3, 4000))) < 0.05
    assert check.cramers_v(a, a) > 0.9


def test_languages_english_share():
    texts = [
        dict(lang=lg, script=s, kind="book")
        for lg, s, n in (
            ("en", "Latin", 500),
            ("ru", "Cyrillic", 500),
            ("ja", "Han", 500),
            ("hi", "Devanagari", 500),
            ("fr", "Latin", 500),
        )
        for _ in range(n)
    ]
    q = plan.languages(texts, 1000)
    assert q["en"] == 400 and q["ru"] == 180 and q["ja"] == 120 and q["hi"] == 180 and q["fr"] == 120


def test_script_ok():
    assert langs.script_ok("Hello world", "en") and not langs.script_ok("Hello world", "hi")
    assert langs.script_ok("नमस्ते दुनिया", "hi") and langs.script_ok("ひらがなと漢字", "ja")


@pytest.mark.parametrize("fmt", ["a4", "a5", "b5"])
def test_paper_geometry(fmt):
    paper.DRAW = False
    try:
        for kind in paper.KINDS:
            p = paper.paper(kind, np.random.default_rng(1), fmt)
            assert p.size[0] == paper.FORMATS[fmt] and p.left + p.right < p.size[0] and p.first < p.size[1]
            assert p.pitch is None or p.pitch >= 4
    finally:
        paper.DRAW = True


@pytest.fixture(scope="module")
def has_fonts():
    import hwsynth as H

    try:
        H.sample_writer("school_cursive", "casual", seed=1).font("Latin")
    except Exception:
        pytest.skip("no hwsynth fonts")


def sample_plan(**kw):
    return dict(
        dict(
            i=0,
            seed=5,
            wseed=3,
            persona="school_cursive",
            style="casual",
            pen="ballpoint",
            paper="lined",
            fmt="a4",
            dense=False,
            xh=2.4,
            xh_ratio=0.35,
            fill=0.9,
            dy=0.2,
        ),
        **kw,
    )


def test_page_end_to_end(has_fonts):
    items = [
        dict(k="h", t="Notes"),
        dict(k="p", t="The quick brown fox jumps over the lazy dog. " * 8),
        dict(k="li", t="1. First item"),
        dict(k="table", rows=[["Name", "Score"], ["Ann", "9"], ["Bob", "7"]]),
    ]
    img, meta = page.make(sample_plan(), items)
    assert img.shape == (1440, 1024, 3) and img.dtype == np.uint8
    assert (
        meta["words"] and meta["tallest_mm"] <= 15.0 and meta["xh_mm"] <= util.MAX_XH and meta["uncovered_chars"] == 0
    )
    assert all(0 <= b[0] <= b[2] <= 1024 for _, b, _ in meta["words"])


def test_page_columns_and_notebook_format(has_fonts):
    col = lambda t: dict(x=0, w=0.46, start=0, items=[dict(k="p", t=t, gap=0)])
    items = [dict(k="cols", cols=[dict(col("left " * 60), x=0), dict(col("right " * 60), x=0.54)], gap=0)]
    img, meta = page.make(sample_plan(fmt="a5", paper="squared"), items)
    xs = [(t, b) for t, b, _ in meta["words"]]
    assert (
        max(b[2] for t, b in xs if t == "left") < min(b[0] for t, b in xs if t == "right") + 5
    )  # the columns do not overlap


def test_structure_alignment_levels_and_gap_after_columns():
    items = structure.to_items(
        [
            blk("h", "Title", 200, 0, 400, 30, 16),
            blk("l", "Name Surname", 450, 40, 600, 52),
            blk("li", "- sub item", 40, 60, 600, 72) + (dict(lvl=2),),
            blk("p", "a" * 300, 0, 80, 600, 150),
        ]
    )
    assert items[0].get("align") == "center" and items[1].get("align") == "right" and items[2]["lvl"] == 2
    left = [blk("p", "l" * 300, 0, 100, 290, 200)]
    right = [blk("p", "r" * 300, 310, 100, 600, 200)]
    items = structure.to_items(
        [blk("p", "t" * 300, 0, 0, 600, 80)] + left + right + [blk("p", "f" * 300, 0, 205, 600, 260)]
    )
    assert items[1]["k"] == "cols" and items[2]["gap"] == 0  # the gap after a band is measured from its bottom


def test_src_gap_caps_blank_lines():
    assert [page.src_gap(g) for g in (0, 1, 2, 3, 5)] == [0, 1, 1, 2, 2]


def _pdf_page(lines, size=11, x0=72):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    pg = doc.new_page()
    y = 90
    for t in lines:
        if t is None:
            y += 2 * size
            continue
        pg.insert_text((x0, y), t, fontsize=size)
        y += size * 1.3
    return pg


def test_blocks_of_merges_wrapped_lines_and_keeps_short_lines():
    from dspages.hand import texts

    long = "word " * 16
    pg = _pdf_page(["John Smith", None, long, long, long, "end of it.", None, "- first item", "- second item"])
    kinds = [b[0] for b in texts.blocks_of(pg)]
    assert kinds.count("p") == 1 and "l" in kinds and kinds.count("li") == 2


def test_pagekind_features_poem_vs_prose():
    from dspages.hand import pagekind

    prose = pagekind.features(_pdf_page(["word " * 18] * 30))
    poem = pagekind.features(
        _pdf_page(
            sum((["a short line", "another and somewhat longer line of verse", "the end", None] for _ in range(6)), [])
        )
    )
    f = dict(zip(pagekind.NAMES, zip(prose, poem)))
    assert f["full"][0] > 0.8 > f["full"][1] and f["short"][1] > 0.5 and f["gap_big"][1] > 0.2 > f["gap_big"][0]
    assert pagekind.kind(_pdf_page(["word " * 18] * 30))[0] == "plain"
