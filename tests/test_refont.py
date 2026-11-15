import numpy as np
import pymupdf
import pytest

from dspages.config import Paths
from dspages.layout import refont, template

WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon "
    "phi chi psi omega"
).split()


def text(n, seed):
    rng = np.random.default_rng(seed)
    return " ".join(rng.choice(WORDS, n))


def page(doc, left_words, right_words, head=True):
    """Two text columns with a heading across both right below them, then two more columns."""
    p = doc.new_page(width=595, height=842)
    if head:
        p.insert_textbox(
            pymupdf.Rect(50, 60, 545, 100), "A heading across both columns", fontsize=18, fontname="helv", align=1
        )
    ends = []
    for x0, n, seed in ((50, left_words // 2, 1), (310, right_words // 2, 2)):
        box = pymupdf.Rect(x0, 110, x0 + 235, 330)
        ends.append(box.y1 - p.insert_textbox(box, text(n, seed), fontsize=11, fontname="helv", align=3))
    y = max(ends) + 12
    p.insert_textbox(
        pymupdf.Rect(50, y, 545, y + 40),
        "A second heading that spans both columns",
        fontsize=16,
        fontname="helv",
        align=1,
    )
    p.insert_textbox(pymupdf.Rect(50, y + 50, 285, 800), text(left_words, 3), fontsize=11, fontname="helv", align=3)
    p.insert_textbox(pymupdf.Rect(310, y + 50, 545, 800), text(right_words, 4), fontsize=11, fontname="helv", align=3)
    return doc


def lay_out(doc, seed, fonts):
    src = refont.Source(doc)
    lay = template.layout(src, 0, np.random.default_rng(seed), fonts, 1024)
    return lay, refont.render(src, lay)


def blank_runs(img, x0, x1):
    inked = (np.asarray(img)[:, int(x0) : int(x1)] < 200).any(axis=1)
    ys = np.flatnonzero(inked)
    if not len(ys):
        return 0, 0
    runs, start = [], None
    for y in range(ys[0], ys[-1] + 1):
        if not inked[y]:
            start = y if start is None else start
        elif start is not None:
            runs.append(y - start)
            start = None
    return max(runs, default=0), int(ys[-1])


@pytest.fixture(scope="module")
def fonts():
    return refont.font_set(Paths().font_dirs)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_no_gap_in_columns_when_next_page_has_text(fonts, seed):
    doc = pymupdf.open()
    page(doc, 100, 90)
    page(doc, 110, 110, head=False)
    page(doc, 110, 110, head=False)
    lay, img = lay_out(doc, seed, fonts)
    line = lay["info"]["body_src_pt"] * lay["g"] * 1.3
    for c0, c1 in lay["cols"]:
        longest, last = blank_runs(img, c0, c1)
        assert longest <= 4 * line, f"column {c0:.0f}-{c1:.0f} has a {longest}px empty stretch (line {line:.0f}px)"
        bottom = max(r["y1"] for r in lay["rects"] if not r["pinned"])
        assert last == 0 or last >= bottom - 3 * line, (
            f"column {c0:.0f}-{c1:.0f} ends at {last}px, the text at {bottom:.0f}px"
        )


def test_columns_are_found_and_text_stays_in_them(fonts):
    doc = pymupdf.open()
    page(doc, 100, 90)
    lay, _ = lay_out(doc, 1, fonts)
    assert len(lay["cols"]) == 2
    for r in lay["rects"]:
        if not r["pinned"]:
            assert (
                any(c0 - 2 <= r["x0"] and r["x1"] <= c1 + 2 for c0, c1 in lay["cols"])
                or r["align"] == "center"
                or r["x1"] - r["x0"] > lay["cols"][0][1] - lay["cols"][0][0]
            )


def columns_with_a_short_first_line(doc, header=False):
    p = doc.new_page(width=595, height=842)
    if header:
        p.insert_textbox(pymupdf.Rect(200, 40, 400, 60), "RUNNING HEADER", fontsize=10, fontname="helv", align=1)
    p.insert_textbox(pymupdf.Rect(50, 110, 285, 420), text(120, 5), fontsize=11, fontname="helv", align=3)
    p.insert_textbox(pymupdf.Rect(310, 110, 545, 126), text(5, 6), fontsize=11, fontname="helv", align=3)
    p.insert_textbox(pymupdf.Rect(310, 136, 545, 420), text(105, 7), fontsize=11, fontname="helv", align=3)
    return doc


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_a_long_paragraph_does_not_hold_back_its_neighbour_column(fonts, seed):
    doc = pymupdf.open()
    columns_with_a_short_first_line(doc)
    lay, img = lay_out(doc, seed, fonts)
    line = lay["info"]["body_src_pt"] * lay["g"] * 1.3
    for c0, c1 in lay["cols"]:
        longest, _ = blank_runs(img, c0, c1)
        assert longest <= 4 * line, f"column {c0:.0f}-{c1:.0f} has a {longest}px empty stretch (line {line:.0f}px)"


def test_running_headers_of_the_next_pages_are_not_set_into_the_text(fonts):
    doc = pymupdf.open()
    for _ in range(3):
        columns_with_a_short_first_line(doc, header=True)
    lay, _ = lay_out(doc, 1, fonts)
    assert sum(1 for item in lay["items"] if item[3] == "RUNNING") == 1


def test_sheet_keeps_its_margin_and_is_reproducible(fonts, tmp_path):
    doc = pymupdf.open()
    page(doc, 100, 90)
    path = str(tmp_path / "p.pdf")
    doc.save(path)
    a, info = template.sheet(path, 0, 5, fonts, 0.1, (512, 724))
    b, _ = template.sheet(path, 0, 5, fonts, 0.1, (512, 724))
    assert a.shape == (724, 512) and (a == b).all()
    ink = a < 128
    assert not ink[:, :50].any() and not ink[:, -50:].any() and not ink[:72].any() and not ink[-72:].any() and ink.any()
    assert info["body_px"] >= info["body_src_pt"] * (512 - 2 * 51) / 595  # the source size, scaled with the page


def test_tiles_of_one_picture_are_joined_and_scattered_dots_are_not():
    grid = [(x, y, x + 10, y + 10) for x in range(0, 100, 10) for y in range(0, 50, 10)]
    dots = [(x, 0, x + 1, 1) for x in range(0, 100, 5)] + [(x, 99, x + 1, 100) for x in range(0, 100, 5)]
    assert [tuple(r) for r in refont.join_tiles(grid)] == [(0, 0, 100, 50)]
    assert len(refont.join_tiles(dots)) == len(dots)


def justified(doc, wide):
    """A justified paragraph whose first line has words set far apart (each word a line of its own when wide)."""
    p = doc.new_page(width=595, height=842)
    if wide:
        for x, w in ((50, "alpha"), (200, "beta"), (350, "gamma")):
            p.insert_text((x, 100), w, fontsize=11, fontname="helv")
    for k in range(6):
        p.insert_text((50, 114 + 14 * k), text(14, k), fontsize=11, fontname="helv")
    return doc


def test_words_set_far_apart_in_justified_text_stay_one_paragraph():
    doc = justified(pymupdf.open(), True)
    blocks = refont.page_blocks(refont.Source(doc), 0, True)
    assert len(blocks) == 1 and len(blocks[0]["lines"]) == 7


def test_a_list_marker_joins_its_text_and_numbered_items_are_paragraphs():
    doc = pymupdf.open()
    p = doc.new_page(width=595, height=842)
    for k in range(3):
        p.insert_text((50, 100 + 30 * k), f"{k + 1}.", fontsize=11, fontname="helv")
        p.insert_text((80, 100 + 30 * k), text(12, k), fontsize=11, fontname="helv")
        p.insert_text((50, 114 + 30 * k), text(12, k + 5) + ".", fontsize=11, fontname="helv")
    blocks = refont.page_blocks(refont.Source(doc), 0, True)
    assert [refont._text(b["lines"][0])[:2] for b in blocks] == ["1.", "2.", "3."] and all(
        len(b["lines"]) == 2 for b in blocks
    )


def test_a_label_with_a_field_on_its_line_is_not_flowing_text():
    doc = pymupdf.open()
    p = doc.new_page(width=595, height=842)
    for k in range(8):
        p.insert_text((50, 100 + 20 * k), f"Label number {k} of the form", fontsize=11, fontname="helv")
        p.draw_rect(pymupdf.Rect(250, 88 + 20 * k, 500, 102 + 20 * k))
    blocks = refont.page_blocks(refont.Source(doc), 0, True)
    assert all(l.get("tabbed") for b in blocks for l in b["lines"])


def test_a_line_does_not_join_a_paragraph_over_a_block_between_them():
    doc = pymupdf.open()
    p = doc.new_page(width=595, height=842)
    line = text(11, 1)
    p.insert_text((50, 100), line, fontsize=11, fontname="helv")
    p.insert_text((50, 114), "small note", fontsize=6, fontname="helv")
    p.insert_text((50, 128), line, fontsize=11, fontname="helv")
    blocks = refont.page_blocks(refont.Source(doc), 0, True)
    assert sorted(len(b["lines"]) for b in blocks) == [1, 1, 1]


def test_a_quarter_turned_page_has_the_turned_shape_and_its_text_fits(fonts):
    doc = pymupdf.open()
    page(doc, 100, 90)
    src = refont.Source(doc)
    lay = template.layout(src, 0, np.random.default_rng(1), fonts, 1024, 1.0, True)
    w, h = lay["size"]
    assert abs(w / h - 842 / 595) < 0.01
    body = [r for r in lay["rects"] if not r["pinned"] and r["y1"] - r["y0"] >= 4]
    assert all(r["x0"] >= -1 and r["x1"] <= w + 1 and r["y1"] <= h + 1 for r in lay["rects"])
    for i, a in enumerate(body):
        for b in body[i + 1 :]:
            overlap = (min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])) * (min(a["y1"], b["y1"]) - max(a["y0"], b["y0"]))
            assert not (
                min(a["x1"], b["x1"]) > max(a["x0"], b["x0"])
                and min(a["y1"], b["y1"]) > max(a["y0"], b["y0"])
                and overlap
                > 0.2 * min((a["x1"] - a["x0"]) * (a["y1"] - a["y0"]), (b["x1"] - b["x0"]) * (b["y1"] - b["y0"]))
            )


def test_turned_page_stretches_tables_and_keeps_the_shape_of_pictures(fonts):
    doc = pymupdf.open()
    p = doc.new_page(width=595, height=842)
    p.insert_textbox(pymupdf.Rect(50, 60, 545, 200), text(80, 1), fontsize=11, fontname="helv", align=3)
    pix = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 120, 60), False)
    pix.set_rect(pix.irect, (80,))
    p.insert_image(pymupdf.Rect(60, 300, 240, 390), pixmap=pix)
    for y in (450, 480, 510):
        p.draw_line((50, y), (545, y))
    p.draw_line((50, 450), (50, 510))
    p.draw_line((545, 450), (545, 510))
    lay = template.layout(refont.Source(doc), 0, np.random.default_rng(1), fonts, 1190, 1.0, True)
    ((x, t, pic),) = lay["pics"]
    assert abs(pic["rect"].width / pic["rect"].height - 2.0) < 0.02
    horizontal = [d["rect"].width for d, fixed, dy in lay["elems"] if d["rect"].height < 1]
    assert horizontal and all(abs(w - 495 * 842 / 595) < 1 for w in horizontal)


def test_a_tounicode_map_that_only_spells_out_winansi_is_no_text_map():
    cmap = b"begincmap\n2 beginbfrange\n<20><20><0020>\n<41><43><0041>\nendbfrange\n2 beginbfchar\n<80><20ac>\n<c7><00c7>\nendbfchar\nendcmap"
    assert refont._winansi_only(cmap)
    assert not refont._winansi_only(cmap.replace(b"<c7><00c7>", b"<c7><0995>"))
    assert not refont._winansi_only(b"begincmap\nendcmap")
