import numpy as np
import pymupdf
import pytest

from dspages.config import Paths
from dspages.layout import refont

WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon "
         "phi chi psi omega").split()


def text(n, seed):
    rng = np.random.default_rng(seed)
    return " ".join(rng.choice(WORDS, n))


def page(doc, left_words, right_words, head=True):
    """Two text columns with a heading across both right below them, then two more columns."""
    p = doc.new_page(width=595, height=842)
    if head:
        p.insert_textbox(pymupdf.Rect(50, 60, 545, 100), "A heading across both columns", fontsize=18, fontname="helv", align=1)
    ends = []
    for x0, n, seed in ((50, left_words // 2, 1), (310, right_words // 2, 2)):
        box = pymupdf.Rect(x0, 110, x0 + 235, 330)
        ends.append(box.y1 - p.insert_textbox(box, text(n, seed), fontsize=11, fontname="helv", align=3))
    y = max(ends) + 12
    p.insert_textbox(pymupdf.Rect(50, y, 545, y + 40), "A second heading that spans both columns", fontsize=16, fontname="helv", align=1)
    p.insert_textbox(pymupdf.Rect(50, y + 50, 285, 800), text(left_words, 3), fontsize=11, fontname="helv", align=3)
    p.insert_textbox(pymupdf.Rect(310, y + 50, 545, 800), text(right_words, 4), fontsize=11, fontname="helv", align=3)
    return doc


def blank_runs(img, x0, x1):
    inked = (np.asarray(img)[:, int(x0):int(x1)] < 200).any(axis=1)
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
    lay = refont.layout(doc, 0, np.random.default_rng(seed), fonts, 11.0, 1.0, 1024)
    img = refont.render(doc, lay)
    line = lay["info"]["body_pt"] * lay["size"][0] / refont.C.sheet_pt_w * 1.3
    for c0, c1 in lay["cols"]:
        longest, last = blank_runs(img, c0, c1)
        assert longest <= 4 * line, f"column {c0:.0f}-{c1:.0f} has a {longest}px empty stretch (line {line:.0f}px)"
        assert last == 0 or last >= lay["limit"] - 3 * line or c0 != lay["cols"][0][0], f"column {c0:.0f}-{c1:.0f} ends at {last}px, text area ends at {lay['limit']:.0f}px"


def test_columns_are_found_and_text_stays_in_them(fonts):
    doc = pymupdf.open()
    page(doc, 100, 90)
    lay = refont.layout(doc, 0, np.random.default_rng(1), fonts, 11.0, 1.0, 1024)
    assert len(lay["cols"]) == 2
    for r in lay["rects"]:
        if not r["pinned"]:
            assert any(c0 - 2 <= r["x0"] and r["x1"] <= c1 + 2 for c0, c1 in lay["cols"]) or r["align"] == "center" or r["x1"] - r["x0"] > lay["cols"][0][1] - lay["cols"][0][0]


def anchor(x0, x1, oy0, oy1, ny0, ny1):
    return dict(x0=x0, x1=x1, oy0=oy0, oy1=oy1, ny0=ny0, ny1=ny1, pinned=False, n=1)


def test_column_gap_is_found_when_a_block_waited_for_its_neighbours():
    cols = [(0, 100), (110, 210)]
    done = [anchor(0, 100, 0, 50, 0, 50), anchor(110, 210, 0, 50, 0, 200), anchor(0, 210, 60, 80, 210, 230)]
    above, below, col, size = refont.column_gap(done, cols, pitch=20, sc=1.0)
    assert col == (0, 100) and size == 160 and below["ny0"] == 210


def test_column_gap_ignores_a_gap_the_source_has():
    cols = [(0, 100)]
    done = [anchor(0, 100, 0, 50, 0, 50), anchor(0, 100, 250, 300, 250, 300)]
    assert refont.column_gap(done, cols, pitch=20, sc=1.0) is None


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
    lay = refont.layout(doc, 0, np.random.default_rng(seed), fonts, 11.0, 1.0, 1024)
    img = refont.render(doc, lay)
    line = lay["info"]["body_pt"] * lay["size"][0] / refont.C.sheet_pt_w * 1.3
    for c0, c1 in lay["cols"]:
        longest, _ = blank_runs(img, c0, c1)
        assert longest <= 4 * line, f"column {c0:.0f}-{c1:.0f} has a {longest}px empty stretch (line {line:.0f}px)"


def test_running_headers_of_the_next_pages_are_not_set_into_the_text(fonts):
    doc = pymupdf.open()
    for _ in range(3):
        columns_with_a_short_first_line(doc, header=True)
    lay = refont.layout(doc, 0, np.random.default_rng(1), fonts, 11.0, 1.0, 1024)
    assert sum(1 for item in lay["items"] if item[3] == "RUNNING") == 1


def test_sheet_keeps_its_margin_and_is_reproducible(fonts, tmp_path):
    doc = pymupdf.open()
    page(doc, 100, 90)
    path = str(tmp_path / "p.pdf")
    doc.save(path)
    a, info = refont.sheet(path, 0, 5, fonts, 12.0, 1.0, 0.1, (512, 724))
    b, _ = refont.sheet(path, 0, 5, fonts, 12.0, 1.0, 0.1, (512, 724))
    assert a.shape == (724, 512) and (a == b).all()
    ink = a < 128
    assert not ink[:, :50].any() and not ink[:, -50:].any() and not ink[:72].any() and not ink[-72:].any() and ink.any()
    assert 11.0 <= info["body_pt"] <= 12.0
