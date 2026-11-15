"""pagegen: a handwritten page from a fixed seed, line metrics of form handwriting."""

import random

import pytest

pytest.importorskip("pymupdf")

from dspages.pagegen.config import PAGE, GenCfg
from dspages.pagegen.hand import line_pitch, writing_height
from dspages.pagegen.layouts import LAYOUTS
from dspages.pagegen.page import handwritten_page

POOL = {
    "heading": ["Meeting notes", "Next steps"],
    "para": [
        "The committee agreed to move the review to the second week of March and to send the draft to all members."
    ]
    * 6,
    "item": ["send the draft", "collect the figures", "book the room"] * 3,
}


def test_layouts_registered():
    assert "plain" in LAYOUTS and len(LAYOUTS) >= 3


def lines(pitch, n=20):
    return [(10, 100 + i * pitch, 200, 100 + i * pitch + 0.6 * pitch) for i in range(n)]


def test_height_follows_the_pitch_and_the_page_limits():
    H = 1000
    for pitch in (50, 70, 90):
        got = [writing_height(lines(pitch), H, random.Random(s)) for s in range(50)]
        assert min(got) >= pitch * PAGE.pitch_share[0] - 1e-6 and max(got) <= pitch * PAGE.pitch_share[1] + 1e-6
    tiny = [writing_height(lines(5), H, random.Random(s)) for s in range(50)]
    assert min(tiny) >= PAGE.min_hw * H and max(tiny) <= PAGE.min_hw * H * (1 + PAGE.min_jitter)
    assert max(writing_height(lines(300), H, random.Random(s)) for s in range(20)) == PAGE.max_hw * H


def test_pitch_and_default():
    assert abs(line_pitch(lines(20)) - 20) < 1e-6 and line_pitch([]) == 14.0


@pytest.mark.slow
def test_handwritten_page_deterministic():
    pages = [handwritten_page("plain", POOL, GenCfg(), random.Random(s)) for s in (1, 1)]
    if pages[0] is None:
        pytest.skip("seed rejected")
    assert pages[0].img.ndim == 3 and pages[0].words > 0
    assert (pages[0].img == pages[1].img).all()
