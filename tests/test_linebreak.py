"""Line breaks through libicu: spaces, the text end, and dictionary breaks of a script without spaces."""

import glob

import pytest

if not glob.glob("/usr/lib/*/libicuuc.so.[0-9]*") + glob.glob("/usr/lib/libicuuc.so.[0-9]*"):
    pytest.skip("libicu not installed", allow_module_level=True)

from dspages.layout.linebreak import breaks


def test_breaks_after_spaces_and_at_the_end():
    b = breaks("hello world again")
    assert 6 in b and 12 in b and b[-1] == len("hello world again")


def test_thai_breaks_inside_a_run_without_spaces():
    t = "สวัสดีครับวันนี้อากาศดี"
    b = breaks(t)
    assert b[-1] == len(t) and len(b) >= 3


def test_offsets_count_characters_beyond_the_bmp():
    t = "𝐀𝐁 cd"
    assert breaks(t)[-1] == len(t) and 3 in breaks(t)
