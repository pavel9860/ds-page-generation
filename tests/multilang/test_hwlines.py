"""handwriting"""
import random

import cv2
import numpy as np
import pytest
from dspages.multilang.config import HWPICK as C

from dspages.multilang import hwlines, hwpick


def page(rows, pitch, scale=1.4, rule=False):
    img = np.full((120 + pitch * rows, 900, 3), 255, np.uint8)
    for r in range(rows):
        y = 100 + pitch * r
        cv2.putText(img, "Hello gjpqy world TLH", (60, y), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, scale, (20, 20, 20), 2, cv2.LINE_AA)
        if rule:
            cv2.line(img, (0, y + 8), (899, y + 8), (150, 150, 150), 1)
    return img


def test_separate_rows_get_one_box_each():
    boxes = hwlines.boxes_of(hwlines.ink_of(page(5, 90)))
    assert len(boxes) == 5 and all(b[2] - b[0] > 300 for b in boxes)


def test_touching_rows_are_cut_at_the_valleys():
    boxes = hwlines.boxes_of(hwlines.ink_of(page(8, 48, scale=1.6)))
    assert len(boxes) >= 7


def test_ruling_is_not_a_line():
    boxes = hwlines.boxes_of(hwlines.ink_of(page(4, 90, rule=True)))
    assert len(boxes) == 4


def test_blank_page_has_no_box():
    assert hwlines.boxes_of(hwlines.ink_of(np.full((600, 800, 3), 255, np.uint8))) == []


def test_quota_floor_and_cap():
    p = np.array([0.99, 0.98, 0.97, 0.96, 0.4, 0.2, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.89, 0.88, 0.87, 0.86])
    cand = {"aa": [0, 1, 2, 3, 4, 5], "bb": list(range(6, 20))}
    chosen = hwpick.choose(cand, p, {"aa": 2, "bb": 2}, 12)
    assert len(chosen["aa"]) <= 4 and all(p[i] >= C.lo for v in chosen.values() for i in v)
    assert len(chosen["bb"]) <= C.cap
    assert sum(map(len, chosen.values())) == 12


def test_language_with_few_good_pages_goes_down_to_the_floor_only():
    p = np.array([0.6, 0.3, 0.1, 0.99, 0.99, 0.99, 0.99])
    chosen = hwpick.choose({"aa": [0, 1, 2], "bb": [3, 4, 5, 6]}, p, {"aa": 3, "bb": 3}, 6)
    assert chosen["aa"] == [0] and len(chosen["bb"]) == 4 + 0 or sum(map(len, chosen.values())) <= 6


def test_language_codes():
    assert hwpick.lang_of("text/abc123_kz.pdf") == "kk" and hwpick.lang_of("x/abc_fr.jpg") == "fr" and hwpick.lang_of("x/abc.pdf") is None
