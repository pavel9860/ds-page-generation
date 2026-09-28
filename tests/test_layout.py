import time

import cv2
import numpy as np

from dspages.config import page_px
from dspages.layout.content import content_mask, select_window, window_fill
from dspages.layout.generate import make_layout
from dspages.layout.text import char_budget, is_rtl, render_text


def _brute_fill(mask, win, grid):
    out = []
    for o in range(mask.shape[0] - win + 1):
        r = np.round(np.linspace(0, win, grid + 1)).astype(int) + o
        c = np.round(np.linspace(0, mask.shape[1], grid + 1)).astype(int)
        cells = [mask[r[i]:r[i + 1], c[j]:c[j + 1]] for i in range(grid) for j in range(grid)]
        out.append(np.mean([cl.sum() > 0.01 * cl.size for cl in cells]))
    return np.array(out)


def test_window_fill_matches_brute_force():
    mask = np.random.default_rng(0).random((300, 120)) < 0.02
    mask[100:180] = False
    assert np.allclose(window_fill(mask, 150, 10), _brute_fill(mask, 150, 10))


def test_select_window_aspect_and_speed():
    rng = np.random.default_rng(1)
    for aspect in (0.5, 0.707, 1.0, 1.5):
        mask = np.zeros((512, 400), bool)
        mask[40:470:6, 30:380] = True
        t = time.perf_counter()
        (y0, x0, h, w), fill = select_window(mask, aspect, rng, 0.8, 10)
        assert time.perf_counter() - t < 0.05
        assert abs(w / h - aspect) < 0.02 and fill >= 0.8
        assert 40 <= y0 and y0 + h <= 470 and 30 <= x0 and x0 + w <= 380


def test_content_mask_finds_text_and_images():
    page = np.full((800, 600), 255, np.uint8)
    cv2.putText(page, "Text line", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)
    cv2.rectangle(page, (100, 400), (500, 700), 90, -1)
    cv2.circle(page, (300, 550), 80, 20, 6)
    mask, s = content_mask(page)
    assert mask[int(200 * s) - 5:int(200 * s) + 2].any()
    assert mask[int(550 * s)].any()
    assert not mask[: int(120 * s)].any()


def test_text_fills_the_page(fonts):
    rng = np.random.default_rng(0)
    words = " ".join(["paper"] * 5000)
    pg = render_text(words, rng, 600, 800, 4.0, fonts, 10, 1.3)
    rows = (pg < 128).any(1)
    assert rows.mean() > 0.5 and rows[-60:].any()
    assert is_rtl("שלום עולם") and not is_rtl("hello")
    assert char_budget(600, 800, 4.0, 10, 1.3) > 2000


def test_layouts_filled_visible_realistic(full_small, manifest, fonts):
    rng = np.random.default_rng(0)
    pw, ph = page_px(full_small.layout.sheet_mm, full_small.layout.canvas_px)
    kinds = set()
    for k, idx in enumerate(rng.choice(len(manifest), 8, replace=False)):
        L = make_layout(np.random.default_rng(k), manifest[idx], full_small, fonts)
        page, gray, m = L["page"], L["gray"], L["meta"]
        kinds.add(m["kind"])
        assert page.shape == (ph, pw, 3) and gray.shape == (ph, pw)
        mx, my = round(m["margin"] * pw), round(m["margin"] * ph)
        if mx:
            assert gray[:, :mx].min() == 255 and gray[:, -mx:].min() == 255
        inner = gray[my:ph - my, mx:pw - mx]
        mask, _ = content_mask(inner)
        cells = [c for row in np.array_split(mask, 4) for c in np.array_split(row, 4, axis=1)]
        assert np.mean([c.any() for c in cells]) >= 0.75
        ink, paper_ = gray < 100, gray > 200
        if ink.any():
            lum = page.mean(-1)
            assert lum[paper_].mean() - lum[ink].mean() > 60
        assert 150 < page[paper_].mean() < 250
    assert kinds
