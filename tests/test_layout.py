
import cv2
import numpy as np

from dspages.config import page_px
from dspages.layout.content import content_mask, place_window, window_fill
from dspages.layout.generate import make_layout
from dspages.layout.text import is_rtl, render_text


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


def _place(box, aspect, M, min_w, seed=0):
    mask = np.ones((400, 300), bool)
    return place_window(np.random.default_rng(seed), box, aspect, M, min_w, mask, 1.0, np.zeros_like(mask), 0.5, 10,
                        0.6)


def test_window_keeps_margins_within_limit():
    for seed in range(20):
        (x, y, w, h), _, _ = _place((50, 60, 200, 290), 0.707, 0.1, 0.0, seed)
        assert abs(w / h - 0.707) < 1e-9
        assert x <= 50 and y <= 60 and x + w >= 250 and y + h >= 350
        assert max(50 - x, x + w - 250, 60 - y, y + h - 350) <= 0.1 * max(w, h) + 1e-9


def test_window_crops_only_the_long_axis_and_respects_min_width():
    (x, y, w, h), _, _ = _place((50, 10, 200, 380), 1.0, 0.0, 0.0)
    assert w == h == 200 and x == 50 and 10 <= y and y + h <= 390
    (x, y, w, h), _, _ = _place((50, 60, 100, 100), 1.0, 0.1, 150.0)
    assert w == 150


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


def test_layouts_filled_visible_realistic(full_small, manifest, fonts):
    rng = np.random.default_rng(0)
    pw, ph = page_px(full_small.layout.sheet_mm, full_small.layout.canvas_px)
    kinds = set()
    for k, idx in enumerate(rng.choice(len(manifest), 8, replace=False)):
        L = make_layout(np.random.default_rng(k), manifest[idx], full_small, fonts)
        page, gray, m = L["page"], L["gray"], L["meta"]
        kinds.add(m["kind"])
        assert page.shape == (ph, pw) and gray.shape == (ph, pw)
        mx, my = round(m["margin"] * pw), round(m["margin"] * ph)
        if mx and m["kind"] == "book_text":
            assert gray[:, :mx].min() == 255 and gray[:, -mx:].min() == 255
        inner = gray[my:ph - my, mx:pw - mx]
        mask, _ = content_mask(inner)
        cells = [c for row in np.array_split(mask, 4) for c in np.array_split(row, 4, axis=1)]
        assert np.mean([c.any() for c in cells]) >= 0.75
        ink, paper_ = gray < 100, gray > 200
        if ink.any():
            lum = page.astype(float)
            assert lum[paper_].mean() - lum[ink].mean() > 60
        assert 150 < page[paper_].mean() < 250
        assert m.get("upscale", 1.0) <= 1.0 + 1e-6 or not m["source"].lower().endswith(".pdf")
        c = full_small.layout
        assert m.get("image_share", 0.0) <= c.max_image + 1e-6 or m.get("fill", 1) < c.min_fill
    assert kinds


def test_picture_share_limit():
    mask = np.ones((400, 100), bool)
    pictures = np.zeros_like(mask)
    pictures[:200] = True
    _, _, share = place_window(np.random.default_rng(0), (0, 0, 100, 400), 1.0, 0.0, 0.0, mask, 1.0, pictures, 0.5,
                               10, 0.3)
    assert share <= 0.3


def test_ruled_form_is_not_a_picture(full_small):
    from pathlib import Path

    import pytest
    from dspages.layout.pictures import picture_mask
    if not Path(full_small.layout.pictures.model).expanduser().exists():
        pytest.skip("picture model not installed")
    page = np.full((1200, 850), 255, np.uint8)
    for y in range(100, 1100, 40):
        cv2.rectangle(page, (60, y), (790, y + 40), 0, 2)
        cv2.putText(page, "Name / Date / Signature", (80, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    assert picture_mask(page, page.shape, 1.0, full_small.layout.pictures).mean() <= full_small.layout.max_image
