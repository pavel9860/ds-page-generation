import numpy as np
import pymupdf
from dspages.multilang import bank, pages
from dspages.multilang.config import Config

CFG = Config()


def test_yolo_rows_clip_and_drop_tiny():
    rows = pages.yolo_rows([(0, [(-5, 0, 50, 100), (10, 10, 11, 11)]), (1, [(0, 0, 100, 100)])], 100, 100, CFG)
    assert rows == ["0 0.25000 0.50000 0.50000 1.00000", "1 0.50000 0.50000 1.00000 1.00000"]


def test_page_segments_split_leaders_labels_gaps():
    doc = pymupdf.open()
    pg = doc.new_page(width=600, height=200)
    pg.insert_text((20, 50), "Name:  ..............  Date:", fontsize=11)
    pg.insert_text((20, 100), "left", fontsize=11)
    pg.insert_text((400, 100), "right", fontsize=11)
    prints, fields, labels = pages.page_segments(pg, CFG)
    assert len(fields) == 1
    assert len(labels) == 2
    assert sum(b[1] > 80 for b in prints) == 2


def test_strip_keeps_header_boxes_relative():
    img = np.zeros((1000, 800, 3), np.uint8)
    bg = pages.Bg(img, [(10, 20, 100, 40), (10, 500, 100, 520)], [], [], img[..., 0], [], img[..., 0], [])
    strip = pages.print_strip(bg, type("R", (), {"random": lambda s: 0.0})(), CFG)
    assert strip["top"] and strip["img"].shape[0] == 46
    assert strip["boxes"] == [(10, 20, 100, 40)]


def test_chunk_write_is_whole(tmp_path):
    p = tmp_path / "a.pkl"
    bank.write_chunk(p, [1, 2])
    assert p.exists() and not list(tmp_path.glob("*.tmp"))


def test_config_sha_tracks_values():
    from dataclasses import replace
    assert CFG.sha() == Config().sha() != replace(CFG, seed=1).sha()


def test_render_bg_finds_checkboxes():
    doc = pymupdf.open()
    pg = doc.new_page(width=300, height=300)
    for y in (50, 90, 130, 170):
        pg.insert_text((60, y + 9), "option text line", fontsize=11)
    pg.draw_rect(pymupdf.Rect(30, 80, 42, 92))
    doc.save("/tmp/hwdet_test_checks.pdf")
    import random
    from dataclasses import replace
    bg = pages.render_bg("/tmp/hwdet_test_checks.pdf", random.Random(0), replace(CFG, ticks=True))
    assert len(bg.checks) == 1
    assert pages.render_bg("/tmp/hwdet_test_checks.pdf", random.Random(0), CFG).checks == []


def test_degrade_keeps_shape_and_type():
    import random
    img = np.full((120, 90, 3), 230, np.uint8)
    for seed in range(6):
        out = pages.degrade(img, random.Random(seed), CFG, 1.0)
        assert out.shape == img.shape and out.dtype == np.uint8


def test_items_are_never_enlarged():
    import random

    items = [dict(ink=np.zeros((h * n, 40), np.uint8), boxes=[(0, 0, 1, 1)] * n) for h, n in ((12, 1), (30, 1), (44, 2))]
    rng = random.Random(0)
    for target in (8, 20, 30, 60):
        for _ in range(50):
            it = pages.pick(rng, items, target, set())
            assert pages.fit(it, target) <= 1.0
            assert pages.line_h(it) >= target or it is max(items, key=pages.line_h)
