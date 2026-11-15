import ast
import json
from pathlib import Path

import numpy as np
import pytest

from dspages.fillpic import CONFIG, Config, Detector
from dspages.fillpic.detector import blank, fit
from dspages.fillpic.features import chosen, columns, derive
from dspages.fillpic.kaggle import SHIPPED
from dspages.fillpic.maps import map_groups, map_names
from dspages.fillpic.tiles import tile_matrices, tile_mean, window
from dspages.layout import content
from dspages.prep import manifest as MF

PKG = Path(__file__).resolve().parents[1] / "src" / "dspages" / "fillpic"


def test_ink_rule_is_the_digital_one():
    assert (CONFIG.geometry.mask_side, CONFIG.ink.line_ksize, CONFIG.ink.line_thresh, CONFIG.ink.eps) == (
        content.SCAN_MAX_SIDE,
        content.LINE_KSIZE,
        content.LINE_THRESH,
        content.PATCH_INK_EPS,
    )
    assert CONFIG.geometry.grid == MF._LAYOUT.fill_grid


def test_config_overlay(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"geometry": {"margin": 0.1}, "maps": {"std_kernels": [3, 9]}}))
    c = Config.load(p)
    assert (
        c.geometry.margin == 0.1
        and c.geometry.infer_margin == 0.05
        and c.maps.std_kernels == (3, 9)
        and c.geometry.grid == 16
    )
    assert Config.load() == CONFIG


def test_shipped_modules_are_standalone():
    for name in SHIPPED:
        for node in ast.walk(ast.parse((PKG / name).read_text())):
            if isinstance(node, ast.ImportFrom):
                assert node.level <= 1 and "dspages" not in (node.module or ""), (name, node.module)
            if isinstance(node, ast.Import):
                assert all("dspages" not in a.name for a in node.names), name


def test_tile_sums_match_the_window():
    rng = np.random.default_rng(0)
    m = (rng.random((60, 40)) < 0.3).astype(np.float32)
    win = (-4.0, 6.0, 44.0, 52.0)
    got = tile_mean(m, win, 4)
    ys, xs = (
        np.clip(np.round(6 + 52 * np.linspace(0, 1, 5)).astype(int), 0, 60),
        np.clip(np.round(-4 + 44 * np.linspace(0, 1, 5)).astype(int), 0, 40),
    )
    want = np.array([[m[ys[i] : ys[i + 1], xs[j] : xs[j + 1]].sum() for j in range(4)] for i in range(4)]) / (
        44 * 52 / 16
    )
    assert np.allclose(got, want, atol=1e-6)
    Ty, Tx, area = tile_matrices(win, m.shape, 4)
    assert Ty.shape == (4, 60) and Tx.shape == (4, 40) and area == 44 * 52 / 16


def test_window():
    mask = np.zeros((512, 384), bool)
    assert window(mask, 0.025) is None
    mask[100:200, 50:300] = True
    x, y, w, h = window(mask, 0.0)
    assert w == 250 and h == pytest.approx(250 * 1024 / 768) and x == 50 and y == 100
    x2, y2, w2, h2 = window(mask, 0.1)
    assert w2 == pytest.approx(w / 0.8) and (x2 + w2 / 2) == pytest.approx(x + w / 2)


def test_columns_match_derive():
    n = len(map_names())
    cells = np.random.default_rng(1).random((2, 16, 16, n)).astype(np.float32)
    X = derive(cells, np.full((2, 2), 100.0, np.float32))
    assert X.shape == (2 * 256, len(columns()))
    assert len(map_names()) == sum(len(v) for v in map_groups().values())
    assert chosen(set(map_names()), set(CONFIG.features.base_derived) | set(CONFIG.features.candidate_derived)).all()
    assert chosen(set(), {"page"}).sum() == 3


def test_fit_stretches_to_the_canvas():
    assert fit(np.zeros((500, 300), np.uint8)).shape == CONFIG.geometry.canvas
    g = np.zeros(CONFIG.geometry.canvas, np.uint8)
    assert fit(g) is g


def test_detector_blank_page_and_shapes(tmp_path):
    xgb = pytest.importorskip("xgboost")
    n = len(columns())
    rng = np.random.default_rng(2)
    X = rng.random((300, n)).astype(np.float32)
    ink = xgb.XGBClassifier(n_estimators=3, max_depth=2).fit(X, rng.random(300) < 0.5)
    pic = xgb.XGBRegressor(n_estimators=3, max_depth=2, objective="reg:logistic").fit(X, rng.random(300))
    det = Detector(ink, pic, np.ones(n, bool))
    white = np.full(CONFIG.geometry.canvas, 255, np.uint8)
    text = white.copy()
    text[200:230, 100:600] = 0
    r_blank, r_text = det.predict([white, text])
    assert r_blank.window == blank().window and r_blank.fill == 0 and r_blank.pic_mean == 0
    assert r_text.ink.shape == (16, 16) and 0 <= r_text.fill <= 1 and 0 <= r_text.pic_mean <= 1 and r_text.window[2] > 0
    det.save(tmp_path)
    again = Detector.load(tmp_path)
    assert again.predict([text])[0].fill == r_text.fill
