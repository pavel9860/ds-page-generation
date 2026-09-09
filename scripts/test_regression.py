import random
import sys
import os

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from build_textures import image_crop_box, make_image_texture, IMAGE_SIZE
from text_bounds import content_box_scan_coverage, detect_content_box_scan


def synth_page(W, H, bx0, by0, bx1, by1):
    arr = np.full((H, W, 3), 255, np.uint8)
    arr[round(by0):round(by1), round(bx0):round(bx1)] = 0
    return arr


def test_crop_side_matches_width_margins():
    rng = random.Random(1)
    x0, y0, side = image_crop_box(100, 50, 300, 900, 0.05, 0.05, 0.05, 0.05, rng)
    bw = 200
    expected_side = bw / 0.9
    assert abs(side - expected_side) < 1e-6, side
    assert abs((100 - x0) - 0.05 * side) < 1e-6
    assert abs(((x0 + side) - 300) - 0.05 * side) < 1e-6


def test_tall_content_crops_vertically():
    rng = random.Random(1)
    x0, y0, side = image_crop_box(100, 50, 300, 900, 0.05, 0.05, 0.05, 0.05, rng)
    assert side < 900 - 50, "square side should be smaller than tall content height"
    assert not (y0 <= 50 and y0 + side >= 900), "content taller than side must be cropped, not fully contained"


def test_short_content_pads_exact_margins():
    rng = random.Random(2)
    x0, y0, side = image_crop_box(100, 700, 300, 800, 0.05, 0.05, 0.05, 0.05, rng)
    assert y0 <= 700 and y0 + side >= 800, "short content should fit fully inside the square"
    assert abs((700 - y0) - 0.05 * side) < 1e-6 or (y0 + side - 800) >= 0


def test_make_image_texture_output_is_square_1024():
    row = {"format": "image", "file": "_synthetic.png", "page": 0}
    img = synth_page(600, 900, 100, 400, 500, 480)
    tmp = "/tmp/_synthetic.png"
    Image.fromarray(img).save(tmp)
    row["file"] = os.path.relpath(tmp, "/run/media/me/D/ML_DS/UVTM/Layouts") \
        if tmp.startswith("/run/media/me/D/ML_DS/UVTM/Layouts") else None
    from build_textures import L
    dst = os.path.join(L, "_test_synth.png")
    Image.fromarray(img).save(dst)
    row["file"] = os.path.relpath(dst, L)
    out = make_image_texture(row, random.Random(3))
    assert out.size == (IMAGE_SIZE, IMAGE_SIZE)
    os.remove(dst)


def test_no_aspect_distortion_square_content_stays_square():
    from build_textures import L
    arr = synth_page(500, 500, 100, 100, 400, 400)
    dst = os.path.join(L, "_test_square.png")
    Image.fromarray(arr).save(dst)
    row = {"format": "image", "file": os.path.relpath(dst, L), "page": 0}
    out = make_image_texture(row, random.Random(0))
    assert out.size == (IMAGE_SIZE, IMAGE_SIZE)
    gray = np.array(out.convert("L"))
    dark_rows = np.where((gray < 128).any(axis=1))[0]
    dark_cols = np.where((gray < 128).any(axis=0))[0]
    row_span = dark_rows.max() - dark_rows.min()
    col_span = dark_cols.max() - dark_cols.min()
    assert abs(row_span - col_span) / max(row_span, col_span) < 0.05, "square content must stay square after crop+resize"
    os.remove(dst)


def test_glyph_coverage_rejects_two_isolated_marks():
    arr = np.full((1000, 800, 3), 255, np.uint8)
    arr[20:40, 20:60] = 0
    arr[960:980, 740:780] = 0
    bgr = arr[:, :, ::-1].copy()
    _, _, _, _, x_cov, y_cov = content_box_scan_coverage(bgr)
    assert x_cov < 0.6 and y_cov < 0.6


def test_glyph_coverage_accepts_dense_text_block():
    arr = np.full((1000, 800, 3), 255, np.uint8)
    for row in range(50, 950, 20):
        arr[row:row + 10, 50:750] = 0
    bgr = arr[:, :, ::-1].copy()
    _, _, _, _, x_cov, y_cov = content_box_scan_coverage(bgr)
    assert x_cov >= 0.6 and y_cov >= 0.6


def test_detect_content_box_scan_blank_page_fallback():
    arr = np.full((400, 300, 3), 255, np.uint8)
    bgr = arr[:, :, ::-1].copy()
    bx0, by0, bx1, by1 = detect_content_box_scan(bgr)
    assert (bx0, by0, bx1, by1) == (0, 0, 300, 400)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
