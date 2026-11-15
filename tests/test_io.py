"""Sample files: written whole or not at all, arrays and meta read back unchanged, previews where asked."""

import numpy as np

from dspages import io


def test_round_trip_with_meta_and_numpy_scalars(tmp_path):
    a = {"flat": np.arange(12, dtype=np.uint8).reshape(3, 4), "grid": np.ones((2, 2, 2), np.float32)}
    io.save(tmp_path / "s" / "0.npz", a, meta=dict(n=np.int64(3), v=np.float32(0.5), arr=np.array([1, 2])))
    got, meta = io.load(tmp_path / "s" / "0.npz")
    assert set(got) == set(a) and all((got[k] == a[k]).all() and got[k].dtype == a[k].dtype for k in a)
    assert meta == dict(n=3, v=0.5, arr=[1, 2])
    assert not list((tmp_path / "s").glob("*.tmp"))


def test_no_meta_and_a_preview(tmp_path):
    img = np.zeros((8, 8, 3), np.uint8)
    io.save(tmp_path / "1.npz", {"x": img}, preview=img)
    assert io.load(tmp_path / "1.npz")[1] == {} and (tmp_path / "1.jpg").exists()


def test_height_preview_is_rgb_of_the_grid():
    z = np.linspace(0, 5, 20).reshape(4, 5)
    out = io.height_preview(z)
    assert out.shape == (4, 5, 3) and out.dtype == np.uint8
    assert io.height_preview(np.zeros((3, 3))).shape == (3, 3, 3)
