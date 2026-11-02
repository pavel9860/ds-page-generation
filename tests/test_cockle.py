import numpy as np

from dspages.creases import cockle

STEP = (210 / 255, 297 / 359)


def _flat():
    y, x = np.mgrid[0:360, 0:256].astype(float)
    return np.dstack([x * STEP[0], y * STEP[1], np.zeros_like(x)])


def _ratio(G, ts, seeds, scale=1.0):
    return np.array([cockle.bands(cockle.apply(G, np.random.default_rng(s), t, scale)[..., 2]).std((1, 2)) / t["rms"]
                     for t in ts for s in seeds])


def test_flat_page_reaches_every_band():
    r = _ratio(_flat(), cockle.targets()[:8], range(3))
    assert np.all(np.abs(np.sqrt((r ** 2).mean(0)) - 1) < 0.1), np.sqrt((r ** 2).mean(0))
    assert np.all(np.abs(r[:, :3] - 1) < 0.2)             # fine bands: many cycles per page, little scatter


def test_present_energy_is_kept_not_doubled():
    t = cockle.targets()[0]
    once = cockle.apply(_flat(), np.random.default_rng(1), t)
    r = cockle.bands(cockle.apply(once, np.random.default_rng(2), t)[..., 2]).std((1, 2)) / t["rms"]
    assert np.all(r < 1.25), r


def test_scale_and_xy_shift():
    G = _flat()
    r = _ratio(G, cockle.targets()[3:4], range(4), scale=0.5)
    assert np.all(np.abs(np.sqrt((r ** 2).mean(0)) - 0.5) < 0.07), np.sqrt((r ** 2).mean(0))
    C = cockle.apply(G, np.random.default_rng(0), cockle.targets()[3])
    assert np.allclose(C[..., :2], G[..., :2]) and C[..., 2].min() == 0


def test_elongation_along_height():
    t = dict(rms=np.full(6, 0.5), lx=np.full(6, 1.0), ly=np.full(6, 2.0))
    m = cockle.measure(cockle.field((360, 256), np.random.default_rng(3), t), STEP)
    iso = cockle.measure(np.random.default_rng(0).standard_normal((360, 256)), STEP)
    assert m["ly"][3] / m["lx"][3] > 1.3 * iso["ly"][3] / iso["lx"][3]


def test_surface_with_water(full_small):
    """Added energy on the coarse test grid: Z resampled to the band tables' pitch reaches the drawn page's bands."""
    import cv2
    from dspages.geometry.surface import make_surface
    cond = dict(combo=("book", 0, False, False), two=False, along_long=False, bend_dir=0.0, gsm=80.0, shallow=False, deep="")
    g, p = full_small.geometry, full_small.paper
    dry = make_surface(np.random.default_rng(0), g, p, cond)
    wet = make_surface(np.random.default_rng(0), g, p, dict(cond, water=1.0))
    b0, b1 = (cockle.bands(cv2.resize(s["Z"].astype(float), (256, 360))).std((1, 2)) for s in (dry, wet))
    assert np.all(b1[2:] > b0[2:]), (b0, b1)
    assert np.isfinite(wet["Z"]).all() and wet["Z"].min() >= 0
