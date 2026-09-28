from dataclasses import replace

import numpy as np
import pytest

from dspages.conditions import plan
from dspages.config import page_px
from dspages.geometry.surface import make_surface
from dspages.render.generate import make_sample
from dspages.render.raster import grid_faces, rasterize


def test_raster_plane_is_linear():
    nu, nv = 5, 4
    uu, vv = np.meshgrid(np.linspace(0, 1, nu), np.linspace(0, 1, nv))
    pix = np.stack([10 + 80 * uu.ravel(), 20 + 60 * vv.ravel()], -1)
    uv, a, z = rasterize(pix, np.full(nu * nv, 100.0), np.stack([uu.ravel(), vv.ravel(), np.ones(nu * nv)], -1),
                         grid_faces(nv, nu), (100, 100))
    ys, xs = np.mgrid[0:100, 0:100] + 0.5
    inside = np.isfinite(uv[..., 0])
    assert inside[25:75, 15:85].all() and not inside[:15].any()
    assert np.allclose(uv[..., 0][inside], ((xs - 10) / 80)[inside], atol=1e-5)
    assert np.allclose(uv[..., 1][inside], ((ys - 20) / 60)[inside], atol=1e-5)


def _samples(P, n):
    pw, ph = page_px(P.layout.sheet_mm, P.layout.canvas_px)
    yy, xx = np.mgrid[0:ph, 0:pw]
    page = np.where(((xx // 6 + yy // 6) % 2 == 0)[..., None], 30, 230).astype(np.uint8).repeat(3, -1)
    for k, spec in enumerate(plan(P, n, 1, 1)):
        cond = {key: spec[key] for key in ("combo", "flat", "two", "along_long", "bend_dir", "gsm", "shallow", "deep")}
        surf = make_surface(np.random.default_rng(k), P.geometry, P.paper, cond)
        yield page, surf, make_sample(np.random.default_rng(k), page, surf, P)


@pytest.mark.parametrize("name", ["demo", "full"])
def test_samples_in_frame_readable_consistent(name, demo, full_small):
    P = demo if name == "demo" else full_small
    P = replace(P, geometry=replace(P.geometry, mesh=(64, 90), n_slices=3))
    for page, surf, o in _samples(P, 6):
        mask, uv = o["mask"], o["uv"].astype(np.float32)
        h, w = mask.shape
        assert mask.any() and not (mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
        ok = np.isfinite(uv[..., 0])
        assert ok.any() and uv[ok].min() >= -1e-3 and uv[ok].max() <= 1 + 1e-3
        corners = np.array([[0.02, 0.02], [0.98, 0.02], [0.02, 0.98], [0.98, 0.98]])
        d = np.linalg.norm(uv[ok][:, None, :] - corners[None], axis=-1).min(0)
        assert d.max() < 0.05
        view = o["meta"]["view"]
        assert view["max_incidence_deg"] <= P.render.camera.incidence_deg + 1e-6
        assert o["map3d"].shape == (*reversed(P.render.map_px), 3)
        if name == "demo":
            assert abs(o["warped"].mean(-1)[mask].mean() - page.mean()) < 20
