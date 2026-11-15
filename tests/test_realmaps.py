import numpy as np

from dspages.prep.realmaps import MESH, SHEET, draw_aug, sample, step_error


def _page():
    """A sheet bent along its rows (z = 20 sin), equal steps, (360, 256, 3) mm."""
    u = np.linspace(0, 1, MESH[0])
    s = np.linspace(0, SHEET[0], 4000)
    z = 20 * np.sin(np.pi * s / SHEET[0])
    arc = np.r_[0, np.cumsum(np.hypot(np.diff(s), np.diff(z)))]
    t = u * arc[-1]
    x, zz = np.interp(t, arc, s), np.interp(t, arc, z)
    y = np.linspace(SHEET[1], 0, MESH[1])
    return np.stack(np.broadcast_arrays(x[None] * SHEET[0] / arc[-1], y[:, None], zz[None]), -1)


def test_variants_keep_the_sheet():
    G = _page()
    rng = np.random.default_rng(0)
    for mode in ("full", "rot", "slide", "zoom"):
        for _ in range(3):
            g = np.moveaxis(sample(G, draw_aug(rng, mode, *G.shape[:2])), 0, -1)
            assert g.shape == (MESH[1], MESH[0], 3) and np.isfinite(g).all()
            assert step_error(g)[1] < 0.01
            assert abs(g[..., 2].min()) < 1e-3 and g[..., 2].max() > 1.0
            du, dv = np.gradient(g, axis=1), np.gradient(g, axis=0)
            assert (np.cross(du, -dv)[..., 2] > 0).mean() > 0.99  # face up, right-handed
            assert g[0, :, 1].mean() > g[-1, :, 1].mean()  # row 0 at the top
