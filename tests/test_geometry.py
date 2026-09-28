import numpy as np
import pytest

from conftest import DATA
from dspages.creases import deep, shallow
from dspages.geometry.scenes import sample_ends
from dspages.geometry.strip import solve_strip
from dspages.geometry.surface import make_surface


def _reference_cases():
    f = np.load(DATA / "reference_profiles.npz")
    for i in range(int(f["n"])):
        kind, L, *p = str(f[f"c{i}_key"]).split("|")
        L = float(L)
        if kind == "clamp":
            kw = dict(clamp=np.radians(float(p[0])))
        elif kind == "center":
            kw = dict(supports=[(L / 2, float(p[0]))])
        else:
            a = [float(v) for v in p[0].split(",")]
            kw = dict(supports=[(a[0] * L, a[1]), (a[2] * L, a[3])])
        yield L, kw, f[f"c{i}_s"], f[f"c{i}_z"], float(f["q"]), float(f["d"])


@pytest.mark.parametrize("case", list(_reference_cases()), ids=lambda c: str(c[1]))
def test_strip_matches_reference_solvers(case):
    L, kw, s_ref, z_ref, q, d = case
    s, x, z = solve_strip(L, q, d, **kw)
    assert np.abs(np.interp(s_ref, s, z) - z_ref).max() < 2e-5
    assert np.all(np.diff(x) > 0)


def test_scene_feasibility(full_small):
    c = full_small.geometry.scene
    rng = np.random.default_rng(0)
    for i in range(400):
        combo = c.combos[i % len(c.combos)][0]
        L, W = (0.297, 0.21) if i % 2 else (0.21, 0.297)
        sa, sb = sample_ends(rng, c, L, W, True, combo)
        for sc in (sa, sb):
            s = [p for p, _ in sc["supports"]]
            assert np.all(np.diff(s) >= c.support_min_gap - 1e-12)
            prev = (0.0, 0.0) if sc["clamp"] is not None else None
            for p, h in sc["supports"]:
                assert h >= 0
                if prev:
                    assert abs(h - prev[1]) <= c.reach_slope * (p - prev[0]) + 1e-12
                prev = (p, h)
            if sc["folds"]:
                total = np.degrees((sc["clamp"] or 0) + sum(abs(t) for _, t, _ in sc["folds"]))
                assert total <= (c.fold_total_deg_supported if sc["supports"] else c.fold_total_deg) + 1e-6
                assert all(abs(np.degrees(t)) <= c.fold_angle_deg[1] + 1e-6 for _, t, _ in sc["folds"])
        for f in sa["folds"]:
            assert all(f[0] <= min(a, b) - c.fold_min_gap + 1e-12 or f[0] >= max(a, b) + c.fold_min_gap - 1e-12
                       for (a, _), (b, _) in zip(sa["supports"], sb["supports"]))


def _arc(p):
    return np.linalg.norm(np.diff(p, axis=0), axis=1).sum()


def _conds(P):
    for i, (combo, _) in enumerate(P.geometry.scene.combos):
        yield dict(combo=combo, flat=bool(i % 3 == 0), two=not combo[3] and i % 2 == 0, along_long=bool(i % 2),
                   bend_dir=float(-40 + 7 * i), gsm=P.paper.gsm[i % len(P.paper.gsm)], shallow=bool(i % 2),
                   deep=("light", "medium", "heavy")[i % 3] if combo[3] else "")


@pytest.mark.parametrize("i", range(13))
def test_surface_isometric_on_table(full_small, i):
    cond = list(_conds(full_small))[i]
    s = make_surface(np.random.default_rng(i), full_small.geometry, full_small.paper, cond)
    X, Y, Z = s["X"], s["Y"], s["Z"]
    assert np.isfinite(X).all() and np.isfinite(Y).all() and np.isfinite(Z).all()
    assert Z.min() >= -0.02
    assert s["box_strain"] <= (0.005 if cond["deep"] else 0.001)
    w, h = full_small.geometry.sheet_mm
    rows = [np.stack([X[k], Y[k], Z[k]], -1) for k in (0, -1)]
    cols = [np.stack([X[:, k], Y[:, k], Z[:, k]], -1) for k in (0, -1)]
    edges = [_arc(e) / w - 1 for e in rows] + [_arc(e) / h - 1 for e in cols]
    assert np.abs(edges).max() < 0.01


def test_surface_right_handed_face_up(full_small):
    cond = dict(combo=(False, 0, False, False), flat=False, two=False, along_long=False, bend_dir=0.0, gsm=80,
                shallow=False, deep="")
    s = make_surface(np.random.default_rng(0), full_small.geometry, full_small.paper, cond)
    du = np.array([s["X"][0, -1] - s["X"][0, 0], s["Y"][0, -1] - s["Y"][0, 0], 0])
    dv = np.array([s["X"][-1, 0] - s["X"][0, 0], s["Y"][-1, 0] - s["Y"][0, 0], 0])
    assert np.cross(du, -dv)[2] > 0


def test_crease_heights_bounded(full_small):
    g = full_small.geometry
    rng = np.random.default_rng(0)
    c = g.shallow
    lines = shallow.sample(rng, 210, 297, c.mean_groups, c.p_isolated, c.mean_extra, c.radius_mm, c.spread_deg,
                           c.singles_per_group, c.k)
    assert np.abs(shallow.render(lines, 210, 297, 1.0, c.w_max_mm)).max() <= c.w_max_mm
    for level, _ in g.deep.levels:
        h = deep.render(deep.sample(rng, 210, 297, level, g.deep), 210, 297, level, g.deep)
        assert 0 < np.abs(h).max() <= g.deep.w_max_mm[level]
