"""3D page grid from a scene.

The profile is solved along direction bend_dir (to the page's u axis, or its v axis with along_long) on the
page's rotated bounding box: p along the profile, g along its generators. Two-profile pages blend strips solved
at n_slices positions across g. Sag (shell under gravity) acts outside the support strip. Shallow creases and a
deep crease network are height maps in page coordinates, carried onto the box. Rows and columns of the box are
laid out inextensibly, then the page grid (u, v) is sampled from the box and turned back to the page frame.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ..config import GeometryCfg, paper_props
from ..creases import deep, shallow
from .sag import solve_sag
from .scenes import lerp_scene, sample_ends, scene_label
from .strip import solve_strip


def inextensible(x, z, u, v):
    """Grid (x, z) along rows spaced u: rebuild x so each row keeps its arc lengths; y from column arc lengths."""
    x = x[:, :1] + np.concatenate([np.zeros((len(v), 1)),
                                   np.cumsum(np.sqrt(np.maximum(np.diff(u)[None, :] ** 2 - np.diff(z, axis=1) ** 2, 0)),
                                             axis=1)], axis=1)
    dl = np.diff(v)[:, None] ** 2 - np.diff(x, axis=0) ** 2 - np.diff(z, axis=0) ** 2
    return x, np.concatenate([np.zeros((1, len(u))), np.cumsum(np.sqrt(np.maximum(dl, 0)), axis=0)], axis=0)


def strains(X, Y, Z, u, v):
    su = np.sqrt(np.diff(X, axis=1) ** 2 + np.diff(Y, axis=1) ** 2 + np.diff(Z, axis=1) ** 2) / np.diff(u)[None, :] - 1
    sv = np.sqrt(np.diff(X, axis=0) ** 2 + np.diff(Y, axis=0) ** 2 + np.diff(Z, axis=0) ** 2) / np.diff(v)[:, None] - 1
    return su, sv


def _swap(a, b, along_long):
    """Sheet (a along the width, b along the height) <-> L-frame (l along the profile base axis, m across)."""
    return (b, a) if along_long else (a, b)


def _crease_heights(rng, g: GeometryCfg, cond, fold_ls, lm_to_p):
    """Height map [m] on the sheet (rows b, columns a) of the shallow creases and the deep network, its axes,
    and the crease lines [mm]. Shallow creases keep fold_clear_mm from the fold lines (p = const)."""
    w_mm, h_mm = g.sheet_mm
    lines, h, pitch = [], None, None
    if cond["shallow"]:
        c = g.shallow
        folds = np.array(fold_ls)
        cand = shallow.sample(rng, w_mm, h_mm, c.mean_groups, c.p_isolated, c.mean_extra, c.radius_mm, c.spread_deg,
                              c.singles_per_group, c.k)
        lines = [cr for cr in cand if not len(folds) or np.abs(
            lm_to_p(*_swap(cr["line"][:, 0] * 1e-3, cr["line"][:, 1] * 1e-3, cond["along_long"]))[:, None]
            - folds[None, :]).min() > c.fold_clear_mm * 1e-3]
        h, pitch = shallow.render(lines, w_mm, h_mm, c.pitch_mm, c.w_max_mm), c.pitch_mm
    if cond["deep"]:
        net = deep.sample(rng, w_mm, h_mm, cond["deep"], g.deep)
        hd = deep.render(net, w_mm, h_mm, cond["deep"], g.deep)
        h, pitch = (hd if h is None else h + hd), g.deep.pitch_mm
        lines = lines + net
    if h is None:
        return None, None, lines
    return 1e-3 * h, (np.arange(h.shape[0]) * pitch * 1e-3, np.arange(h.shape[1]) * pitch * 1e-3), lines


def make_surface(rng, g: GeometryCfg, paper, cond):
    """-> dict with the page grid X, Y, Z [mm] (rows down the sheet, columns across; face up, right-handed: Y points
    up the sheet) and what
    made it: scenes at both ends, paper, conditions, crease lines, max membrane strain of the laid-out box."""
    pp = paper_props(paper, cond["gsm"])
    sw, sh = (s * 1e-3 for s in g.sheet_mm)
    length, width = _swap(sw, sh, cond["along_long"])
    beta = np.radians(cond["bend_dir"])
    cb, sn = np.cos(beta), np.sin(beta)
    lp, wg = length * cb + width * abs(sn), length * abs(sn) + width * cb
    p0, g0 = (0.0, length * sn) if beta >= 0 else (-width * sn, 0.0)

    def lm_to_p(l_, m_):
        return l_ * cb + m_ * sn + p0

    pitch = min(sw / (g.mesh[0] - 1), sh / (g.mesh[1] - 1))
    up = np.linspace(0.0, lp, int(np.ceil(lp / pitch)) + 1)
    vg = np.linspace(0.0, wg, int(np.ceil(wg / pitch)) + 1)

    sa, sb = sample_ends(rng, g.scene, lp, wg, cond["two"], cond["combo"], cond["flat"])
    t = np.linspace(0.0, 1.0, g.n_slices if cond["two"] else 1)
    slices, y = [], None
    for tk in t:
        sc = lerp_scene(sa, sb, tk)
        s, x, z, y = solve_strip(lp, pp["q"], pp["bending"], sc["folds"], sc["supports"], sc["clamp"],
                                 g.strip_segments, y)
        slices.append((np.interp(up, s, x), np.interp(up, s, z)))
    k = np.clip(np.searchsorted(t, vg / wg, side="right") - 1, 0, max(len(t) - 2, 0))
    wt = ((vg / wg - t[k]) / (t[min(1, len(t) - 1)] - t[0] or 1.0))[:, None]
    xs, zs = (np.array([sl[j] for sl in slices]) for j in (0, 1))
    k1 = np.minimum(k + 1, len(t) - 1)
    x, z = ((1 - wt) * a[k] + wt * a[k1] for a in (xs, zs))
    span = (0.0, wg)
    if sa["supports"]:
        f = rng.uniform(*g.support_frac)
        a0 = rng.uniform(max(0.0, 0.5 - f), min(0.5, 1 - f)) * wg
        span = (a0, a0 + f * wg)
    if span[1] - span[0] < wg:
        z = z + solve_sag(up, vg, z, span, pp, g.shell_nodes)

    hmap, axes, lines = _crease_heights(rng, g, cond, [f[0] for f in sa["folds"]], lm_to_p)
    if hmap is not None:
        P, Gq = np.meshgrid(up, vg)
        a_, b_ = _swap((P - p0) * cb - (Gq - g0) * sn, (P - p0) * sn + (Gq - g0) * cb, cond["along_long"])
        z = z + RegularGridInterpolator(axes, hmap, bounds_error=False, fill_value=None)(
            np.stack([np.clip(b_, 0, sh), np.clip(a_, 0, sw)], -1))
    if cond["deep"]:
        z = z - z.min()
    z = np.maximum(z, 0.0)
    xd, yd = inextensible(x, z, up, vg)

    u = np.linspace(0.0, sw, g.mesh[0])
    v = np.linspace(0.0, sh, g.mesh[1])
    l_, m_ = _swap(*np.meshgrid(u, v), cond["along_long"])
    pts = np.stack([np.clip(-l_ * sn + m_ * cb + g0, 0.0, wg), np.clip(lm_to_p(l_, m_), 0.0, lp)], -1)
    X, Y, Z = (RegularGridInterpolator((vg, up), A)(pts) for A in (xd, yd, z))
    X, Y = _swap((X - p0) * cb - (Y - g0) * sn, (X - p0) * sn + (Y - g0) * cb, cond["along_long"])
    return dict(X=1e3 * X, Y=1e3 * (sh - Y), Z=1e3 * Z, u=1e3 * u, v=1e3 * v, scene_a=sa, scene_b=sb,
                label=scene_label(sa), paper=pp, cond=cond, creases=lines,
                box_strain=float(max(np.abs(a).max() for a in strains(xd, yd, z, up, vg))))
