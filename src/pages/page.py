from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from creases import shallow
from elastica.multifold import solve_strip

from .config import PageConfig
from .scenes import lerp_scene, sample_paper, sample_scene, scene_label, scene_side_b
from .shell import solve_w


@dataclass
class Page:
    kind: str
    params_a: tuple
    params_b: tuple
    along_long: bool
    u: np.ndarray
    v: np.ndarray
    slices: list
    z0: np.ndarray
    sag: np.ndarray
    support_span: tuple
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    strain_u: np.ndarray
    strain_v: np.ndarray
    creases: list = None                # shallow creases: centerlines [mm] in (u, v), severity, sign, group
    crease_h: np.ndarray = None         # their height map [mm] on a (v, u) grid of pitch cfg.shallow_creases.pitch_mm
    bend_dir: float = 0.0               # profile direction to the page's length axis [rad]


def support_span(rng, cfg: PageConfig, width, frac=None):
    frac = rng.uniform(*cfg.scenes.support_frac) if frac is None else frac
    lo = rng.uniform(max(0.0, 0.5 * width - frac * width), min(0.5 * width, width - frac * width))
    return lo, lo + frac * width


def inextensible(x, z, u, v):
    x = x[:, :1] + np.concatenate([np.zeros((len(v), 1)),
                                   np.cumsum(np.sqrt(np.diff(u)[None, :] ** 2 - np.diff(z, axis=1) ** 2), axis=1)], axis=1)
    dl = np.diff(v)[:, None] ** 2 - np.diff(x, axis=0) ** 2 - np.diff(z, axis=0) ** 2
    y = np.concatenate([np.zeros((1, len(u))), np.cumsum(np.sqrt(dl), axis=0)], axis=0)
    return x, y


def strains(X, Y, Z, u, v):
    su = np.sqrt(np.diff(X, axis=1) ** 2 + np.diff(Y, axis=1) ** 2 + np.diff(Z, axis=1) ** 2) / np.diff(u)[None, :] - 1.0
    sv = np.sqrt(np.diff(X, axis=0) ** 2 + np.diff(Y, axis=0) ** 2 + np.diff(Z, axis=0) ** 2) / np.diff(v)[:, None] - 1.0
    return su, sv


def make_page(rng, cfg: PageConfig = PageConfig(), two_profiles=None, along_long=None,
              support_frac=None, span=None, scene=None, bend_dir=None):
    """Profile solved along bend_dir (rad, to the page's length axis) on the page's rotated bounding box
    (p along the profile, g along the generators), then sampled at the page grid (u, v)."""
    along_long = rng.random() < cfg.profile_along_long_prob if along_long is None else along_long
    two = rng.random() < cfg.two_profile_prob if two_profiles is None else two_profiles
    length, width = (cfg.height, cfg.width) if along_long else (cfg.width, cfg.height)
    beta = np.radians(rng.uniform(-cfg.bend_dir_deg, cfg.bend_dir_deg)) if bend_dir is None else bend_dir
    cb, sn = np.cos(beta), np.sin(beta)
    lp, wg = length * cb + width * abs(sn), length * abs(sn) + width * cb   # domain = rotated bounding box
    p0, g0 = (0.0, length * sn) if beta >= 0 else (-width * sn, 0.0)

    def to_domain(uu, vv):
        return uu * cb + vv * sn + p0, -uu * sn + vv * cb + g0

    sa = scene or sample_scene(rng, cfg.scenes, lp)
    sb = scene_side_b(rng, cfg.scenes, sa, lp, wg) if two else sa
    paper = sample_paper(rng, cfg.scenes, cfg.paper)

    up = np.linspace(0.0, lp, cfg.nu)
    vg = np.linspace(0.0, wg, cfg.nv)
    t = np.linspace(0.0, 1.0, cfg.n_slices if two else 1)
    slices = []
    for tk in t:
        sc = lerp_scene(sa, sb, tk)
        s, x, z = solve_strip(lp, paper.q, paper.bending_stiffness,
                              creases=sc["folds"], supports=sc["supports"], clamp=sc["clamp"])
        slices.append((np.interp(up, s, x), np.interp(up, s, z)))
    tv = vg / wg
    x = np.array([np.interp(tv, t, [sl[0][i] for sl in slices]) for i in range(len(up))]).T
    z0 = np.array([np.interp(tv, t, [sl[1][i] for sl in slices]) for i in range(len(up))]).T
    if sa["supports"] or sa["clamp"] is not None:   # sag only where the supports are shorter than the page
        span = span or support_span(rng, cfg, wg, support_frac)
    else:
        span = (0.0, wg)
    sag = solve_w(up, vg, z0, span, paper, cfg.shell_nodes) if span[1] - span[0] < wg else np.zeros_like(z0)
    Zd = z0 + sag
    creases, crease_h = None, None
    shc = cfg.shallow_creases
    if shc.prob > 0 and rng.random() < shc.prob:    # shallow creases (page coords): never on a fold line
        folds_p = np.array([f[0] for f in sa["folds"]])
        creases = [c for c in shallow.sample(rng, length * 1e3, width * 1e3, shc.mean_groups, shc.p_isolated,
                                             shc.mean_extra, shc.radius_mm, shc.spread_deg, shc.singles_per_group)
                   if not len(folds_p) or np.abs(to_domain(c["line"][:, 0] * 1e-3, c["line"][:, 1] * 1e-3)[0][:, None]
                                                 - folds_p[None, :]).min() > shc.fold_clear_mm * 1e-3]
        if creases:
            crease_h = shallow.render(creases, length * 1e3, width * 1e3, shc.pitch_mm)
            Pd, Gd = np.meshgrid(up, vg)           # domain grid -> page coords (clamped to the page edge,
            uu = np.clip((Pd - p0) * cb - (Gd - g0) * sn, 0.0, length)   # so off-page rows continue smoothly)
            vv = np.clip((Pd - p0) * sn + (Gd - g0) * cb, 0.0, width)
            hv, hu = (np.arange(n) * shc.pitch_mm * 1e-3 for n in crease_h.shape)
            hmap = RegularGridInterpolator((hv, hu), crease_h, bounds_error=False, fill_value=None)
            Zd = Zd + 1e-3 * hmap(np.stack([vv, uu], axis=-1))
    Zd = np.maximum(Zd, 0.0)                        # the shell sag and shallow creases know no table: rest on it
    Xd, Yd = inextensible(x, Zd, up, vg)
    u = np.linspace(0.0, length, cfg.nu)
    v = np.linspace(0.0, width, cfg.nv)
    P_, G_ = to_domain(*np.meshgrid(u, v))
    pts = np.stack([np.clip(G_, 0.0, wg), np.clip(P_, 0.0, lp)], axis=-1)
    X, Y, Z, z0p, sagp = (RegularGridInterpolator((vg, up), A)(pts) for A in (Xd, Yd, Zd, z0, sag))
    X, Y = (X - p0) * cb - (Y - g0) * sn, (X - p0) * sn + (Y - g0) * cb   # back to the page frame
    su, sv = strains(X, Y, Z, u, v)
    return Page(scene_label(sa), sa, sb, along_long, u, v, slices, z0p, sagp, span, X, Y, Z, su, sv,
                creases, crease_h, beta)