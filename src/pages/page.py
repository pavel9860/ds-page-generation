from dataclasses import dataclass

import numpy as np

from creases import shallow
from elastica import solve_case

from .config import PageConfig
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


def sample_params(rng, cfg: PageConfig, kind, length):
    r = cfg.ranges
    if kind == "clamp":
        return (np.radians(rng.uniform(*r.clamp_deg)),)
    if kind == "folded":
        return sample_folds(rng, r, length)
    if kind == "center_support":
        return (rng.uniform(*r.center_h),)
    return (rng.uniform(*r.two_p1_frac) * length, rng.uniform(*r.two_h),
            rng.uniform(*r.two_p2_frac) * length, rng.uniform(*r.two_h))


def sample_folds(rng, r, length):
    """Flat (s_1..s_M, theta_1..theta_M, k_1..k_M), unused slots s = nan. Random
    valley/mountain per crease, so C-folds, Z-folds and mixed patterns all occur."""
    keys = list(r.fold_templates)
    tpl = keys[rng.choice(len(keys), p=list(r.fold_templates.values()))]
    while True:
        if tpl is None:
            pos = rng.uniform(r.fold_min_gap, length - r.fold_min_gap, rng.integers(1, r.fold_max + 1))
        else:
            pos = np.array(tpl) * length + rng.normal(0.0, r.fold_jitter, len(tpl))
        pos = np.sort(pos)
        gaps = np.diff(np.concatenate([[0.0], pos, [length]]))
        if gaps.min() >= r.fold_min_gap:
            break
    m = len(pos)
    ang = np.clip(r.fold_angle_deg_median * np.exp(rng.normal(0.0, r.fold_angle_sigma, m)), *r.fold_angle_deg)
    sign = np.where(rng.random(m) < r.fold_valley_prob, 1.0, -1.0)
    k = r.fold_k_median * np.exp(rng.normal(0.0, r.fold_k_sigma, m))
    pad = np.full(r.fold_max - m, np.nan)
    return tuple(np.concatenate([pos, pad, sign * np.radians(ang), pad, k, pad]))


def folded_side_b(rng, cfg: PageConfig, params_a):
    """Creases are straight lines: same positions and stiffness, only the rest angle varies across the width."""
    m = cfg.ranges.fold_max
    p = np.array(params_a)
    p[m:2 * m] *= 1.0 + cfg.ranges.fold_side_b_angle_jitter * rng.uniform(-1.0, 1.0, m)
    return tuple(p)


def support_span(rng, cfg: PageConfig, width, frac=None):
    frac = rng.uniform(*cfg.support_frac) if frac is None else frac
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


def make_page(rng, cfg: PageConfig = PageConfig(), kind=None, two_profiles=None, along_long=None,
              support_frac=None, span=None):
    kind = kind or rng.choice(list(cfg.kind_probs), p=list(cfg.kind_probs.values()))
    along_long = rng.random() < cfg.profile_along_long_prob if along_long is None else along_long
    two = rng.random() < cfg.two_profile_prob if two_profiles is None else two_profiles
    length, width = (cfg.height, cfg.width) if along_long else (cfg.width, cfg.height)

    params_a = sample_params(rng, cfg, kind, length)
    j = cfg.side_b_jitter
    if not two:
        params_b = params_a
    elif kind == "folded":
        params_b = folded_side_b(rng, cfg, params_a)
    else:
        params_b = tuple((1 - j) * a + j * b for a, b in zip(params_a, sample_params(rng, cfg, kind, length)))

    u = np.linspace(0.0, length, cfg.nu)
    v = np.linspace(0.0, width, cfg.nv)
    t = np.linspace(0.0, 1.0, cfg.n_slices if two else 1)
    slices = []
    for tk in t:
        params = tuple((1 - tk) * a + tk * b for a, b in zip(params_a, params_b))
        s, x, z = solve_case(kind, params, length, cfg.paper.q, cfg.paper.bending_stiffness)
        slices.append((np.interp(u, s, x), np.interp(u, s, z)))
    tv = v / width
    x = np.array([np.interp(tv, t, [sl[0][i] for sl in slices]) for i in range(len(u))]).T
    z0 = np.array([np.interp(tv, t, [sl[1][i] for sl in slices]) for i in range(len(u))]).T
    if two or kind == "folded":         # folded page lies on the table: no support-strip sag
        span = (0.0, width)
        sag = np.zeros_like(z0)
    else:
        span = span or support_span(rng, cfg, width, support_frac)
        sag = solve_w(u, v, z0, span, cfg.paper, cfg.shell_nodes)
    Z = z0 + sag
    X, Y = inextensible(x, Z, u, v)
    su, sv = strains(X, Y, Z, u, v)
    page = Page(kind, params_a, params_b, along_long, u, v, slices, z0, sag, span, X, Y, Z, su, sv)
    sc = cfg.shallow_creases
    if sc.prob > 0 and rng.random() < sc.prob:          # drawn after the geometry: surface unchanged
        page.creases, page.crease_h = shallow_crease_map(rng, sc, length * 1e3, width * 1e3)
    return page


def shallow_crease_map(rng, sc, length_mm, width_mm):
    creases = shallow.sample(rng, length_mm, width_mm, sc.mean_groups, sc.p_isolated, sc.mean_extra,
                             sc.radius_mm, sc.spread_deg, sc.singles_per_group)
    return creases, shallow.render(creases, length_mm, width_mm, sc.pitch_mm)
