from dataclasses import dataclass

import numpy as np

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


def sample_params(rng, cfg: PageConfig, kind, length):
    r = cfg.ranges
    if kind == "clamp":
        return (np.radians(rng.uniform(*r.clamp_deg)),)
    if kind == "center_support":
        return (rng.uniform(*r.center_h),)
    return (rng.uniform(*r.two_p1_frac) * length, rng.uniform(*r.two_h),
            rng.uniform(*r.two_p2_frac) * length, rng.uniform(*r.two_h))


def support_lines(kind, params, length):
    if kind == "center_support":
        return [0.5 * length]
    if kind == "two_support":
        return [params[0], params[2]]
    return []


def shell_sag(rng, cfg: PageConfig, kind, params, u, v, z0, support_frac=None):
    frac = rng.uniform(*cfg.support_len_frac) if support_frac is None else support_frac
    half = 0.5 * frac * (v[-1] - v[0])
    span = (0.5 * v[-1] - half, 0.5 * v[-1] + half)
    lines = [(pu, *span) for pu in support_lines(kind, params, u[-1])]
    return solve_w(u, v, z0, lines, span if kind == "clamp" else None, cfg.paper, cfg.shell_nodes), span


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


def make_page(rng, cfg: PageConfig = PageConfig(), kind=None, two_profiles=None, along_long=None, support_frac=None):
    kind = kind or rng.choice(list(cfg.kind_probs), p=list(cfg.kind_probs.values()))
    along_long = rng.random() < cfg.profile_along_long_prob if along_long is None else along_long
    two = rng.random() < cfg.two_profile_prob if two_profiles is None else two_profiles
    length, width = (cfg.height, cfg.width) if along_long else (cfg.width, cfg.height)

    params_a = sample_params(rng, cfg, kind, length)
    j = cfg.side_b_jitter
    params_b = tuple((1 - j) * a + j * b for a, b in zip(params_a, sample_params(rng, cfg, kind, length))) if two else params_a

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
    if two:
        sag, span = np.zeros_like(z0), (0.0, width)
    else:
        sag, span = shell_sag(rng, cfg, kind, params_a, u, v, z0, support_frac)
    Z = z0 + sag
    X, Y = inextensible(x, Z, u, v)
    su, sv = strains(X, Y, Z, u, v)
    return Page(kind, params_a, params_b, along_long, u, v, slices, z0, sag, span, X, Y, Z, su, sv)
