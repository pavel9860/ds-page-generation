"""Scenes: what shapes the page profile, with parameters drawn feasible by construction.

A scene holds, for one page end: clamp (edge tangent angle at s = 0, edge on the table, or None), base [(s0, s1, H)]
(planes the page lies on: a book page beyond s_p), halves (an open book: two book pages clamped at the spine in the
middle, each [clamp, base] along its half from the spine out; None otherwise), supports
[(s, h)] (line supports at arc length s, height h, sorted), folds [(s, theta, k)] (theta > 0 valley, k hinge
stiffness) and crumple (flat page with a deep crease network).
Feasibility:
  folds     gaps >= fold_min_gap; clamp + sum |theta| <= fold_total_deg (fold_total_deg_supported with
            supports); no fold within fold_min_gap of a support line, also while the line turns
  supports  gaps >= support_min_gap; |dh| <= reach_slope ds from the clamp (z = 0 at s = 0) and between supports
  end b     each support line turns by <= min(skew_deg, atan(skew_rise / (n h (1 + height_jitter)))): the strip
            blend across the page then keeps membrane strain <= 0.1 %
"""

import numpy as np
from scipy.special import ndtr, ndtri

from ..config import SceneCfg


def pick(rng, table):
    keys = [k for k, _ in table]
    p = np.array([w for _, w in table], float)
    return keys[rng.choice(len(keys), p=p / p.sum())]


def _spread(rng, m, length, gap):
    """m sorted positions in (0, length) with all gaps, edges included, >= gap."""
    return gap * np.arange(1, m + 1) + np.cumsum(rng.dirichlet(np.ones(m + 1)))[:m] * (length - (m + 1) * gap)


def _lognormal(rng, median, sigma, bounds, m):
    """m draws of a log-normal truncated to bounds (no pile-up at the bounds)."""
    a, b = ndtr(np.log(np.array(bounds) / median) / sigma)
    return median * np.exp(sigma * ndtri(rng.uniform(a, b, m)))


def _folds(rng, c: SceneCfg, length, budget_deg):
    tpl = pick(rng, c.fold_templates)
    if tpl is None:
        pos = _spread(rng, rng.integers(1, c.fold_max + 1), length, c.fold_min_gap)
    else:
        j = min(c.fold_jitter, 0.5 * (min(np.diff(np.r_[0.0, tpl, 1.0])) * length - c.fold_min_gap))
        pos = np.array(tpl) * length + rng.uniform(-j, j, len(tpl))
    m = len(pos)
    ang = _lognormal(rng, c.fold_angle_deg_median, c.fold_angle_sigma, c.fold_angle_deg, m)
    ang *= min(1.0, budget_deg / ang.sum())
    sign = np.where(rng.random(m) < c.fold_valley_prob, 1.0, -1.0)
    k = _lognormal(rng, c.fold_k_median, c.fold_k_sigma, c.fold_k, m)
    return [(float(p), float(t), float(kk)) for p, t, kk in zip(pos, sign * np.radians(ang), k)]


def _heights(rng, c: SceneCfg, s, clamp, wanted=None):
    out, prev = [], (0.0, 0.0) if clamp is not None else None
    for i, si in enumerate(s):
        lo, hi = wanted[i] if wanted is not None else c.support_h
        if prev is not None:
            ds = c.reach_slope * (si - prev[0])
            lo, hi = (np.clip(v, max(prev[1] - ds, 0.0), prev[1] + ds) for v in (lo, hi))
        out.append(float(rng.uniform(lo, hi)))
        prev = (si, out[-1])
    return out


def _support_bounds(c: SceneCfg, length, clamp):
    return (c.clamp_free_len if clamp is not None else 0.0) + c.support_edge * length, (1 - c.support_edge) * length


def _fold_budget(c: SceneCfg, supports, clamp):
    """Degrees left for the folds."""
    return (c.fold_total_deg_supported if supports else c.fold_total_deg) - (np.degrees(clamp) if clamp else 0.0)


def sample_scene(rng, c: SceneCfg, length, combo=None):
    """End-a scene. combo = (clamp, n_supports, folds, crumple) or drawn from c.combos."""
    has_clamp, n_sup, has_folds, crumple = combo or pick(rng, c.combos)
    if has_clamp == "open":
        halves = []
        for _ in range(2):
            clamp, sp = float(np.radians(rng.uniform(*c.open_deg))), float(rng.uniform(*c.open_plane) * 0.5 * length)
            h = rng.uniform(c.open_h[0], min(c.open_h[1], c.book_reach * sp * np.sin(clamp)))
            halves.append(dict(clamp=clamp, base=[(sp, 0.5 * length, float(h))]))
        return dict(clamp=None, supports=[], folds=[], crumple=False, base=[], halves=halves)
    book = has_clamp == "book"
    clamp = float(np.radians(rng.uniform(*(c.book_deg if book else c.clamp_deg)))) if has_clamp else None
    base = []
    if book:
        sp = rng.uniform(*c.book_plane) * length
        base = [
            (float(sp), length, float(rng.uniform(c.book_h[0], min(c.book_h[1], c.book_reach * sp * np.sin(clamp)))))
        ]
    lo, hi = _support_bounds(c, length, clamp)
    s = lo + _spread(rng, n_sup, hi - lo, c.support_min_gap)
    supports = list(zip(s.tolist(), _heights(rng, c, s, clamp)))
    budget = _fold_budget(c, supports, clamp)
    folds = []
    if has_folds and budget >= c.fold_angle_deg[0]:
        folds = [f for f in _folds(rng, c, length, budget) if all(abs(f[0] - p) >= c.fold_min_gap for p, _ in supports)]
    return dict(clamp=clamp, supports=supports, folds=folds, crumple=crumple, base=base, halves=None)


def _side_b(rng, c: SceneCfg, sa, length, width):
    """Other page end: supports turned and re-heighted, clamp and fold angles jittered; fold lines stay straight.
    -> (end a without the folds a turning support line would cross, end b)."""
    if sa["halves"]:
        lo, hi = np.radians(c.open_deg)
        d = np.radians(c.open_side_b_deg)
        halves = []
        for hf in sa["halves"]:
            clamp = float(np.clip(hf["clamp"] + rng.uniform(-d, d), lo, hi))
            halves.append(
                dict(
                    clamp=clamp,
                    base=[
                        (
                            s0,
                            s1,
                            float(
                                min(H * (1 + c.height_jitter * rng.uniform(-1, 1)), c.book_reach * s0 * np.sin(clamp))
                            ),
                        )
                        for s0, s1, H in hf["base"]
                    ],
                )
            )
        return sa, dict(sa, halves=halves)
    lo_clamp, hi_clamp = c.book_deg if sa["base"] else c.clamp_deg
    if sa["folds"]:
        hi_clamp = min(hi_clamp, _fold_budget(c, sa["supports"], 0.0) - c.fold_angle_deg[0])
    clamp = (
        None
        if sa["clamp"] is None
        else float(
            np.clip(
                sa["clamp"] * (1 + c.side_b_jitter * rng.uniform(-1, 1)), np.radians(lo_clamp), np.radians(hi_clamp)
            )
        )
    )
    base = [
        (
            s0,
            s1,
            float(
                min(
                    H * (1 + c.height_jitter * rng.uniform(-1, 1)),
                    c.book_reach * s0 * np.sin(clamp) if clamp is not None else np.inf,
                )
            ),
        )
        for s0, s1, H in sa["base"]
    ]
    lo, hi = _support_bounds(c, length, clamp)
    n = len(sa["supports"])
    pa = [p for p, _ in sa["supports"]]
    shift = [
        width * min(np.tan(np.radians(c.skew_deg)), c.skew_rise / (n * (1 + c.height_jitter) * h))
        for _, h in sa["supports"]
    ]
    upper = []
    for p, d in zip(pa[::-1], shift[::-1]):
        upper.append(min(hi, p + d, upper[-1] - c.support_min_gap if upper else hi))
    s = []
    for p, d, u in zip(pa, shift, upper[::-1]):
        s.append(float(rng.uniform(max(lo, p - d, s[-1] + c.support_min_gap if s else lo), u)))
    wanted = [((1 - c.height_jitter) * h, (1 + c.height_jitter) * h) for _, h in sa["supports"]]
    supports = list(zip(s, _heights(rng, c, s, clamp, wanted)))
    sa = dict(
        sa,
        folds=[
            f
            for f in sa["folds"]
            if all(f[0] <= min(a, b) - c.fold_min_gap or f[0] >= max(a, b) + c.fold_min_gap for a, b in zip(pa, s))
        ],
    )
    budget = max(np.radians(_fold_budget(c, supports, clamp)), 0.0)
    lo_a, hi_a = np.radians(c.fold_angle_deg)
    jit = c.fold_side_b_angle_jitter
    folds = [
        (p, float(np.sign(t) * np.clip(abs(t) * (1 + jit * rng.uniform(-1, 1)), lo_a, hi_a)), k)
        for p, t, k in sa["folds"]
    ]
    f = min(1.0, budget / max(sum(abs(t) for _, t, _ in folds), 1e-12))
    return sa, dict(
        clamp=clamp,
        supports=supports,
        folds=[(p, t * f, k) for p, t, k in folds],
        crumple=sa["crumple"],
        base=base,
        halves=None,
    )


def sample_ends(rng, c: SceneCfg, length, width, two, combo=None):
    """Scenes at both page ends (equal when not two)."""
    sa = sample_scene(rng, c, length, combo)
    return _side_b(rng, c, sa, length, width) if two else (sa, sa)


def lerp_scene(a, b, t):
    def lerp(x, y):
        return (1 - t) * x + t * y

    def planes(pa, pb):
        return [(s0, s1, lerp(h0, h1)) for (s0, s1, h0), (_, _, h1) in zip(pa, pb)]

    return dict(
        clamp=None if a["clamp"] is None else lerp(a["clamp"], b["clamp"]),
        supports=[(lerp(s0, s1), lerp(h0, h1)) for (s0, h0), (s1, h1) in zip(a["supports"], b["supports"])],
        folds=[(s0, lerp(t0, t1), k0) for (s0, t0, k0), (_, t1, _) in zip(a["folds"], b["folds"])],
        crumple=a["crumple"],
        base=planes(a["base"], b["base"]),
        halves=a["halves"]
        and [
            dict(clamp=lerp(ha["clamp"], hb["clamp"]), base=planes(ha["base"], hb["base"]))
            for ha, hb in zip(a["halves"], b["halves"])
        ],
    )


def scene_label(sc):
    return (
        "+".join(
            filter(
                None,
                [
                    "crumple" if sc["crumple"] else "",
                    "book" if sc["base"] else "",
                    "open" if sc["halves"] else "",
                    "clamp" if sc["clamp"] is not None and not sc["base"] else "",
                    f"{len(sc['supports'])}sup" if sc["supports"] else "",
                    f"{len(sc['folds'])}fold" if sc["folds"] else "",
                ],
            )
        )
        or "flat_table"
    )
