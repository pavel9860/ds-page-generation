"""Scenes: what shapes the page profile, with parameters drawn feasible by construction.

A scene holds, for one page end: clamp (edge tangent angle at s = 0, edge on the table, or None), supports
[(s, h)] (line supports at arc length s, height h, sorted), folds [(s, theta, k)] (theta > 0 valley, k hinge
stiffness), flat (near-flat variant) and crumple (flat page with a deep crease network).
Feasibility:
  folds     gaps >= fold_min_gap; clamp + sum |theta| <= fold_total_deg (fold_total_deg_supported with
            supports); no fold within fold_min_gap of a support line, also while the line turns
  supports  gaps >= support_min_gap; |dh| <= reach_slope ds from the clamp (z = 0 at s = 0) and between supports
  end b     each support line turns by <= min(skew_deg, atan(skew_rise / (n h (1 + height_jitter)))): the strip
            blend across the page then keeps membrane strain <= 0.1 %
"""
from dataclasses import replace

import numpy as np

from ..config import SceneCfg


def pick(rng, table):
    keys = [k for k, _ in table]
    p = np.array([w for _, w in table], float)
    return keys[rng.choice(len(keys), p=p / p.sum())]


def _spread(rng, m, length, gap):
    """m sorted positions in (0, length) with all gaps, edges included, >= gap."""
    return gap * np.arange(1, m + 1) + np.cumsum(rng.dirichlet(np.ones(m + 1)))[:m] * (length - (m + 1) * gap)


def _near_flat(c: SceneCfg):
    return replace(c, support_h=c.flat_support_h, clamp_deg=c.flat_clamp_deg, fold_angle_deg=c.flat_fold_deg,
                   fold_angle_deg_median=float(np.sqrt(np.prod(c.flat_fold_deg))))


def _folds(rng, c: SceneCfg, length, budget_deg):
    tpl = pick(rng, c.fold_templates)
    if tpl is None:
        pos = _spread(rng, rng.integers(1, c.fold_max + 1), length, c.fold_min_gap)
    else:
        j = min(c.fold_jitter, 0.5 * (min(np.diff(np.r_[0.0, tpl, 1.0])) * length - c.fold_min_gap))
        pos = np.array(tpl) * length + rng.uniform(-j, j, len(tpl))
    m = len(pos)
    ang = np.clip(c.fold_angle_deg_median * np.exp(rng.normal(0.0, c.fold_angle_sigma, m)), *c.fold_angle_deg)
    ang *= min(1.0, budget_deg / ang.sum())
    sign = np.where(rng.random(m) < c.fold_valley_prob, 1.0, -1.0)
    k = np.clip(c.fold_k_median * np.exp(rng.normal(0.0, c.fold_k_sigma, m)), *c.fold_k)
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


def sample_scene(rng, c: SceneCfg, length, combo=None, flat=None):
    """End-a scene. combo = (clamp, n_supports, folds, crumple) or drawn from c.combos."""
    has_clamp, n_sup, has_folds, crumple = combo or pick(rng, c.combos)
    flat = rng.random() < c.near_flat_prob if flat is None else flat
    c = _near_flat(c) if flat else c
    clamp = float(np.radians(rng.uniform(*c.clamp_deg))) if has_clamp else None
    lo, hi = _support_bounds(c, length, clamp)
    s = lo + _spread(rng, n_sup, hi - lo, c.support_min_gap)
    supports = list(zip(s.tolist(), _heights(rng, c, s, clamp)))
    budget = _fold_budget(c, supports, clamp)
    folds = []
    if has_folds and budget >= c.fold_angle_deg[0]:
        folds = [f for f in _folds(rng, c, length, budget)
                 if all(abs(f[0] - p) >= c.fold_min_gap for p, _ in supports)]
    return dict(clamp=clamp, supports=supports, folds=folds, flat=flat, crumple=crumple)


def _side_b(rng, c: SceneCfg, sa, length, width):
    """Other page end: supports turned and re-heighted, clamp and fold angles jittered; fold lines stay straight.
    -> (end a without the folds a turning support line would cross, end b)."""
    c = _near_flat(c) if sa["flat"] else c
    hi_clamp = c.clamp_deg[1]
    if sa["folds"]:
        hi_clamp = min(hi_clamp, _fold_budget(c, sa["supports"], 0.0) - c.fold_angle_deg[0])
    clamp = None if sa["clamp"] is None else float(np.clip(
        sa["clamp"] * (1 + c.side_b_jitter * rng.uniform(-1, 1)), np.radians(c.clamp_deg[0]), np.radians(hi_clamp)))
    lo, hi = _support_bounds(c, length, clamp)
    n = len(sa["supports"])
    pa = [p for p, _ in sa["supports"]]
    shift = [width * min(np.tan(np.radians(c.skew_deg)), c.skew_rise / (n * (1 + c.height_jitter) * h))
             for _, h in sa["supports"]]
    upper = []
    for p, d in zip(pa[::-1], shift[::-1]):
        upper.append(min(hi, p + d, upper[-1] - c.support_min_gap if upper else hi))
    s = []
    for p, d, u in zip(pa, shift, upper[::-1]):
        s.append(float(rng.uniform(max(lo, p - d, s[-1] + c.support_min_gap if s else lo), u)))
    wanted = [((1 - c.height_jitter) * h, (1 + c.height_jitter) * h) for _, h in sa["supports"]]
    supports = list(zip(s, _heights(rng, c, s, clamp, wanted)))
    sa = dict(sa, folds=[f for f in sa["folds"] if all(
        f[0] <= min(a, b) - c.fold_min_gap or f[0] >= max(a, b) + c.fold_min_gap for a, b in zip(pa, s))])
    budget = max(np.radians(_fold_budget(c, supports, clamp)), 0.0)
    lo_a, hi_a = np.radians(c.fold_angle_deg)
    jit = c.fold_side_b_angle_jitter
    folds = [(p, float(np.sign(t) * np.clip(abs(t) * (1 + jit * rng.uniform(-1, 1)), lo_a, hi_a)), k)
             for p, t, k in sa["folds"]]
    f = min(1.0, budget / max(sum(abs(t) for _, t, _ in folds), 1e-12))
    return sa, dict(clamp=clamp, supports=supports, folds=[(p, t * f, k) for p, t, k in folds],
                    flat=sa["flat"], crumple=sa["crumple"])


def sample_ends(rng, c: SceneCfg, length, width, two, combo=None, flat=None):
    """Scenes at both page ends (equal when not two)."""
    sa = sample_scene(rng, c, length, combo, flat)
    return _side_b(rng, c, sa, length, width) if two else (sa, sa)


def lerp_scene(a, b, t):
    def lerp(x, y):
        return (1 - t) * x + t * y
    return dict(clamp=None if a["clamp"] is None else lerp(a["clamp"], b["clamp"]),
                supports=[(lerp(s0, s1), lerp(h0, h1)) for (s0, h0), (s1, h1) in zip(a["supports"], b["supports"])],
                folds=[(s0, lerp(t0, t1), k0) for (s0, t0, k0), (_, t1, _) in zip(a["folds"], b["folds"])],
                flat=a["flat"], crumple=a["crumple"])


def scene_label(sc):
    return "+".join(filter(None, ["flat" if sc["flat"] else "", "crumple" if sc["crumple"] else "",
                                  "clamp" if sc["clamp"] is not None else "",
                                  f"{len(sc['supports'])}sup" if sc["supports"] else "",
                                  f"{len(sc['folds'])}fold" if sc["folds"] else ""])) or "flat_table"
