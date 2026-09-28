"""Page scenes: which supports shape the profile, and feasible parameters for them.

A scene is dict(clamp, supports, folds) for one page end:
  clamp:    edge tangent angle at s = 0 (edge held at z = 0), or None
  supports: [(s, h)] line supports at arc length s, height h (sorted by s)
  folds:    [(s, theta, k)] plastic folds, theta > 0 valley, k hinge stiffness
Every draw is taken from an interval that is feasible by construction (no retry loops):
  - folds: gaps >= fold_min_gap (stick breaking); total |theta| <= fold_total_deg - clamp
    (fold_total_deg_supported with supports); folds within fold_min_gap of a support line are dropped
  - supports: gaps >= support_min_gap, heights reachable from the clamp / previous support
    (|dh| <= reach_slope * ds)
  - end b: each support line turned by <= min(skew_deg, atan(skew_rise / h)), still ordered, in bounds, reachable
"""
from dataclasses import dataclass, field, replace

import numpy as np


@dataclass
class SceneConfig:
    # combination table: (clamp, n_supports, folds) -> probability
    combos: dict = field(default_factory=lambda: {
        (False, 0, True): 0.20,                   # folded page on the table
        (False, 1, False): 0.10, (False, 2, False): 0.10, (False, 3, False): 0.05,
        (False, 1, True): 0.10, (False, 2, True): 0.10, (False, 3, True): 0.05,
        (True, 0, False): 0.10, (True, 1, False): 0.05, (True, 2, False): 0.03,
        (True, 0, True): 0.07, (True, 1, True): 0.05})
    clamp_deg: tuple = (5.0, 60.0)
    clamp_free_len: float = 0.03       # m, first support at least this far past a clamp
    support_edge: float = 0.05         # supports keep this fraction of the length from the ends
    support_min_gap: float = 0.04      # m, between supports
    support_h: tuple = (0.005, 0.040)  # m
    reach_slope: float = 0.5           # max |dh / ds| between clamp / supports (~27 deg)
    skew_deg: float = 60.0             # a support line may turn up to this across the page ...
    skew_rise: float = 0.0035          # ... and by at most atan(skew_rise / h) at height h: the strip blend keeps
                                       # membrane strain <= 0.1 % below this (measured: 10 mm 29 deg, 40 mm 8 deg)
    side_b_jitter: float = 0.35        # relative jitter of clamp angle at end b
    height_jitter: float = 0.2         # relative jitter of support heights at end b
    # folds: template positions (fractions of length) or None = 1..fold_max random
    fold_templates: dict = field(default_factory=lambda: {
        (0.5,): 0.3, (1 / 3, 2 / 3): 0.3, (0.25, 0.5, 0.75): 0.1, None: 0.3})
    fold_max: int = 4
    fold_jitter: float = 0.006         # m, template position noise (bounded by half the min gap)
    fold_min_gap: float = 0.03         # m, between folds and to the edges
    fold_valley_prob: float = 0.5
    fold_angle_deg_median: float = 25.0   # residual rest angle after unfolding, lognormal
    fold_angle_sigma: float = 0.6
    fold_angle_deg: tuple = (3.0, 60.0)
    fold_total_deg: float = 90.0       # clamp + sum |theta| never passes vertical
    fold_total_deg_supported: float = 45.0   # with supports: the page already hangs steeply beside them
    fold_k_median: float = 0.01        # N*m/m per rad, hinge stiffness, lognormal
    fold_k_sigma: float = 0.7
    fold_k: tuple = (0.003, 0.05)      # softer is a free hinge (mechanism, no unique rest state)
    fold_side_b_angle_jitter: float = 0.25
    # paper, per page: bending stiffness ~ thickness^3 ~ grammage^3, membrane ~ grammage
    grammage_gsm: tuple = (60.0, 120.0)
    stiffness_exp: float = 3.0
    support_frac: tuple = (0.2, 1.0)   # sag: fraction of the width the supports hold
    # near-flat pages: low supports, small folds and clamp
    near_flat_prob: float = 0.3
    flat_support_h: tuple = (0.001, 0.005)
    flat_fold_deg: tuple = (1.0, 8.0)
    flat_clamp_deg: tuple = (2.0, 10.0)


def _pick(rng, table):
    keys = list(table)
    return keys[rng.choice(len(keys), p=np.array(list(table.values())) / sum(table.values()))]


def _gaps(rng, m, length, gap):
    """m sorted positions in (0, length), all gaps (edges included) >= gap: stick breaking."""
    return gap * np.arange(1, m + 1) + np.cumsum(rng.dirichlet(np.ones(m + 1)))[:m] * (length - (m + 1) * gap)


def sample_folds(rng, c: SceneConfig, length, budget_deg):
    tpl = _pick(rng, c.fold_templates)
    if tpl is None:
        pos = _gaps(rng, rng.integers(1, c.fold_max + 1), length, c.fold_min_gap)
    else:
        j = min(c.fold_jitter, 0.5 * (min(np.diff(np.r_[0.0, tpl, 1.0])) * length - c.fold_min_gap))
        pos = np.array(tpl) * length + rng.uniform(-j, j, len(tpl))
    m = len(pos)
    ang = np.clip(c.fold_angle_deg_median * np.exp(rng.normal(0.0, c.fold_angle_sigma, m)), *c.fold_angle_deg)
    ang *= min(1.0, budget_deg / ang.sum())
    sign = np.where(rng.random(m) < c.fold_valley_prob, 1.0, -1.0)
    k = np.clip(c.fold_k_median * np.exp(rng.normal(0.0, c.fold_k_sigma, m)), *c.fold_k)
    return [(float(p), float(t), float(kk)) for p, t, kk in zip(pos, sign * np.radians(ang), k)]


def _heights(rng, c: SceneConfig, s, clamp, lo_hi=None):
    """Heights drawn inside the interval reachable from the clamp (z = 0 at s = 0) or previous support."""
    out, prev = [], (0.0, 0.0) if clamp is not None else None
    for i, si in enumerate(s):
        lo, hi = lo_hi[i] if lo_hi is not None else c.support_h
        if prev is not None:
            ds = c.reach_slope * (si - prev[0])
            rlo, rhi = max(prev[1] - ds, 0.0), prev[1] + ds   # reachable; clip the wanted range into it
            lo, hi = np.clip(lo, rlo, rhi), np.clip(hi, rlo, rhi)
        h = rng.uniform(lo, hi)
        out.append(h)
        prev = (si, h)
    return out


def _support_bounds(c: SceneConfig, length, clamp):
    return (c.clamp_free_len if clamp is not None else 0.0) + c.support_edge * length, (1 - c.support_edge) * length


def _flat(c: SceneConfig):
    return replace(c, support_h=c.flat_support_h, clamp_deg=c.flat_clamp_deg, fold_angle_deg=c.flat_fold_deg,
                   fold_angle_deg_median=float(np.sqrt(np.prod(c.flat_fold_deg))))


def sample_scene(rng, c: SceneConfig, length):
    """End-a scene from the combination table; near-flat with probability near_flat_prob."""
    flat = rng.random() < c.near_flat_prob
    c = _flat(c) if flat else c
    has_clamp, n_sup, has_folds = _pick(rng, c.combos)
    clamp = np.radians(rng.uniform(*c.clamp_deg)) if has_clamp else None
    lo, hi = _support_bounds(c, length, clamp)
    s = lo + _gaps(rng, n_sup, hi - lo, c.support_min_gap)
    supports = list(zip(s.tolist(), _heights(rng, c, s, clamp)))
    budget = (c.fold_total_deg_supported if supports else c.fold_total_deg) - (np.degrees(clamp) if clamp is not None else 0.0)
    folds = _clear(sample_folds(rng, c, length, budget), supports, c) if has_folds and budget >= c.fold_angle_deg[0] else []
    return dict(clamp=clamp, supports=supports, folds=folds, flat=flat)


def _clear(folds, supports, c: SceneConfig):
    """Drop folds on or right next to a support line (the page hangs there; a fold would turn it past vertical)."""
    return [f for f in folds if all(abs(f[0] - s) >= c.fold_min_gap for s, _ in supports)]


def scene_side_b(rng, c: SceneConfig, sa, length, width):
    """Other page end. Each support line turns by <= min(skew_deg, atan(skew_rise / h)); positions stay
    ordered (min gap) and in bounds, heights stay reachable. Fold lines stay straight: same position and stiffness."""
    c = _flat(c) if sa["flat"] else c
    clamp = None if sa["clamp"] is None else float(np.clip(
        sa["clamp"] * (1 + c.side_b_jitter * rng.uniform(-1, 1)), *np.radians(c.clamp_deg)))
    lo, hi = _support_bounds(c, length, clamp)
    sa_s = [p for p, _ in sa["supports"]]
    n = len(sa["supports"])                        # the rise budget is shared: turns of several supports add up
    sh = [width * min(np.tan(np.radians(c.skew_deg)), c.skew_rise / (n * (1 + c.height_jitter) * h)) for _, h in sa["supports"]]
    up = []                                        # upper bounds, back to front: own turn limit, next one's bound
    for p, d in zip(sa_s[::-1], sh[::-1]):
        up.append(min(hi, p + d, up[-1] - c.support_min_gap if up else hi))
    up = up[::-1]
    s = []                                         # forward: never empty, the unturned end-a positions fit all bounds
    for p, d, u in zip(sa_s, sh, up):
        s.append(rng.uniform(max(lo, p - d, s[-1] + c.support_min_gap if s else lo), u))
    jit = [((1 - c.height_jitter) * h, (1 + c.height_jitter) * h) for _, h in sa["supports"]]
    supports = list(zip(s, _heights(rng, c, s, clamp, jit)))
    sa["folds"] = [f for f in sa["folds"] if all(               # fold lines are shared by both ends:
        f[0] <= min(a, b) - c.fold_min_gap or f[0] >= max(a, b) + c.fold_min_gap   # clear of each support
        for (a, _), (b, _) in zip(sa["supports"], supports))]                        # line as it turns
    budget = max(np.radians(c.fold_total_deg_supported if supports else c.fold_total_deg) - (clamp or 0.0), 0.0)
    lo_a, hi_a = np.radians(c.fold_angle_deg)
    folds = [(p, np.sign(t) * np.clip(abs(t) * (1 + c.fold_side_b_angle_jitter * rng.uniform(-1, 1)), lo_a, hi_a), k)
             for p, t, k in sa["folds"]]
    f = min(1.0, budget / max(sum(abs(t) for _, t, _ in folds), 1e-12))
    return dict(clamp=clamp, supports=supports, folds=[(p, t * f, k) for p, t, k in folds], flat=sa["flat"])


def lerp_scene(a, b, t):
    lerp = lambda x, y: (1 - t) * x + t * y
    return dict(flat=a["flat"], clamp=None if a["clamp"] is None else lerp(a["clamp"], b["clamp"]),
                supports=[(lerp(s0, s1), lerp(h0, h1)) for (s0, h0), (s1, h1) in zip(a["supports"], b["supports"])],
                folds=[(s0, lerp(t0, t1), k0) for (s0, t0, k0), (_, t1, _) in zip(a["folds"], b["folds"])])


def scene_label(sc):
    return "+".join(filter(None, ["flat" if sc.get("flat") else "", "clamp" if sc["clamp"] is not None else "",
                                  f"{len(sc['supports'])}sup" if sc["supports"] else "",
                                  f"{len(sc['folds'])}fold" if sc["folds"] else ""]))


def sample_paper(rng, c: SceneConfig, base):
    g = rng.uniform(*c.grammage_gsm)
    r = g / base.grammage_gsm
    return replace(base, grammage_gsm=g, bending_stiffness=base.bending_stiffness * r ** c.stiffness_exp,
                   membrane_stiffness=base.membrane_stiffness * r)
