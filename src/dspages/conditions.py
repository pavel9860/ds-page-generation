"""Condition space of a preset and stratified plans over it.

Every discrete factor (and every continuous one cut into bins) is listed with its probability. Cells are the
combinations of the factors of a group (geometry, view, layout), impossible ones folded into their canonical
form. A plan of n samples gives every factor value round(n p) samples by largest remainder and pairs the factors
by independent shuffles (Latin hypercube): exact marginals for any n, no sampling retries. Continuous values are
drawn within their bin per sample.
"""
import itertools
import json
from dataclasses import replace

import numpy as np

from .config import Preset


def _bins(lo, hi, n, log=False):
    e = np.exp(np.linspace(np.log(lo), np.log(hi), n + 1)) if log else np.linspace(lo, hi, n + 1)
    return tuple((float(a), float(b)) for a, b in zip(e[:-1], e[1:]))


def factors(P: Preset):
    """name -> ((value, weight), ...) for geometry, view and layout."""
    g, s = P.geometry, P.geometry.scene
    bend = _bins(-g.bend_dir_deg, g.bend_dir_deg, 5)
    li, m, lc = P.render.light, P.layout.margin_frac, P.layout
    return dict(
        combo=tuple((c, w) for c, w in s.combos),
        flat=((True, s.near_flat_prob), (False, 1 - s.near_flat_prob)),
        two=((True, g.two_profile_prob), (False, 1 - g.two_profile_prob)),
        along_long=((True, g.along_long_prob), (False, 1 - g.along_long_prob)),
        bend_bin=tuple((b, 1.0) for b in bend),
        gsm=tuple((v, 1.0) for v in P.paper.gsm),
        shallow=((True, g.shallow.prob), (False, 1 - g.shallow.prob)),
        deep=g.deep.levels,
        bend_deep=((True, g.deep.on_bends_prob), (False, 1 - g.deep.on_bends_prob)),
        sharp=((True, li.sharp_prob), (False, 1 - li.sharp_prob)),
        margin_bin=tuple((b, 1.0) for b in _bins(*m, 3)) if m[1] > m[0] else ((m, 1.0),),
        font_bin=tuple((b, 1.0) for b in _bins(*lc.font_pt, 4)),
        scale_bin=tuple((b, 1.0) for b in _bins(*lc.role_scale, 3)))


def _canonical(c):
    crumple = c["combo"][3]
    out = {k: v for k, v in c.items() if k != "bend_deep"}
    return dict(out, flat=c["flat"] and not crumple, two=c["two"] and not crumple,
                deep=c["deep"] if crumple else "light" if c.get("bend_deep") else "")


GROUPS = dict(geometry=("combo", "flat", "two", "along_long", "bend_bin", "gsm", "shallow", "deep", "bend_deep"),
              view=("sharp",), layout=("margin_bin", "font_bin", "scale_bin"))


def cells(P: Preset, group):
    """[(cell dict, probability)] of one factor group, impossible cells merged into canonical ones."""
    f = {k: v for k, v in factors(P).items() if k in GROUPS[group]}
    names = list(f)
    merged = {}
    for combo in itertools.product(*(f[k] for k in names)):
        c = {k: v for k, (v, _) in zip(names, combo)}
        c = _canonical(c) if group == "geometry" else c
        p = float(np.prod([w / sum(x for _, x in f[k]) for k, (_, w) in zip(names, combo)]))
        key = json.dumps(c, sort_keys=True)
        merged[key] = (c, merged.get(key, (c, 0.0))[1] + p)
    return list(merged.values())


def _exact(rng, table, n):
    """n values with the table's proportions exactly (largest remainder), shuffled."""
    k = _counts([x for _, x in table], n)
    vals = [v for (v, _), m in zip(table, k) for _ in range(m)]
    return [vals[i] for i in rng.permutation(n)]


def script(entry, scripts):
    return next((g for g, langs in scripts.items() if entry["language"] in langs), "latin")


def _counts(w, n):
    """Integer counts with sum n in proportion to w (largest remainder)."""
    p = np.asarray(w, float) / np.sum(w)
    k = np.floor(n * p).astype(int)
    k[np.argsort(-(n * p - k))[:n - k.sum()]] += 1
    return k


def entry_pools(P: Preset, manifest):
    """Per script group: manifest indices of its pages (restyled and raster)."""
    pools = {g: [] for g, _ in P.layout.script_mix}
    for i, e in enumerate(manifest):
        pools.setdefault(script(e, P.layout.scripts), []).append(i)
    return pools


def plan(P: Preset, n, seed, manifest):
    """Up to n sample specs with exact factor marginals, script groups included. Groups of use_all take all their
    pages; script_mix shares the rest of n. Pages are never repeated: a group short of pages stays under its quota,
    its samples dropped."""
    rng = np.random.default_rng(seed)
    c = P.layout
    pools = entry_pools(P, manifest)
    whole = {g: len(pools[g]) for g in c.use_all if pools.get(g)}
    rest = max(0, n - sum(whole.values()))
    share = [(g, w) for g, w in c.script_mix if g not in whole and pools.get(g)]
    wsum = sum(w for _, w in share)
    mix = tuple((g, float(v)) for g, v in whole.items()) + tuple((g, rest * w / wsum) for g, w in share)
    cols = {k: _exact(rng, v, n) for k, v in (*factors(P).items(), ("script", mix))}
    rows = {g: np.flatnonzero(np.array(cols["script"]) == g) for g, _ in mix}
    entry = np.full(n, -1)
    for g, idx in rows.items():
        e = rng.permutation(pools[g])[:len(idx)]
        entry[idx[:len(e)]] = e
    specs = []
    for i in np.flatnonzero(entry >= 0):
        s = _canonical({k: cols[k][i] for k in cols})
        s["bend_dir"] = float(rng.uniform(*s.pop("bend_bin")))
        if s["combo"][0] == "book":
            sc = P.geometry.scene
            s["bend_dir"] = float(rng.uniform(-sc.book_bend_deg, sc.book_bend_deg))
            s["along_long"] = bool(rng.random() < sc.book_along_long_prob)
        s["entry"] = int(entry[i])
        specs.append(s)
    return specs


def geometry_cond(spec):
    return {k: spec[k] for k in ("combo", "flat", "two", "along_long", "bend_dir", "gsm", "shallow", "deep")}


def view_preset(P: Preset, spec):
    """The preset with the light of the spec."""
    r = P.render
    return replace(P, render=replace(r, light=replace(r.light, sharp_prob=float(spec["sharp"]))))


def layout_preset(P: Preset, spec):
    return replace(P, layout=replace(P.layout, margin_frac=spec["margin_bin"], font_pt=spec["font_bin"],
                                     role_scale=spec["scale_bin"]))
