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


def _bend_cdf(g):
    """(directions [deg], CDF) of the bend direction density cos(2 dir) + bend_floor over +-bend_dir_deg."""
    t = np.linspace(-g.bend_dir_deg, g.bend_dir_deg, 1001)
    F = 0.5 * np.sin(np.radians(2 * t)) + g.bend_floor * np.radians(t)
    return t, (F - F[0]) / (F[-1] - F[0])


def factors(P: Preset):
    """name -> ((value, weight), ...) for geometry, view and layout."""
    g, s = P.geometry, P.geometry.scene
    t, F = _bend_cdf(g)
    bend = tuple((b, float(np.diff(np.interp(b, t, F))[0])) for b in _bins(-g.bend_dir_deg, g.bend_dir_deg, 5))
    li, m = P.render.light, P.layout.margin_frac
    return dict(
        combo=tuple((c, w) for c, w in s.combos),
        two=((True, g.two_profile_prob), (False, 1 - g.two_profile_prob)),
        along_long=((True, g.along_long_prob), (False, 1 - g.along_long_prob)),
        bend_bin=bend,
        gsm=tuple((v, 1.0) for v in P.paper.gsm),
        shallow=((True, g.shallow.prob), (False, 1 - g.shallow.prob)),
        deep=g.deep.levels,
        bend_deep=((True, g.deep.on_bends_prob), (False, 1 - g.deep.on_bends_prob)),
        sharp=((True, li.sharp_prob), (False, 1 - li.sharp_prob)),
        frame=P.render.camera.frames,
        margin_bin=tuple((b, 1.0) for b in _bins(*m, 3)) if m[1] > m[0] else ((m, 1.0),))


def _canonical(c):
    crumple = c["combo"][3]
    out = {k: v for k, v in c.items() if k != "bend_deep"}
    return dict(out, two=c["two"] and not crumple,
                deep=c["deep"] if crumple else "light" if c.get("bend_deep") else "")


GROUPS = dict(geometry=("combo", "two", "along_long", "bend_bin", "gsm", "shallow", "deep", "bend_deep"),
              view=("sharp", "frame"), layout=("margin_bin",))


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


def open_scale(c):
    """Size of an open book's page on its sheet relative to the same page on a whole sheet: half the landscape sheet."""
    return c.sheet_mm[1] / 2 / c.sheet_mm[0]


def plan(P: Preset, n, seed, manifest):
    """Up to n sample specs with exact factor marginals, script groups included. Groups of use_all take all their
    pages; script_mix shares the rest of n. Pages are never repeated: a group short of pages stays under its quota,
    its samples dropped. Open books (a landscape spread, the spine across the width) take restyled pages only: a group
    short of them trades its open combos for other combos of groups with restyled pages to spare. Restyled pages get a
    body size (font_pt) spread evenly over font_pt, open books over open_font_pt, by the rank of their own (open book
    pages on half the sheet): every page as near its source size as the even spread allows. Book pages bend with the
    spine on the long side (bend direction over the whole range)."""
    rng = np.random.default_rng(seed)
    c = P.layout
    pools = entry_pools(P, manifest)
    whole = {g: len(pools[g]) for g in c.use_all if pools.get(g)}
    rest = max(0, n - sum(whole.values()))
    share = [(g, w) for g, w in c.script_mix if g not in whole and pools.get(g)]
    wsum = sum(w for _, w in share)
    mix = tuple((g, float(v)) for g, v in whole.items()) + tuple((g, rest * w / wsum) for g, w in share)
    cols = {k: _exact(rng, v, n) for k, v in (*factors(P).items(), ("script", mix))}
    rows = {g: np.flatnonzero(np.array(cols["script"]) == g)[:len(pools[g])] for g, _ in mix}
    entry = np.full(n, -1)
    refont = {g: sum(manifest[e]["kind"] == "refont" for e in pools[g]) for g in rows}
    opened = np.array([cb[0] == "open" for cb in cols["combo"]])
    excess = [i for g, idx in rows.items() for i in idx[opened[idx]][refont[g]:]]
    room = [i for g, idx in rows.items() for i in idx[~opened[idx]][:max(refont[g] - opened[idx].sum(), 0)]]
    for a, b in zip(excess, rng.permutation(room)):
        cols["combo"][a], cols["combo"][b] = cols["combo"][b], cols["combo"][a]
    opened = np.array([cb[0] == "open" for cb in cols["combo"]])
    for g, idx in rows.items():
        perm = rng.permutation(pools[g]).astype(int)
        o = perm[[manifest[e]["kind"] == "refont" for e in perm]][:opened[idx].sum()]
        e = perm[~np.isin(perm, o)][:(~opened[idx]).sum()]
        entry[idx[opened[idx]][:len(o)]] = o
        entry[idx[~opened[idx]][:len(e)]] = e
    t, F = _bend_cdf(P.geometry)
    font = np.full(n, np.nan)
    for sel, (lo, hi), k in ((opened, c.open_font_pt, open_scale(c)), (~opened, c.font_pt, 1.0)):
        idx = np.flatnonzero(sel & (entry >= 0))
        idx = idx[[manifest[entry[i]]["kind"] == "refont" for i in idx]]
        rank = np.argsort(np.argsort([manifest[entry[i]]["body_pt"] * k + rng.random() * 1e-6 for i in idx]))
        font[idx] = lo + (rank + rng.random(len(idx))) / max(len(idx), 1) * (hi - lo)
    specs = []
    for i in np.flatnonzero(entry >= 0):
        s = _canonical({k: cols[k][i] for k in cols})
        book = s["combo"][0] == "book"
        b = np.interp(s.pop("bend_bin"), t, F)
        s["bend_dir"] = float(np.interp(rng.uniform(*((0.0, 1.0) if book else b)), F, t))
        if book or opened[i]:
            s["along_long"] = False
        if opened[i]:
            s["bend_dir"] = 0.0
            s["gsm"] = float(rng.uniform(*P.paper.open_gsm))
        s["entry"] = int(entry[i])
        if np.isfinite(font[i]):
            s["font_pt"] = float(font[i])
        specs.append(s)
    _water(P.geometry.water, specs, seed)
    return specs


def _water(w, specs, seed):
    """round(prob * len(specs)) water-damaged pages, drawn among the pages without creases (no crumple, shallow or
    deep creases); own streams, so the other factors stay as without water."""
    ok = [s for s in specs if not (s["combo"][3] or s["shallow"] or s["deep"])]
    k = min(int(round(w.prob * len(specs))), len(ok))
    for s in np.random.default_rng([seed, 1]).permutation(np.array(ok, dtype=object))[:k]:
        s["water"] = float(np.random.default_rng([seed, 2, s["entry"]]).uniform(*w.scale))


def geometry_cond(spec):
    return {k: spec[k] for k in ("combo", "two", "along_long", "bend_dir", "gsm", "shallow", "deep", "water") if k in spec}


def sample_preset(P: Preset, spec):
    """The preset of a sample: its light, frame (restyled pages whose frame cuts the page get cut_margin_frac) and layout bins;
    an open book on the sheet turned landscape (two pages)."""
    r, c, g = P.render, P.layout, P.geometry
    flip = (lambda t: t[::-1]) if spec["combo"][0] == "open" else (lambda t: t)
    frame = spec.get("frame", "page")
    return replace(P, render=replace(r, light=replace(r.light, sharp_prob=float(spec["sharp"])), camera=replace(r.camera, frame=frame)),
                   layout=replace(c, margin_frac=c.cut_margin_frac if frame != "page" else spec["margin_bin"],
                                  sheet_mm=flip(c.sheet_mm), page_px=flip(c.page_px)),
                   geometry=replace(g, sheet_mm=flip(g.sheet_mm), mesh=flip(g.mesh)))
