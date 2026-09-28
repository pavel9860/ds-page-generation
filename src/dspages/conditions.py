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

from .config import Preset, page_px
from .layout.content import window_limits
from .layout.sources import is_pdf


def _bins(lo, hi, n, log=False):
    e = np.exp(np.linspace(np.log(lo), np.log(hi), n + 1)) if log else np.linspace(lo, hi, n + 1)
    return tuple((float(a), float(b)) for a, b in zip(e[:-1], e[1:]))


def factors(P: Preset):
    """name -> ((value, weight), ...) for geometry, view and layout."""
    g, s = P.geometry, P.geometry.scene
    bend = _bins(-g.bend_dir_deg, g.bend_dir_deg, 5)
    cam, li, m = P.render.camera, P.render.light, P.layout.margin_frac
    return dict(
        combo=tuple((c, w) for c, w in s.combos),
        flat=((True, s.near_flat_prob), (False, 1 - s.near_flat_prob)),
        two=((True, g.two_profile_prob), (False, 1 - g.two_profile_prob)),
        along_long=((True, g.along_long_prob), (False, 1 - g.along_long_prob)),
        bend_bin=tuple((b, 1.0) for b in bend),
        gsm=tuple((v, 1.0) for v in P.paper.gsm),
        shallow=((True, g.shallow.prob), (False, 1 - g.shallow.prob)),
        deep=g.deep.levels,
        dist_bin=tuple((b, 1.0) for b in _bins(*cam.dist_mm, 4, log=True)),
        tilt_bin=tuple((b, 1.0) for b in _bins(*cam.tilt_deg, 3)),
        sharp=((True, li.sharp_prob), (False, 1 - li.sharp_prob)),
        margin_bin=tuple((b, 1.0) for b in _bins(*m, 3)) if m[1] > m[0] else ((m, 1.0),))


def _canonical(c):
    crumple = c["combo"][3]
    return dict(c, flat=c["flat"] and not crumple, two=c["two"] and not crumple, deep=c["deep"] if crumple else "")


GROUPS = dict(geometry=("combo", "flat", "two", "along_long", "bend_bin", "gsm", "shallow", "deep"),
              view=("dist_bin", "tilt_bin", "sharp"), layout=("margin_bin",))


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


def entry_pools(P: Preset, manifest):
    """Per script group: manifest indices of raster pages and of book texts. Raster pages count only when a window
    of the sheet's aspect around their content (margins <= margin_frac[1]) reaches the sheet width at 1:1:
    directly for images, within max_zoom for PDF pages."""
    c = P.layout
    pw, ph = page_px(c.sheet_mm, c.canvas_px)
    pools = {g: ([], []) for g, _ in c.script_mix}
    for i, e in enumerate(manifest):
        g = pools.setdefault(script(e, c.scripts), ([], []))
        if e["kind"] == "book_text":
            g[1].append(i)
        elif window_limits(e["bbox_w"], e["bbox_h"], pw / ph, c.margin_frac[1])[1] * (
                c.max_zoom if is_pdf(e) else 1.0) >= pw:
            g[0].append(i)
    return pools


def _counts(w, n):
    """Integer counts with sum n in proportion to w (largest remainder)."""
    p = np.asarray(w, float) / np.sum(w)
    k = np.floor(n * p).astype(int)
    k[np.argsort(-(n * p - k))[:n - k.sum()]] += 1
    return k


def _fill(rng, raster, books, q, chars):
    """q (entry, text offset) pairs: raster pages first, never repeated, then book snippets for the rest, spread over
    the books by their length at evenly spaced offsets from a random phase (distinct text while a book holds
    them). A group short of both stays under its quota."""
    r = rng.permutation(raster)[:q].astype(int)
    e, o = [r], [np.zeros(len(r), int)]
    if q > len(r) and len(books):
        for b, k in zip(books, _counts([chars[b] for b in books], q - len(r))):
            e.append(np.full(k, b))
            o.append((rng.integers(chars[b]) + np.arange(k) * chars[b] // max(k, 1)) % chars[b])
    return np.concatenate(e), np.concatenate(o)


def plan(P: Preset, n, seed, manifest):
    """Up to n sample specs with exact factor marginals, script groups included. Groups of use_all take all their
    raster pages; script_mix shares the rest of n. A group's entries are its raster pages (never repeated), then
    book snippets for the rest of its quota; groups without either stay under quota, their samples dropped."""
    rng = np.random.default_rng(seed)
    c = P.layout
    pools = entry_pools(P, manifest)
    whole = {g: len(pools[g][0]) for g in c.use_all if g in pools and len(pools[g][0])}
    rest = max(0, n - sum(whole.values()))
    share = [(g, w) for g, w in c.script_mix if g not in whole and any(pools.get(g, ((), ())))]
    wsum = sum(w for _, w in share)
    mix = tuple((g, float(v)) for g, v in whole.items()) + tuple((g, rest * w / wsum) for g, w in share)
    cols = {k: _exact(rng, v, n) for k, v in (*factors(P).items(), ("script", mix))}
    rows = {g: np.flatnonzero(np.array(cols["script"]) == g) for g, _ in mix}
    chars = {i: manifest[i].get("chars", 1) for _, b in pools.values() for i in b}
    entry, offset = np.full(n, -1), np.zeros(n, int)
    for g, idx in rows.items():
        e, o = _fill(rng, *pools[g], len(idx), chars)
        perm = rng.permutation(len(e))
        entry[idx[:len(e)]], offset[idx[:len(e)]] = e[perm], o[perm]
    specs = []
    for i in np.flatnonzero(entry >= 0):
        s = _canonical({k: cols[k][i] for k in cols})
        s["bend_dir"] = float(rng.uniform(*s.pop("bend_bin")))
        s["entry"], s["offset"] = int(entry[i]), int(offset[i])
        specs.append(s)
    return specs


def geometry_cond(spec):
    return {k: spec[k] for k in ("combo", "flat", "two", "along_long", "bend_dir", "gsm", "shallow", "deep")}


def view_preset(P: Preset, spec):
    """The preset with camera and light narrowed to the spec's bins."""
    r = P.render
    return replace(P, render=replace(r, camera=replace(r.camera, dist_mm=spec["dist_bin"], tilt_deg=spec["tilt_bin"]),
                                     light=replace(r.light, sharp_prob=float(spec["sharp"]))))


def layout_preset(P: Preset, spec):
    return replace(P, layout=replace(P.layout, margin_frac=spec["margin_bin"]))
