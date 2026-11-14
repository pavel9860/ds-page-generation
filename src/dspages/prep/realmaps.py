"""Real 3D page maps (GridPages/3d_maps: a printed grid page, its nodes found in a photo and lifted to 3D) as page geometry
for the generator: X, Y, Z [mm] on the (256, 360) page grid, the whole sheet, in the frame of dspages.geometry (rows down
the sheet, X across, Y up, Z toward the viewer, table at Z = 0, sheet 210 x 297 mm).

Source (prepare): every detected node of the CSV lattice is kept; missing nodes are put between them (linear, straight on
past the border). Rows and columns are moved along themselves to equal steps (`even`: alternating passes as the generator's
ISO_PASSES, a node shifts along the curve it is on, so the surface stays), the lattice is cut at (256, 360) nodes by a cubic
spline and put in the page frame, sheet-sized. Sources whose shapes match (shots of one page) form a group; the split into
train / val / test is made by group.
Variants (sample): flip, half turn, crop with a turn of 5-15 deg or a slide of 5-15 %, a 1.5x zoom for the water-damaged
pages; the crop is cut at (256, 360) nodes again (the cropped page is the whole sheet: the surface is scaled up with it).

python -m dspages.prep.realmaps [--src DIR] [--out DIR] [--total 10000] [--workers 6] [--plain] [--preview]

OUT/maps/m<source>_<variant>.npz  grid3d (3, 360, 256) float32: X, Y, Z [mm]   (source numbers are shuffled: no category or shot time)
OUT/maps.jsonl                    id, source, group, split, aug (flip, half, mode, theta, scale, centre)
OUT/sources.jsonl                 source, group, split, variants, category, stem, qc; OUT/sources/s<source>.npz the unaugmented grid3d
OUT/qc.json                       checks of the written maps (files, leakage between splits, step uniformity)
"""
import argparse
import csv
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator

from ..config import UVTM

SRC = Path(f"{UVTM}/GridPages/3d_maps")
OUT = Path(f"{UVTM}/GridPages/maps3d_256x360")
SHEET = (210.0, 297.0)                     # mm (width, height)
MESH = (256, 360)                          # nodes (across, down)
ROUNDS = 4                                 # column and row passes of `even`
ROT_DEG, SLIDE, ZOOM = (5.0, 15.0), (0.05, 0.15), (1.4, 1.6)
DUP_MM = 1.0                               # rms height difference [mm] under which two shapes are one group
SPLIT = (("val", 0.05), ("test", 0.05))
RARE = ("Book_v1.1.0", "Book_water1.1.0")  # few sources: they get the most variants
ZOOMED = ("Book_water1.1.0",)


def load(path):
    """(R, C, 3) nodes [grid units, camera frame] and the valid mask of one CSV."""
    with open(path) as f:
        rd = csv.reader(f)
        head = next(rd)
        d = np.array([[float(x) if x else np.nan for x in row] for row in rd]).reshape(-1, len(head))
    d = {k: d[:, i] for i, k in enumerate(head)}
    g = np.isfinite(d["row"]) & np.isfinite(d["col"])                   # nodes without a grid index: left as holes
    d = {k: v[g] for k, v in d.items()}
    r, c = d["row"].astype(int), d["col"].astype(int)
    P = np.full((r.max() + 1, c.max() + 1, 3), np.nan)
    P[r, c] = np.stack([d["X"], d["Y"], d["Z"]], -1)
    ok = np.zeros(P.shape[:2], bool)
    ok[r, c] = (d["kept"] > 0) & np.isfinite(d["Z"]) & (d["Z"] > 1.0)
    return P, ok


def frame(P, camera=None):
    """P (R, C, 3) in the page frame: Z along the page normal (mean cross of the row and column directions), X along the
    columns, Y up, rows going down; shifted to X, Y, Z >= 0. With `camera` (a point), the columns are reversed first when the
    page is seen from behind."""
    def axes(P):
        du, dv = np.gradient(P, axis=1), np.gradient(P, axis=0)
        n = np.cross(dv, du).mean((0, 1))
        return n / np.linalg.norm(n), du.mean((0, 1))
    n, du = axes(P)
    if camera is not None and n @ (camera - P.mean((0, 1))) < 0:
        P = P[:, ::-1]
        n, du = axes(P)
    ex = du - (du @ n) * n
    ex /= np.linalg.norm(ex)
    Q = (P - P.mean((0, 1))) @ np.stack([ex, np.cross(n, ex), n], 1)
    return Q - Q.min((0, 1))


def even(V, d, axis):
    """V (H, W, 3) with every curve along `axis` moved along itself to steps of d (anchored where the nodes move least;
    past the curve's ends it goes straight on). The curve keeps its shape: nodes shift along it only."""
    V = V if axis == 1 else V.transpose(1, 0, 2)
    out, j = np.empty_like(V), np.arange(V.shape[1])
    for i, q in enumerate(V):
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
        t = j * d + np.mean(s - j * d)
        out[i] = np.stack([np.interp(t, s, q[:, k]) for k in range(3)], -1)
        lo, hi = t < 0.0, t > s[-1]
        tl, th = q[1] - q[0], q[-1] - q[-2]
        out[i][lo] = q[0] + t[lo][:, None] * tl / np.linalg.norm(tl)
        out[i][hi] = q[-1] + (t[hi] - s[-1])[:, None] * th / np.linalg.norm(th)
    return out if axis == 1 else out.transpose(1, 0, 2)


def fill(P, ok):
    """P with its missing nodes on a linear interpolation of the detected ones; outside their hull, the nearest detected
    node plus the index offset times the mean row and column steps."""
    V = P.copy()
    if ok.all():
        return V
    rc, miss = np.stack(np.nonzero(ok), 1), np.stack(np.nonzero(~ok), 1)
    lin = LinearNDInterpolator(rc, P[ok])(miss)
    out = np.isnan(lin[:, 0])
    pair = [ok[1:] & ok[:-1], ok[:, 1:] & ok[:, :-1]]
    step = [np.diff(P, axis=a)[pair[a]].mean(0) for a in (0, 1)]
    near = cKDTree(rc).query(miss[out])[1]
    off = miss[out] - rc[near]
    lin[out] = P[ok][near] + off[:, :1] * step[0] + off[:, 1:] * step[1]
    V[~ok] = lin
    return V


def spaced(G, axis):
    """Curves of G (H, W, 3) along `axis` walked to equal arc-length steps (each curve keeps its shape and its length)."""
    Q = np.moveaxis(G, axis, 1)
    out = np.empty_like(Q)
    for i, q in enumerate(Q):
        t = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=-1))]
        u = np.linspace(0.0, t[-1], len(q))
        out[i] = np.stack([np.interp(u, t, q[:, k]) for k in range(3)], -1)
    return np.moveaxis(out, 1, axis)


def uniform(G, passes=3):
    """G (H, W, 3) [any scale] with equal steps along every row and column (row and column passes, as in the generator's
    ISO_PASSES), scaled so that the mean row is the sheet width and the mean column its height, then in the page frame."""
    for _ in range(passes):
        G = spaced(spaced(G, 1), 0)
    w = np.linalg.norm(np.diff(G, axis=1), axis=-1).sum(1).mean()
    h = np.linalg.norm(np.diff(G, axis=0), axis=-1).sum(0).mean()
    G = G * np.mean([SHEET[0] / w, SHEET[1] / h])
    return frame(G)


def step_error(G):
    """Largest and 99th percentile |step / nominal step - 1| of the (360, 256) grid G, rows and columns."""
    e = np.concatenate([(np.linalg.norm(np.diff(G, axis=1), axis=-1) / (SHEET[0] / (MESH[0] - 1)) - 1).ravel(),
                        (np.linalg.norm(np.diff(G, axis=0), axis=-1) / (SHEET[1] / (MESH[1] - 1)) - 1).ravel()])
    return float(np.abs(e).max()), float(np.percentile(np.abs(e), 99))


def prepare(args):
    """(sources row, base page grid (360, 256, 3) [mm]) of one CSV (src, category, stem)."""
    src, cat, stem = args
    P, ok = load(Path(src) / cat / "csv" / f"{stem}.csv")
    if P.shape[0] < P.shape[1]:
        P, ok = np.rot90(P), np.rot90(ok)
    H, W = ok.shape
    unit = np.mean([SHEET[0] / (W - 1), SHEET[1] / (H - 1)])           # mm per grid unit: the nodes span the sheet
    V = fill(np.where(ok[..., None], P, 0.0) * unit, ok)
    for _ in range(ROUNDS):
        V = even(even(V, unit, 0), unit, 1)
    V = frame(V, np.zeros(3))
    rr, cc = np.meshgrid(np.linspace(0, H - 1, MESH[1]), np.linspace(0, W - 1, MESH[0]), indexing="ij")
    G = uniform(np.stack([ndimage.map_coordinates(V[..., k], [rr, cc], order=3) for k in range(3)], -1), 1)
    err = step_error(G)
    return dict(category=cat, stem=stem, missing=float(1 - ok.mean()), step_err=err, relief_mm=float(G[..., 2].max())), G.astype(np.float32)


def sample(G, aug):
    """(3, 360, 256) float32 page grid of base grid G (360, 256, 3) [mm] under `aug`: flip, half (turn), theta [deg], scale
    (side of the crop / side of the page), centre (offsets as a share of the page, within the slack). The crop is the whole
    sheet again: the surface is scaled up with it and given equal steps."""
    G = G.astype(np.float64)
    if aug["flip"]:
        G = (G * np.array([-1.0, 1.0, 1.0]))[:, ::-1]
    if aug["half"]:
        G = G[::-1, ::-1]
    G = frame(G)
    R, C = G.shape[:2]
    t, s = np.radians(aug["theta"]), aug["scale"]
    if s < 1.0 or t:
        w, h = s * (C - 1), s * (R - 1)
        i, j = np.meshgrid(np.linspace(-0.5, 0.5, MESH[0]) * w, np.linspace(-0.5, 0.5, MESH[1]) * h)
        col = (C - 1) / 2 + aug["centre"][0] * (C - 1) + i * np.cos(t) - j * np.sin(t)
        row = (R - 1) / 2 + aug["centre"][1] * (R - 1) + i * np.sin(t) + j * np.cos(t)
        G = np.stack([ndimage.map_coordinates(G[..., k], [row, col], order=3, mode="nearest") for k in range(3)], -1)
    return np.moveaxis(uniform(G, 2), -1, 0).astype(np.float32)


def max_scale(R, C, t):
    """Largest crop side (share of the page) at a turn t [rad] that stays inside the grid."""
    t = abs(t)
    return min((C - 1) / ((C - 1) * np.cos(t) + (R - 1) * np.sin(t)), (R - 1) / ((C - 1) * np.sin(t) + (R - 1) * np.cos(t)))


def draw_aug(rng, mode, R, C):
    flip, half = bool(rng.integers(2)), bool(rng.integers(2))
    a = dict(flip=flip, half=half, mode=mode, theta=0.0, scale=1.0, centre=[0.0, 0.0])
    if mode == "rot":
        a["theta"] = float(rng.choice([-1, 1]) * rng.uniform(*ROT_DEG))
        a["scale"] = float(max_scale(R, C, np.radians(a["theta"])))
    elif mode in ("slide", "zoom"):
        a["scale"] = float(1 - rng.uniform(*SLIDE)) if mode == "slide" else float(1 / rng.uniform(*ZOOM))
        if mode == "zoom" and rng.random() < 0.5:
            a["theta"] = float(rng.choice([-1, 1]) * rng.uniform(*ROT_DEG))
            a["scale"] = min(a["scale"], float(max_scale(R, C, np.radians(a["theta"]))))
        slack = (1 - a["scale"]) / 2 if a["theta"] == 0 else 0.0
        if a["theta"]:
            t = abs(np.radians(a["theta"]))
            w, h = a["scale"] * (C - 1), a["scale"] * (R - 1)
            slack_xy = [((C - 1) - (w * np.cos(t) + h * np.sin(t))) / 2 / (C - 1), ((R - 1) - (w * np.sin(t) + h * np.cos(t))) / 2 / (R - 1)]
        else:
            slack_xy = [slack, slack]
        a["centre"] = [float(rng.uniform(-v, v)) for v in slack_xy]
    return a


def variants(rng, cat, n, R, C):
    """n distinct (flip, half, mode) variants of a source; the first is the page itself."""
    modes = ["full", "rot", "slide"] + (["zoom"] if cat in ZOOMED else [])
    combos = [(f, h, m) for m in modes for f in (False, True) for h in (False, True)]
    out = []
    for f, h, m in [combos[0]] + [combos[k + 1] for k in rng.permutation(len(combos) - 1)[: n - 1]]:
        out.append({**draw_aug(rng, m, R, C), "flip": f, "half": h})
    return out


def groups(grids, mm=DUP_MM):
    """Group number per source: shapes that match under the flips and half turn within `mm` (rms height) are one group."""
    def desc(P):
        z = ndimage.zoom(P[..., 2], (24 / P.shape[0], 17 / P.shape[1]), order=1)
        return z - z.mean()
    D = [desc(P) for P in grids]
    V = np.stack([np.stack([d, d[:, ::-1], d[::-1], d[::-1, ::-1]]).ravel() for d in D]).reshape(len(D), 4, -1)
    base = np.stack(D).reshape(len(D), -1)
    parent = list(range(len(D)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a in range(len(D)):
        dist = np.sqrt(((V[a + 1:] - base[a]) ** 2).mean(-1)).min(1)
        for b in np.where(dist < mm)[0] + a + 1:
            parent[find(b)] = find(a)
    roots = {}
    return [roots.setdefault(find(a), len(roots)) for a in range(len(D))]


def write(args):
    out, sid, k, aug, G = args
    np.savez(out / "maps" / f"m{sid:04d}_{k:02d}.npz", grid3d=sample(G, aug))


def split_groups(grp, rng):
    """Split name per source: val and test take whole groups (random order) until each has its share of the sources."""
    size, order, split, k = np.bincount(grp), rng.permutation(max(grp) + 1), {}, 0
    for name, share in SPLIT:
        got = 0
        while got < share * len(grp):
            split[order[k]] = name
            got += size[order[k]]
            k += 1
    return [split.get(g, "train") for g in grp]


def allocate(cats, total, rare_cap, rng):
    """Variants per source: rare categories rare_cap each, the rest share what is left of `total` evenly."""
    rare = np.array([c in RARE for c in cats])
    per = np.where(rare, rare_cap, 0)
    rest, idx = total - per.sum(), np.flatnonzero(~rare)
    per[idx] = rest // len(idx)
    per[rng.permutation(idx)[: rest % len(idx)]] += 1
    return per


def check(out, maps, srcs):
    """Checks of the written maps -> OUT/qc.json: every file there, finite, (3, 360, 256); a source and a group in one split
    only; names carry no category or stem; step uniformity over the maps."""
    bad, errs = [], []
    for m in maps:
        G = np.moveaxis(np.load(out / "maps" / f"{m['id']}.npz")["grid3d"], 0, -1)
        if G.shape != (MESH[1], MESH[0], 3) or not np.isfinite(G).all():
            bad.append(m["id"])
        errs.append(step_error(G))
    errs = np.array(errs)
    leak = {k: sum(len({s["split"] for s in srcs if s[k] == v}) > 1 for v in {s[k] for s in srcs}) for k in ("source", "group")}
    cats = {s["category"] for s in srcs} | {s["stem"] for s in srcs}
    qc = dict(maps=len(maps), sources=len(srcs), groups=len({s["group"] for s in srcs}), bad_files=bad,
              split_maps={n: sum(m["split"] == n for m in maps) for n in ("train", "val", "test")},
              split_sources={n: sum(s["split"] == n for s in srcs) for n in ("train", "val", "test")},
              leaks_across_splits=leak, names_with_category_or_stem=sum(any(c in m["id"] for c in cats) for m in maps),
              maps_with_every_step_within_1pct=float((errs[:, 0] < 0.01).mean()), maps_p99_step_within_1pct=float((errs[:, 1] < 0.01).mean()),
              maps_with_every_step_within_3pct=float((errs[:, 0] < 0.03).mean()),
              maps_with_filled_nodes=float(np.mean([s["missing"] > 0 for s in srcs])),
              sources_over_5pct_missing=sum(s["missing"] > 0.05 for s in srcs))
    (out / "qc.json").write_text(json.dumps(qc, indent=1))
    return qc


def preview(out, rows, n=48):
    import cv2
    tiles = []
    for r in rows[:n]:
        Z = np.load(out / "maps" / f"{r['id']}.npz")["grid3d"][2]
        img = cv2.applyColorMap(np.clip(Z / max(Z.max(), 1) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        tiles.append(cv2.putText(cv2.resize(img, (128, 180)), f"{Z.max():.0f}mm {r['aug']['mode']}", (2, 12), 0, 0.4, (255, 255, 255), 1))
    cols = 12
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    cv2.imwrite(str(out / "preview.jpg"), np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--total", type=int, default=10000)
    ap.add_argument("--rare-cap", type=int, default=12, help="variants per source of the rare categories")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plain", action="store_true", help="one unaugmented map per source")
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()
    out, rng = Path(a.out), np.random.default_rng(a.seed)
    for d in ("maps", "sources"):
        (out / d).mkdir(parents=True, exist_ok=True)
    items = sorted((a.src, r["category"], r["stem"]) for r in csv.DictReader(open(Path(a.src) / "index_stats.csv")))
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(prepare, items, chunksize=8))
    res = [res[k] for k in rng.permutation(len(res))]                      # source numbers carry no category or time
    grp = groups([G for _, G in res])
    split = split_groups(grp, rng)
    per = np.ones(len(res), int) if a.plain else allocate([r["category"] for r, _ in res], a.total, a.rare_cap, rng)
    jobs, maps, srcs = [], [], []
    for sid, ((r, G), g, sp, n) in enumerate(zip(res, grp, split, per)):
        np.savez(out / "sources" / f"s{sid:04d}.npz", grid3d=np.moveaxis(G, -1, 0))
        srcs.append(dict(source=sid, group=int(g), split=sp, variants=int(n), **r))
        augs = [dict(flip=False, half=False, mode="full", theta=0.0, scale=1.0, centre=[0.0, 0.0])] if a.plain \
            else variants(rng, r["category"], int(n), *G.shape[:2])
        for k, aug in enumerate(augs):
            jobs.append((out, sid, k, aug, G))
            maps.append(dict(id=f"m{sid:04d}_{k:02d}", source=sid, group=int(g), split=sp, aug=aug))
    with ProcessPoolExecutor(a.workers) as ex:
        list(ex.map(write, jobs, chunksize=16))
    for name, rows in (("maps", maps), ("sources", srcs)):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(check(out, maps, srcs), indent=1))
    if a.preview:
        preview(out, [maps[i] for i in rng.permutation(len(maps))])


if __name__ == "__main__":
    main()
