"""Five ways to pick N in-scope pool pages (ds_grid mask, pages still in emb_out) spread over the page kinds, inside and between their
clusters; selections on a Kaggle GPU, compared locally. Data: ds100k/spread (OUT).

Methods (KERNEL):
  hkm        hierarchical k-means + balanced sampling (Vo et al. 2024, the DINOv2 / DINOv3 data curation) on the 1536-d unit vectors: 4 levels,
             clusters per level = input / RATIOS (743M -> 10M / 500k / 50k / 10k of the paper), 10 resampling steps on the levels above the
             first (r_t = half the mean cluster size, the points closest to the centroid), budget split top-down by water filling (small
             clusters give all), random pages inside the level-1 clusters
  dbp        density-based pruning (Abbas et al. 2024) on the 1536-d unit vectors: spherical k-means K=500, complexity d_inter (mean cosine
             distance to the 20 nearest centroids) x d_intra (mean cosine distance to the centroid), quota softmax(C / 0.1) N clipped to
             [1, size] (the QP of the paper), the least prototypical pages (farthest from the centroid) of every cluster
  fused_fps  k-center greedy (farthest-point sampling) in the multi-view space: blocks embedding 1536 | lite layout 161 | font
             families (PCs of the indicator block, 90 % of its variance), each centred block divided by its first singular value
             (multiple factor analysis weighting)
  fused_hkm  hkm in the multi-view space
  strat      strata of the data's own fields, language x plan type x handwriting (hw_det hw > 0): quota by water filling, k-center greedy
             on the 1536-d unit vectors inside a stratum

  lite      lite layout vectors (tools/pages_extra.py lite_job) of the pool pages pages_topup has none for -> OUT/lite/NNNN.npz (resumable)
  prep      -> OUT/aux (mask, strata), dataset AUX; with --fused also OUT/fused (lite, language, fonts blocks, MFA weights), dataset FUSED
  push M    kernel of method M;  fetch M -> OUT/M.parquet (pool_i)
  test M    the kernel on the CPU on a subsample (SUB pages, N = SUB / 10); --full: all pages, N, -> OUT/out/M and OUT/M.parquet (as fetch)
  extent M  share of every level-1 cluster of M (hkm, fused_hkm) by its extent: greedy k-center inside the cluster (first page: the one
            nearest the cluster mean), insertion distance of every page; the N largest over all clusters (one common spacing) ->
            OUT/out/M_extent/M_extent.npz (pick, pool_i, cl, ideal = pages of the cluster taken), OUT/M_extent.parquet
  compare   pool / raw FPS (ds100k select.parquet) / the methods: plan type, handwriting, scan, language shares; coverage (distance in the
            1536-d space of a page to the nearest selected page) of the labelled pages by class and of a pool sample by k10-density decile

python tools/ds_spread.py lite | prep [--fused] | push M [--n 100000] | fetch M | test M [--full] | extent M [--n 100000] | compare"""

import argparse
import glob
import json
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor

import diverse_ds as DD
import ds100k as D
import ds_border as B
import ds_grid as G
import ds_lda as LD
import kaggle_fps as KF
import more_pages as M
import numpy as np
import pages_extra as X
import pandas as pd

OUT = D.OUT / "spread"
AUX, FUSED = "pavel9860/ds-spread-aux", "pavel9860/ds-spread-fused"
METHODS = ("hkm", "dbp", "fused_fps", "fused_hkm", "strat", "fused_knn", "fused_elim", "fused_herd", "emb_knn")
EXTENT = ("hkm_extent", "fused_hkm_extent")
TOPUP = M.LAYOUTS / "archive" / "info" / "pages_topup" / "lite_feats"
CHUNK, PARTS, SAMPLE, SUB = 400, 4, 200_000, 20_000
FONT_DOCS, FONT_VAR = 100, 0.90

KERNEL = r'''
import glob, time, numpy as np, torch
t0 = time.time()
def log(*a):
    print(f"[{int(time.time() - t0)} s]", *a, flush=True)
def find(name):
    return sorted(glob.glob(f"{ROOT}/**/{name}", recursive=True))
mask = np.load(find("mask.npy")[0])
real = np.flatnonzero(mask)
X = np.concatenate([np.load(f) for f in find("pool_*.npy")])
rows = np.arange(len(real)) if SUB is None else np.sort(np.random.default_rng(0).choice(len(real), SUB, replace=False))
n = len(rows)
torch.manual_seed(0)


def water(s, total):
    s = np.asarray(s, np.int64)
    if s.sum() <= total:
        return s.copy()
    lo, hi = 0, int(s.max())
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if np.minimum(s, mid).sum() <= total else (lo, mid - 1)
    q = np.minimum(s, lo)
    room = np.flatnonzero(s > q)
    q[room[:int(total - q.sum())]] += 1
    return q


def space(kind):
    if kind == "emb":
        U = torch.empty((n, X.shape[1]), dtype=torch.float16, device=DEV)
        for a in range(0, n, 100_000):
            U[a:a + 100_000] = torch.from_numpy(X[real[rows[a:a + 100_000]]]).to(DEV)
        return U
    w = np.load(find("mfa.npz")[0])
    Lz = np.concatenate([np.load(f) for f in find("lite_*.npy")])[rows]
    Fp = np.concatenate([np.load(f) for f in find("fonts_*.npy")])[rows]
    lg = np.load(find("lang.npy")[0])[rows]
    muE, p = torch.from_numpy(w["muE"]).to(DEV), torch.from_numpy(w["p"]).to(DEV)
    U = torch.empty((n, X.shape[1] + Lz.shape[1] + Fp.shape[1]), dtype=torch.float16, device=DEV)         # no language block
    for a in range(0, n, 100_000):
        b = min(a + 100_000, n)
        e = (torch.from_numpy(X[real[rows[a:b]]]).to(DEV).float() - muE) * float(w["wE"])
        l = torch.from_numpy(Lz[a:b]).to(DEV).float() * float(w["wL"])
        f = torch.from_numpy(Fp[a:b]).to(DEV).float() * float(w["wF"])
        U[a:b] = torch.cat([e, l, f], 1).half()
    log("multi-view space", tuple(U.shape), "block weights", {k: round(float(w[k]), 3) for k in ("wE", "wL", "wF")})
    return U


def norms(U):
    return torch.cat([(U[a:a + 100_000].float() ** 2).sum(1) for a in range(0, len(U), 100_000)])


def fps(U, sq, N, B=100, radii=None):
    """k-center greedy, B pages a round (each farther from the round's pages than from the set); radii: list that gets the distance of
    every pick to the set when picked."""
    d = torch.full((len(U),), float("inf"), device=DEV)
    pick, acc, r_acc = [], [0], [float("inf")]
    while True:
        pick += acc
        if radii is not None:
            radii += r_acc
        P = torch.tensor(acc, device=DEV)
        torch.minimum(d, (sq[:, None] + sq[P][None] - 2 * (U @ U[P].T).float()).clamp_(min=0).min(1).values, out=d)
        d[P] = -1                                                       # never again (identical pages: distance 0)
        if len(pick) >= N:
            return np.array(pick)
        v, c = torch.topk(d, min(4 * B, len(U)))
        acc, r_acc = [], []
        for dv, j in zip(v.tolist(), c.tolist()):
            if dv < 0:
                break
            if acc and float((sq[acc] + sq[j] - 2 * (U[acc] @ U[j]).float()).min()) < dv:
                continue
            acc.append(j)
            r_acc.append(dv ** 0.5)
            if len(acc) == B or len(pick) + len(acc) == N:
                break


def assign(U, sq, C, chunk=16384):
    cs, Ch = (C.float() ** 2).sum(1), C.half()
    lab = torch.empty(len(U), dtype=torch.long, device=DEV)
    dist = torch.empty(len(U), device=DEV)
    for a in range(0, len(U), chunk):
        v, i = (sq[a:a + chunk, None] + cs[None] - 2 * (U[a:a + chunk] @ Ch.T).float()).min(1)
        lab[a:a + chunk], dist[a:a + chunk] = i, v.clamp(min=0)
    return lab, dist


def kpp(U, sq, k, seed):
    """k-means++ seeds."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    idx = [int(torch.randint(len(U), (1,), generator=g, device=DEV))]
    d = (sq + sq[idx[0]] - 2 * (U @ U[idx[0]]).float()).clamp(min=0)
    for _ in range(k - 1):
        j = int(torch.multinomial(d + 1e-12, 1, generator=g))
        idx.append(j)
        torch.minimum(d, (sq + sq[j] - 2 * (U @ U[j]).float()).clamp(min=0), out=d)
    return U[torch.tensor(idx, device=DEV)].float()


def kmeans(U, sq, k, iters, seed=0, C=None, spherical=False):
    """Lloyd k-means (spherical: unit centroids); -> centroids (float), labels; empty clusters keep their centroid."""
    C = kpp(U, sq, k, seed) if C is None else C.float().clone()
    for _ in range(iters):
        lab, _ = assign(U, sq, C)
        S = torch.zeros_like(C)
        for a in range(0, len(U), 100_000):
            S.index_add_(0, lab[a:a + 100_000], U[a:a + 100_000].float())
        cnt = torch.bincount(lab, minlength=k)
        ok = cnt > 0
        C[ok] = S[ok] / cnt[ok, None]
        if spherical:
            C = torch.nn.functional.normalize(C, dim=1)
    lab, _ = assign(U, sq, C)
    return C, lab


def compact(C, lab):
    u, lab = torch.unique(lab, return_inverse=True)
    return C[u], lab


def hkm(U, sq, N, ratios=RATIOS, m=10, iters=30):
    levels, pts, psq = [], U, sq
    for t, r in enumerate(ratios):
        k = max(2, round(len(pts) / r))
        C, lab = kmeans(pts, psq, k, iters, seed=t)
        for s in range(m if t else 0):
            C, lab = compact(C, lab)
            k = len(C)
            rt = max(1, round(len(pts) / k / 2))
            dist = ((pts.float() - C[lab]) ** 2).sum(1)
            o = torch.argsort(lab.double() * 1e6 + dist.double() / (dist.max().double() + 1))
            ls = lab[o]
            first = torch.searchsorted(ls, torch.arange(k, device=DEV))
            rank = torch.arange(len(ls), device=DEV) - first[ls]
            sel = o[rank < rt]
            C, _ = kmeans(pts[sel], psq[sel], k, iters, C=C)
            lab, _ = assign(pts, psq, C)
        C, lab = compact(C, lab)
        levels.append(lab.cpu().numpy())
        log(f"level {t + 1}: {len(C)} clusters of {len(pts)} points")
        pts = C.half()
        psq = (C ** 2).sum(1)
    sizes = [np.bincount(levels[0])]
    for t in range(1, len(levels)):
        sizes.append(np.bincount(levels[t], weights=sizes[-1]).astype(np.int64))
    alloc = water(sizes[-1], N)
    for t in range(len(levels) - 1, 0, -1):
        child = np.zeros(len(sizes[t - 1]), np.int64)
        o = np.argsort(levels[t], kind="stable")
        for j, ch in enumerate(np.split(o, np.cumsum(np.bincount(levels[t], minlength=len(sizes[t])))[:-1])):
            child[ch] = water(sizes[t - 1][ch], alloc[j])
        alloc = child
    ideal = np.full(len(sizes[-1]), N / len(sizes[-1]))                 # equal split without the size cap: pages to add where it is short
    for t in range(len(levels) - 1, 0, -1):
        nch = np.bincount(levels[t], minlength=len(sizes[t]))
        ideal = ideal[levels[t]] / nch[levels[t]]
    rng = np.random.default_rng(0)
    o = np.argsort(levels[0], kind="stable")
    groups = np.split(o, np.cumsum(sizes[0])[:-1])
    return np.concatenate([rng.choice(g, q, replace=False) for g, q in zip(groups, alloc) if q]), levels[0], ideal


def dbp(U, sq, N, K=500, ell=20, tau=0.1):
    C, lab = compact(*kmeans(U, sq, K, 100, spherical=True))
    K = len(C)
    cos = torch.cat([(U[a:a + 100_000].float() * C[lab[a:a + 100_000]]).sum(1) for a in range(0, len(U), 100_000)])
    size = torch.bincount(lab, minlength=K).double()
    intra = torch.zeros(K, device=DEV, dtype=torch.double).index_add_(0, lab, (1 - cos).double()) / size
    cc = 1 - C @ C.T
    cc.fill_diagonal_(float("inf"))
    inter = cc.topk(ell, largest=False).values.mean(1).double()
    P = torch.softmax(inter * intra / tau, 0).cpu().numpy()
    M_ = size.cpu().numpy()
    lo, hi = -float(N), float(N)
    for _ in range(200):
        lam = (lo + hi) / 2
        lo, hi = (lam, hi) if np.clip(P * N + lam, 1, M_).sum() < N else (lo, lam)
    x = np.clip(P * N + lam, 1, M_)
    q = np.floor(x).astype(np.int64)
    room = np.flatnonzero(q < M_)
    q[room[np.argsort(-(x - q)[room])][:max(0, N - int(q.sum()))]] += 1
    log(f"dbp: {K} clusters, quota min {q.min()} median {int(np.median(q))} max {q.max()}, clusters taken whole {int((q == M_).sum())}")
    labn, dist = lab.cpu().numpy(), (1 - cos).cpu().numpy()
    o = np.lexsort((-dist, labn))
    first = np.searchsorted(labn[o], np.arange(K))
    return np.concatenate([o[first[j]:first[j] + q[j]] for j in range(K)]), labn, P * N


def strat(U, sq, N):
    s = np.load(find("strata.npy")[0])[rows]
    cnt = np.bincount(s)
    q = water(cnt, N)
    o = np.argsort(s, kind="stable")
    out = []
    for j, idx in enumerate(np.split(o, np.cumsum(cnt)[:-1])):
        if q[j] >= len(idx):
            out.append(idx)
        elif q[j]:
            t = torch.from_numpy(idx).to(DEV)
            out.append(idx[fps(U[t], sq[t], int(q[j]))])
    log(f"strat: {len(cnt)} strata, cap {int(q.max())}, taken whole {int((q == cnt).sum())}")
    return np.concatenate(out), s, np.full(len(cnt), N / len(cnt))


U = space("fused" if METHOD.startswith("fused") else "emb")
del X
sq = norms(U)
log("space", tuple(U.shape))
if METHOD in ("fused_knn", "emb_knn"):                     # KN nearest pages + k-center radii up to N + exact all-pairs statistics at the spacing r
    def rows_pass(fn, block=256):
        for a in range(0, n, block):
            fn(a, (sq[a:a + block, None] + sq[None] - 2 * (U[a:a + block] @ U.T).float()).clamp_(min=0))
    ki, kd = np.empty((n, KN), np.int32), np.empty((n, KN), np.float16)
    def knn(a, d2):
        d2[torch.arange(len(d2)), torch.arange(a, a + len(d2))] = float("inf")
        v, i = d2.topk(KN, dim=1, largest=False)
        ki[a:a + len(d2)], kd[a:a + len(d2)] = i.int().cpu().numpy(), v.sqrt().half().cpu().numpy()
    rows_pass(knn)
    log("knn done")
    rad = []
    fpick = fps(U, sq, N, radii=rad)
    r = rad[-1]
    r_min = r * (1 - (N / n) ** 1.5) * 0.65                    # weighted sample elimination (Yuksel 2015): r_max = r, gamma 1.5, beta 0.65
    log("k-center spacing at N", round(r, 4))
    dens = torch.zeros(n, dtype=torch.float64, device=DEV)
    cnt_r, cnt_2r, w0 = torch.zeros(n, device=DEV), torch.zeros(n, device=DEV), torch.zeros(n, dtype=torch.float64, device=DEV)
    f_self = (1 - r_min / r) ** 8
    def stats(a, d2):
        b = a + len(d2)
        dens[a:b] = torch.exp(-d2 / r ** 2).sum(1).double()
        d = d2.sqrt_()
        cnt_r[a:b] = (d < r).sum(1).float() - 1
        cnt_2r[a:b] = (d < 2 * r).sum(1).float() - 1
        w0[a:b] = ((1 - d.clamp_(2 * r_min, 2 * r) / (2 * r)) ** 8).sum(1).double() - f_self
    rows_pass(stats)
    np.savez(f"{OUTDIR}/{METHOD}.npz", pool_i=real[rows], ki=ki, kd=kd, pick=real[rows[fpick]], rad=np.array(rad, np.float32),
             dens=dens.cpu().numpy(), cnt_r=cnt_r.cpu().numpy(), cnt_2r=cnt_2r.cpu().numpy(), w0=w0.cpu().numpy(), r_min=r_min)
    log("DONE", METHOD, "spacing", round(r, 4), "pages within r: median", float(cnt_r.median()), "within 2r: median", float(cnt_2r.median()),
        "p99", float(cnt_2r.quantile(.99)) if n <= 16_000_000 else -1)
    raise SystemExit
if METHOD in ("fused_elim", "fused_herd"):                     # on the statistics of fused_knn (kernel output attached)
    z = np.load(find("fused_knn.npz")[0])
    assert len(z["pool_i"]) == n
    r, r_min = float(z["rad"][-1]), float(z["r_min"])
    if METHOD == "fused_elim":                # weighted sample elimination (Yuksel 2015): 2 r_max = the k-center spacing r (the distance between
        rm = r / 2                            # samples), r_min = r_max (1 - (N / n)^1.5) 0.65, alpha 8; the page of the largest weight goes;
        r_min = rm * (1 - (N / n) ** 1.5) * 0.65                     # a round takes up to BE pages, each r or more from the round's others
        f_self = (1 - r_min / rm) ** 8
        w = torch.zeros(n, dtype=torch.float64, device=DEV)
        for a in range(0, n, 256):
            d = (sq[a:a + 256, None] + sq[None] - 2 * (U[a:a + 256] @ U.T).float()).clamp_(min=0).sqrt_()
            w[a:a + 256] = ((1 - d.clamp_(2 * r_min, 2 * rm) / (2 * rm)) ** 8).sum(1).double() - f_self
        log("weights done; pages with a neighbour within r: median", float(torch.from_numpy(z["cnt_r"]).median()))
        alive = torch.ones(n, dtype=torch.bool, device=DEV)
        left, rounds = n, 0
        while left > N:
            v, c = torch.topk(torch.where(alive, w, torch.full_like(w, -1e300)), min(4 * BE, left))
            near = ((sq[c][:, None] + sq[c][None] - 2 * (U[c] @ U[c].T).float()) < r * r).cpu().numpy()
            acc, block = [], np.zeros(len(c), bool)
            for k in range(len(c)):
                if block[k]:
                    continue
                acc.append(k)
                block |= near[k]
                if len(acc) == BE or len(acc) == left - N:
                    break
            A = c[torch.tensor(acc, device=DEV)]
            alive[A] = False
            left -= len(acc)
            d = (sq[:, None] + sq[A][None] - 2 * (U @ U[A].T).float()).clamp_(min=0).sqrt_()
            w -= ((1 - d.clamp_(2 * r_min, 2 * rm) / (2 * rm)) ** 8).sum(1).double()
            rounds += 1
            if rounds % 1000 == 0:
                log(f"elimination: {left} left, {rounds} rounds, last round {len(acc)}")
        np.savez(f"{OUTDIR}/{METHOD}.npz", pool_i=real[rows], pick=real[rows[alive.cpu().numpy()]])
    else:                                     # kernel herding to the uniform-over-support target (1 / kernel density), full kernel
        dens = torch.from_numpy(z["dens"]).to(DEV)
        pi = (1 / dens) / (1 / dens).sum()
        mu = torch.zeros(n, dtype=torch.float64, device=DEV)
        for a in range(0, n, 256):
            d2 = (sq[a:a + 256, None] + sq[None] - 2 * (U[a:a + 256] @ U.T).float()).clamp_(min=0)
            mu[a:a + 256] = torch.exp(-d2 / r ** 2).double() @ pi
        log("target kernel mean done")
        A = torch.zeros(n, dtype=torch.float64, device=DEV)
        count = torch.zeros(n, dtype=torch.int64, device=DEV)
        for t in range(N):
            j = int(torch.argmax((t + 1) * mu - A))
            count[j] += 1
            A += torch.exp(-(sq + sq[j] - 2 * (U @ U[j]).float()).clamp_(min=0) / r ** 2).double()
            if (t + 1) % 10_000 == 0:
                log(f"herding {t + 1}, distinct pages {int((count > 0).sum())}")
        c = count.cpu().numpy()
        np.savez(f"{OUTDIR}/{METHOD}.npz", pool_i=real[rows], pick=real[rows[c > 0]], count=c)
    log("DONE", METHOD)
    raise SystemExit
f = {"hkm": hkm, "dbp": dbp, "fused_fps": lambda U, sq, N: (fps(U, sq, N), None, None), "fused_hkm": hkm, "strat": strat}[METHOD]
pick, cl, ideal = f(U, sq, N)
assert len(pick) == N and len(np.unique(pick)) == N, (len(pick), len(np.unique(pick)))
np.savez(f"{OUTDIR}/{METHOD}.npz", pick=real[rows[np.asarray(pick)]], **({} if cl is None else {"pool_i": real[rows], "cl": cl.astype(np.int32), "ideal": ideal}))
log("DONE", METHOD, len(pick))
'''


def alive_mask():
    """The ds_grid mask without the pool pages that left the embedding since (documents moved as copies, identical pages)."""
    P = pd.read_parquet(D.OUT / "pool.parquet", columns=["doc", "page"])
    ids = np.load(D.EMB / "ids.npy")
    cur = np.unique(ids[:, 0].astype(np.int64) * M.KEY + ids[:, 1])
    return np.load(G.WORK / "aux" / "mask.npy") & np.isin(
        P.doc.to_numpy().astype(np.int64) * M.KEY + P.page.to_numpy(), cur
    )


def lite_store():
    """key -> row of the lite vectors (pages_topup, then OUT/lite)."""
    keys, Vs = [], []
    for f in sorted(TOPUP.glob("*.npz")) + sorted((OUT / "lite").glob("*.npz")):
        z = np.load(f)
        keys.append(z["key"]), Vs.append(z["V"])
    key, V = np.concatenate(keys), np.concatenate(Vs)
    key, i = np.unique(key, return_index=True)
    return key, V[i]


def cmd_lite(a):
    P = pd.read_parquet(D.OUT / "pool.parquet", columns=["doc", "page", "path", "scan"])
    m = alive_mask()
    key = P.doc.to_numpy().astype(np.int64) * M.KEY + P.page.to_numpy()
    have = np.concatenate([np.load(f)["key"] for f in sorted(TOPUP.glob("*.npz"))])
    Q = P[m & ~np.isin(key, have)]
    jobs = []
    for doc, g in Q.groupby("doc"):
        f = DD.resolve(g.path.iloc[0])
        if f is not None:
            jobs.append((int(doc), str(f), bool(g.scan.iloc[0]), sorted(g.page.astype(int).tolist())))
    jobs = jobs[: a.limit] if a.limit else jobs
    print(
        f"{len(Q)} pages without lite vectors; {len(jobs)} documents, {sum(len(j[3]) for j in jobs)} pages to look at",
        flush=True,
    )
    fd = OUT / ("lite_test" if a.limit else "lite")
    fd.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(M.WORKERS) as ex:
        for c in range(0, len(jobs), CHUNK):
            f = fd / f"{c // CHUNK:04d}.npz"
            if f.exists():
                continue
            res = list(ex.map(X.lite_job, jobs[c : c + CHUNK], chunksize=4))
            k = [d * M.KEY + p for d, o, _ in res for p in o]
            V = np.stack([v for _, o, _ in res for v in o.values()]) if k else np.zeros((0, X.LDIM), np.float32)
            np.savez(
                f"{f}.tmp.npz", key=np.array(k, np.int64), V=V, err=json.dumps({int(d): e for d, _, e in res if e})
            )
            f.with_name(f.name + ".tmp.npz").rename(f)
            print(f"lite {c + len(res)}/{len(jobs)} docs, {int(time.time() - t0)} s", flush=True)


def strata(P):
    with sqlite3.connect(f"file:{M.LAYOUTS / 'manifest.sqlite'}?mode=ro", uri=True) as db:
        typ = dict(db.execute("select doc, type from plan"))
    t = P.h.map(typ).fillna("_untyped")
    return pd.Series(
        P.lang.fillna("unknown").to_numpy() + "|" + t.to_numpy() + "|" + (P.hw > 0).astype(str).to_numpy(),
        index=P.index,
    )


def top_var(Z, share):
    """Eigen-decomposition of the covariance of the rows Z; (mean, eigenvalues desc, vectors desc)."""
    import torch

    T = torch.from_numpy(Z).double()
    mu = T.mean(0)
    ev, V = torch.linalg.eigh(torch.cov(T.T))
    return mu.numpy(), ev.flip(0).numpy(), V.flip(1).numpy()


def upload(d, ident):
    (d / "dataset-metadata.json").write_text(
        json.dumps({"title": ident.split("/")[1], "id": ident, "licenses": [{"name": "CC-BY-NC-SA-4.0"}]})
    )
    print(KF.kaggle("datasets", "create", "-p", str(d), timeout=7200)[-300:])


def cmd_prep(a):
    P = pd.read_parquet(D.OUT / "pool.parquet")
    m = alive_mask()
    real = np.flatnonzero(m)
    R = P.iloc[real]
    d = OUT / "aux"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "mask.npy", m)
    s = strata(R)
    np.save(d / "strata.npy", pd.factorize(s)[0].astype(np.int32))
    print(f"in-scope pages {len(real)}; strata (language x plan type x hw) {s.nunique()}")
    if not a.fused:
        upload(d, AUX)
        return
    rng = np.random.default_rng(0)
    samp = np.sort(rng.choice(len(real), SAMPLE, replace=False))
    f = OUT / "fused"
    f.mkdir(parents=True, exist_ok=True)
    E = B.pool_X()
    muE, evE, _ = top_var(E[real[samp]].astype(np.float32), 1)
    key, V = lite_store()
    V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
    for lo, hi, flag in (
        (0, X.LITE, X.LDIM - 2),
        (X.LITE, X.LITE + M.NP, X.LDIM - 1),
    ):  # z-score of each block (pages_topup select)
        h = V[:, flag] > 0
        mu, sd = V[h, lo:hi].mean(0), V[h, lo:hi].std(0) + 1e-6
        V[:, lo:hi] = np.where(h[:, None], np.clip((V[:, lo:hi] - mu) / sd, -5, 5), 0)
    V[:, -2:] *= 3
    pk = R.doc.to_numpy().astype(np.int64) * M.KEY + R.page.to_numpy()
    at = np.searchsorted(key, pk)
    ok = (at < len(key)) & (key[np.minimum(at, len(key) - 1)] == pk)
    L = np.zeros((len(real), X.LDIM), np.float32)
    L[ok] = V[at[ok]]
    print(
        f"lite vectors: {int(ok.sum())} of {len(real)} pages ({len(real) - int(ok.sum())} without: empty / unreadable page, zeros)"
    )
    muL, evL, _ = top_var(L[samp], 1)
    L -= muL.astype(np.float32)
    langs, lg = np.unique(R.lang.fillna("unknown").to_numpy(), return_inverse=True)
    p = np.bincount(lg, minlength=len(langs)) / len(lg)
    evG = np.linalg.eigvalsh(np.diag(p) - np.outer(p, p))[::-1]
    with sqlite3.connect(f"file:{M.LAYOUTS / 'manifest.sqlite'}?mode=ro", uri=True) as db:
        Fo = pd.read_sql("select doc, font from fonts", db)
    Fo["fam"] = Fo.font.str.split("+").str[-1].str.split(r"[-,]", regex=True).str[0].str.lower()
    Fo = Fo[Fo.doc.isin(set(R.h))].drop_duplicates(["doc", "fam"])
    vc = Fo.fam.value_counts()
    fams = vc.index[vc >= FONT_DOCS]
    Fo = Fo[Fo.fam.isin(fams)]
    di = {h: i for i, h in enumerate(R.h.unique())}
    from scipy import sparse

    Ind = sparse.csr_matrix(
        (np.ones(len(Fo), np.float32), (Fo.doc.map(di).to_numpy(), pd.Index(fams).get_indexer(Fo.fam))),
        shape=(len(di), len(fams)),
    )
    pdoc = R.h.map(di).to_numpy()
    muF, evF, VF = top_var(Ind[pdoc[samp]].toarray(), 1)
    kF = int(np.searchsorted(np.cumsum(evF) / evF.sum(), FONT_VAR) + 1)
    Fp = np.concatenate(
        [
            ((Ind[pdoc[a : a + 100_000]].toarray() - muF) @ VF[:, :kF]).astype(np.float16)
            for a in range(0, len(real), 100_000)
        ]
    )
    print(
        f"font families in >= {FONT_DOCS} documents: {len(fams)}; {kF} PCs hold {FONT_VAR:.0%}; pages with a listed family "
        f"{float(np.isin(R.h, Fo.doc.unique()).mean()):.3f}"
    )
    w = {"wE": 1 / np.sqrt(evE[0]), "wL": 1 / np.sqrt(evL[0]), "wG": 1 / np.sqrt(evG[0]), "wF": 1 / np.sqrt(evF[0])}
    inert = {"E": evE.sum() / evE[0], "L": evL.sum() / evL[0], "G": evG.sum() / evG[0], "F": evF[:kF].sum() / evF[0]}
    print(
        "MFA weights (1 / first singular value):",
        {k: round(float(v), 3) for k, v in w.items()},
        "| block inertia after weighting:",
        {k: round(float(v), 2) for k, v in inert.items()},
    )
    for k, part in enumerate(np.array_split(np.arange(len(real)), PARTS)):
        np.save(f / f"lite_{k}.npy", L[part].astype(np.float16))
        np.save(f / f"fonts_{k}.npy", Fp[part])
    np.save(f / "lang.npy", lg.astype(np.int16))
    np.savez(
        f / "mfa.npz",
        muE=muE.astype(np.float32),
        p=p.astype(np.float32),
        langs=langs,
        fams=np.array(fams),
        **{k: np.float32(v) for k, v in w.items()},
    )
    upload(f, FUSED)


def kernel_code(method, n, root, outdir, dev, sub):
    ratios = (74, 20, 10, 5)
    return (
        f"METHOD, N, ROOT, OUTDIR, DEV, SUB, RATIOS, KN, BE = {method!r}, {n}, {root!r}, {outdir!r}, {dev!r}, {sub}, {ratios}, 64, 500\n"
        + KERNEL
    )


def kid(method):
    return f"pavel9860/ds-spread-{method.replace('_', '-')}"


def cmd_push(a):
    d = OUT / "kernel" / a.method
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.py").write_text(kernel_code(a.method, a.n, "/kaggle/input", "/kaggle/working", "cuda", None))
    src = [KF.DS, AUX] + ([FUSED] if a.method.startswith("fused") else [])
    ks = [kid("fused_knn")] if a.method in ("fused_elim", "fused_herd") else []
    (d / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": kid(a.method),
                "title": kid(a.method).split("/")[1],
                "code_file": "run.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_tpu": False,
                "enable_internet": False,
                "dataset_sources": src,
                "kernel_sources": ks,
                "competition_sources": [],
                "model_sources": [],
                "machine_shape": "NvidiaTeslaT4",
            }
        )
    )
    print(KF.kaggle("kernels", "push", "-p", str(d)))


def cmd_test(a):
    """The kernel on the CPU: pool parts of kaggle_fps WORK/data, OUT/aux and OUT/fused linked into one folder."""
    import torch

    torch.set_num_threads(M.WORKERS)
    root = D.TESTS / "spread" / "test" / "in"
    root.mkdir(parents=True, exist_ok=True)
    for f in [
        *glob.glob(str(KF.WORK / "data" / "pool_*.npy")),
        *glob.glob(str(OUT / "aux" / "*.npy")),
        *glob.glob(str(OUT / "fused" / "*.np*")),
    ]:
        t = root / f.rsplit("/", 1)[1]
        if not t.exists():
            t.symlink_to(f)
    out = OUT / "out" / a.method if a.full else D.TESTS / "spread" / "test"
    out.mkdir(parents=True, exist_ok=True)
    code = kernel_code(a.method, a.n if a.full else SUB // 10, str(root), str(out), "cpu", None if a.full else SUB)
    exec(compile(code, "kernel", "exec"), {"__name__": "kernel"})
    if a.full:
        pd.DataFrame({"pool_i": np.load(out / f"{a.method}.npz")["pick"]}).to_parquet(OUT / f"{a.method}.parquet")


def cmd_fetch(a):
    st = KF.kaggle("kernels", "status", kid(a.method))
    print(st)
    if "COMPLETE" not in st.upper():
        return
    out = OUT / "out" / a.method
    out.mkdir(parents=True, exist_ok=True)
    print(KF.kaggle("kernels", "output", kid(a.method), "-p", str(out))[-300:])
    z = np.load(out / f"{a.method}.npz")
    pd.DataFrame({"pool_i": z["pick"]}).to_parquet(OUT / f"{a.method}.parquet")
    print(len(z["pick"]), "pages ->", OUT / f"{a.method}.parquet")


def fused_rows():
    """The multi-view vectors of the in-scope pages (rows = OUT/aux mask order), fp16, as the kernel builds them."""
    import torch

    real = np.flatnonzero(np.load(OUT / "aux" / "mask.npy"))
    E = B.pool_X()
    f = OUT / "fused"
    w = np.load(f / "mfa.npz")
    Lz = np.concatenate([np.load(f / f"lite_{k}.npy") for k in range(PARTS)])
    Fp = np.concatenate([np.load(f / f"fonts_{k}.npy") for k in range(PARTS)])
    np.load(f / "lang.npy")
    muE, _p = torch.from_numpy(w["muE"]), torch.from_numpy(w["p"])
    U = np.empty((len(real), E.shape[1] + Lz.shape[1] + Fp.shape[1]), np.float16)  # no language block
    for a in range(0, len(real), 100_000):
        b = min(a + 100_000, len(real))
        e = (torch.from_numpy(E[real[a:b]]).float() - muE) * float(w["wE"])
        U[a:b] = (
            torch.cat(
                [
                    e,
                    torch.from_numpy(Lz[a:b]).float() * float(w["wL"]),
                    torch.from_numpy(Fp[a:b]).float() * float(w["wF"]),
                ],
                1,
            )
            .half()
            .numpy()
        )
    return U


def kcenter_radii(V):
    """Greedy k-center order of the rows V (first: the row nearest their mean) -> insertion distance per row in pick order (first: inf), order."""
    import torch

    T = torch.from_numpy(V).float()
    sq = (T * T).sum(1)
    j = int(((T - T.mean(0)) ** 2).sum(1).argmin())
    d = torch.full((len(T),), float("inf"))
    rad, order = np.empty(len(T), np.float32), np.empty(len(T), np.int64)
    for t in range(len(T)):
        order[t], rad[t] = j, np.inf if t == 0 else float(d[j]) ** 0.5
        torch.minimum(d, (sq + sq[j] - 2 * (T @ T[j])).clamp_(min=0), out=d)
        d[order[: t + 1]] = -1
        j = int(d.argmax())
    return rad, order


def cmd_extent(a):
    import torch

    torch.set_num_threads(M.WORKERS)
    z = np.load(OUT / "out" / a.method / f"{a.method}.npz")
    cl, pool_i = z["cl"], z["pool_i"]
    real = np.flatnonzero(np.load(OUT / "aux" / "mask.npy"))
    assert (pool_i == real).all()
    U = fused_rows() if a.method.startswith("fused") else B.pool_X()[real]
    o = np.argsort(cl, kind="stable")
    groups = np.split(o, np.cumsum(np.bincount(cl))[:-1])
    print(
        f"{len(groups)} clusters, pages per cluster median {int(np.median([len(g) for g in groups]))} max {max(len(g) for g in groups)}",
        flush=True,
    )
    t0, rad, idx = time.time(), [], []
    for k, g in enumerate(groups):
        r, order = kcenter_radii(U[g])
        rad.append(r), idx.append(g[order])
        if k % 1000 == 0:
            print(f"{k} clusters, {int(time.time() - t0)} s", flush=True)
    rad, idx = np.concatenate(rad), np.concatenate(idx)
    top = np.argsort(-rad, kind="stable")[: a.n]
    print(f"common spacing (the {a.n}th insertion distance): {rad[top[-1]]:.4f}")
    ideal = np.bincount(cl[idx[top]], minlength=len(groups))
    name = f"{a.method}_extent"
    out = OUT / "out" / name
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{name}.npz", pick=real[idx[top]], pool_i=pool_i, cl=cl, ideal=ideal.astype(np.float64))
    pd.DataFrame({"pool_i": real[idx[top]]}).to_parquet(OUT / f"{name}.parquet")
    print(
        f"{a.n} pages -> {OUT / f'{name}.parquet'}; pages per cluster: median {int(np.median(ideal))}, max {ideal.max()}, "
        f"clusters with one page {int((ideal == 1).sum())}, clusters taken whole {int((ideal == np.bincount(cl)).sum())}"
    )


CONT = ("fused_elim", "fused_herd", "invdens", "dmap")
ID_K, DENS_K = 20, 10


def pps(mass, seed=0):
    """Pages of the target masses: mass >= 1 taken, the rest by systematic sampling with inclusion probability = mass (random order)."""
    o = np.random.default_rng(seed).permutation(len(mass))
    c = np.cumsum(np.minimum(mass[o], 1.0))
    k = np.searchsorted(c, np.arange(np.random.default_rng(seed + 1).random(), c[-1], 1.0))
    return np.unique(o[k])


def cmd_cont(a):
    """invdens / dmap from the statistics of fused_knn; fused_elim / fused_herd: the kernel outputs (fetched) -> OUT/M.parquet."""
    z = np.load(OUT / "out" / a.base / f"{a.base}.npz")
    pool_i, N, mass = z["pool_i"], a.n, None
    name = a.method if a.base == "fused_knn" else f"{a.method}_{a.base.split('_')[0]}"
    if a.method in ("fused_elim", "fused_herd"):
        f = OUT / "out" / a.method
        f.mkdir(parents=True, exist_ok=True)
        print(KF.kaggle("kernels", "output", kid(a.method), "-p", str(f))[-200:])
        k = np.load(f / f"{a.method}.npz")
        pick = np.searchsorted(pool_i, k["pick"])
        if "count" in k.files:
            mass = k["count"].astype(np.float64)
    elif a.method == "invdens":  # volume per page T_k^d (Levina-Bickel intrinsic dimension)
        T = z["kd"][:, :ID_K].astype(np.float64) + 1e-9
        inv_m = np.log(T[:, -1:] / T[:, :-1]).mean(1)
        dim = 1 / inv_m.mean()
        lv = dim * np.log(T[:, DENS_K - 1])
        v = np.exp(lv - lv.max())
        mass = N * v / v.sum()
        print(
            f"intrinsic dimension (Levina-Bickel, k={ID_K}, MacKay-Ghahramani mean): {dim:.1f}; local median {np.median(1 / inv_m):.1f}"
        )
        pick = pps(mass)
    else:  # 1 / kernel density at the k-center spacing (Coifman-Lafon alpha = 1)
        v = 1 / z["dens"]
        mass = N * v / v.sum()
        pick = pps(mass)
    if mass is not None:
        add = np.maximum(mass - 1, 0)
        print(
            f"target mass per page: median {np.median(mass):.3f}, max {mass.max():.1f}; pages with mass > 1: {int((mass > 1).sum())}; "
            f"pages to add {int(round(add.sum()))}"
        )
        out = OUT / "out" / name
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / f"{name}.npz", pick=pool_i[pick], pool_i=pool_i, mass=mass)
    pd.DataFrame({"pool_i": pool_i[pick]}).to_parquet(OUT / f"{name}.parquet")
    print(f"{name}: {len(pick)} real pages -> {OUT / f'{name}.parquet'}")


def coverage(E, q, s, block=1024):
    """Distance of every query page q (pool rows) to the nearest other page of the set s (unit rows of E; a query page in s is not its own
    neighbour)."""
    import torch

    s = np.sort(s)
    St = torch.from_numpy(E[s]).float()
    out = []
    for a in range(0, len(q), block):
        v, i = (torch.from_numpy(E[q[a : a + block]]).float() @ St.T).topk(2, dim=1)
        own = torch.from_numpy(s[i[:, 0].numpy()] == q[a : a + block])
        c = torch.where(own, v[:, 1], v[:, 0])
        out.append((2 - 2 * c).clamp(min=0).sqrt().numpy())
    return np.concatenate(out)


def cmd_compare(a):
    import torch

    torch.set_num_threads(M.WORKERS)
    P = pd.read_parquet(D.OUT / "pool.parquet", columns=["h", "lang", "hw", "scan"])
    m = alive_mask()
    with sqlite3.connect(f"file:{M.LAYOUTS / 'manifest.sqlite'}?mode=ro", uri=True) as db:
        typ = dict(db.execute("select doc, type from plan"))
    P["type"] = P.h.map(typ).fillna("_untyped")
    P["hw>0"] = P.hw > 0
    raw = pd.read_parquet(D.OUT / "select.parquet").pool_i.to_numpy()
    sets = {"pool": np.flatnonzero(m), "raw_fps": raw[m[raw]]}
    sets |= {
        k: pd.read_parquet(OUT / f"{k}.parquet").pool_i.to_numpy()
        for k in a.sets.split(",")
        if (OUT / f"{k}.parquet").exists()
    }
    print("pages per set:", {k: len(v) for k, v in sets.items()})
    for col in ("type", "hw>0", "scan", "lang"):
        T = (
            pd.DataFrame({k: P[col].iloc[v].value_counts(normalize=True) for k, v in sets.items()})
            .fillna(0)
            .sort_values("pool", ascending=False)
        )
        if col == "lang":
            print("languages:", {k: int((P.lang.iloc[v].value_counts() > 0).sum()) for k, v in sets.items()})
            T = T.head(12)
        print(f"\n{col} share")
        print(T.round(4).to_string())
    print("\npages to add (target mass of a page above one page)")
    A = {}
    for k in a.sets.split(","):
        f = OUT / "out" / k / f"{k}.npz"
        if not f.exists() or "mass" not in np.load(f).files:
            continue
        z = np.load(f)
        w = np.maximum(z["mass"] - 1, 0)
        print(f"  {k}: {int(round(w.sum()))} pages to add at {int((w > 0).sum())} pages")
        A[(k, "type")] = pd.Series(w).groupby(P.type.iloc[z["pool_i"]].to_numpy()).sum()
        A[(k, "lang")] = pd.Series(w).groupby(P.lang.iloc[z["pool_i"]].fillna("unknown").to_numpy()).sum()
    for col in ("type", "lang"):
        T = pd.DataFrame({k: v for (k, c), v in A.items() if c == col}).fillna(0)
        if len(T):
            print(f"pages to add by {col} (of the clusters' pages)")
            print(T.sort_values(T.columns[0], ascending=False).head(15).round(0).astype(int).to_string())
    E = B.pool_X()
    L, _ = LD.labelled()
    L = L[m[L.pool_i.to_numpy()]]
    kn = np.load(D.OUT / "border" / "knn.npz")["d"][:, 9].astype(np.float32)
    rq = np.sort(np.random.default_rng(0).choice(np.flatnonzero(m), 20_000, replace=False))
    dec = pd.qcut(kn[rq], 10, labels=False) + 1
    C, Dn, In = {}, {}, {}
    for k, v in sets.items():
        if k == "pool":
            continue
        C[k] = pd.Series(coverage(E, L.pool_i.to_numpy(), v)).groupby(L.cls.to_numpy()).median()
        In[k] = pd.Series(np.isin(L.pool_i.to_numpy(), v)).groupby(L.cls.to_numpy()).mean()
        Dn[k] = pd.Series(coverage(E, rq, v)).groupby(dec).median()
    print(f"\nlabelled pages (pages per class: {L.cls.value_counts().to_dict()}): share in the set")
    print(pd.DataFrame(In).round(3).to_string())
    print("median distance to the nearest other selected page")
    print(pd.DataFrame(C).round(3).to_string())
    print(
        "\n20,000 pool pages by k10-distance decile (1 = densest): median distance to the nearest other selected page"
    )
    print(pd.DataFrame(Dn).round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("lite").add_argument("--limit", type=int, default=0)
    sp.add_parser("prep").add_argument("--fused", action="store_true")
    for c in ("push", "fetch", "test"):
        p = sp.add_parser(c)
        p.add_argument("method", choices=METHODS)
        p.add_argument("--n", type=int, default=100_000)
        p.add_argument("--full", action="store_true")
    p = sp.add_parser("cont")
    p.add_argument("method", choices=CONT)
    p.add_argument("--n", type=int, default=100_000)
    p.add_argument("--base", default="fused_knn", choices=("fused_knn", "emb_knn"))
    p = sp.add_parser("extent")
    p.add_argument("method", choices=("hkm", "fused_hkm"))
    p.add_argument("--n", type=int, default=100_000)
    sp.add_parser("compare").add_argument("--sets", default="fused_fps," + ",".join(CONT))
    a = ap.parse_args()
    {
        "lite": cmd_lite,
        "prep": cmd_prep,
        "push": cmd_push,
        "fetch": cmd_fetch,
        "test": cmd_test,
        "extent": cmd_extent,
        "cont": cmd_cont,
        "compare": cmd_compare,
    }[a.cmd](a)
