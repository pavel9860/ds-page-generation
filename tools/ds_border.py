"""Borders of the in-scope pool on the full 1536-d unit page vectors (tools/ds100k.py pool, Kaggle dataset pavel9860/ds100k-pool): exact
k nearest pool pages of every pool page on a Kaggle GPU; the pages farthest from their K-th neighbour are the border / outlier candidates
(valid but rare pages, or junk), rendered for review. Data: ds100k/border (OUT).

  push       kernel pavel9860/ds100k-knn: distances (sqrt(2 - 2 cos)) and pool indices of the KMAX nearest pages of every page
  fetch      -> OUT/knn.npz (d float16 (n, KMAX), i int32 (n, KMAX)), OUT/border.parquet (pool_i, k1, k5, k10, k20, rank by k10)
  border     border samples by projection depth on 1536-d: for each of the top NDIR principal directions of the pool, at both ends the
             page at the QB quantile of the projection (the border band, not the single extreme); deleted outliers (labelled) left out
             -> <out>/border/<dir>_<end>_<doc>_<page>.jpg, <out>/border.parquet
  same       pages with an identical copy (k-NN distance 0) grouped -> <out>/same/<group size>_<group>_<doc>_<page>.jpg, <out>/same.parquet
  outliers   the --n pages of largest k10 -> <out>/outliers/<rank>_<doc>_<page>.jpg (1440 x 1024), <out>/outliers.parquet

python tools/ds_border.py push | fetch | outliers [--n 1000] [--out DIR] | same [--out DIR]"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import ds1k_options as O
import ds100k as D
import kaggle_fps as KF
import numpy as np
import pandas as pd

from dspages import paths as _P  # noqa: E402

OUT = D.OUT / "border"
WORK = OUT / "kaggle"
KID = "pavel9860/ds100k-knn"
KMAX = 20
TESTS = f"{_P.ROOT}/tests/ds_border"
NDIR, QB = 500, 0.995

KERNEL = r"""
import glob, time, numpy as np, torch
X = torch.from_numpy(np.concatenate([np.load(f) for f in sorted(glob.glob("/kaggle/input/**/pool_*.npy", recursive=True))])).cuda()
n, Q = len(X), 2048
D = np.empty((n, KMAX), np.float16); I = np.empty((n, KMAX), np.int32)
t0 = time.time()
for a in range(0, n, Q):
    s = X[a:a + Q] @ X.T                                            # cosine, fp16
    s[torch.arange(len(s)), torch.arange(a, a + len(s))] = -2       # not itself
    v, i = s.topk(KMAX, dim=1)
    D[a:a + Q] = (2 - 2 * v.float()).clamp_(min=0).sqrt().cpu().numpy()
    I[a:a + Q] = i.cpu().numpy()
    if a // Q % 50 == 0:
        print(a, int(time.time() - t0), "s", flush=True)
np.savez("/kaggle/working/knn.npz", d=D, i=I)
print("DONE", n, int(time.time() - t0), "s", flush=True)
"""


def cmd_push(_):
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "run.py").write_text(f"KMAX = {KMAX}\n" + KERNEL)
    (WORK / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": KID,
                "title": KID.split("/")[1],
                "code_file": "run.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_tpu": False,
                "enable_internet": False,
                "dataset_sources": [KF.DS],
                "kernel_sources": [],
                "competition_sources": [],
                "model_sources": [],
                "machine_shape": "NvidiaTeslaT4",
            }
        )
    )
    print(KF.kaggle("kernels", "push", "-p", str(WORK)))


def cmd_fetch(_):
    st = KF.kaggle("kernels", "status", KID)
    print(st)
    if "COMPLETE" not in st.upper():
        return
    out = WORK / "out"
    out.mkdir(parents=True, exist_ok=True)
    print(KF.kaggle("kernels", "output", KID, "-p", str(out), timeout=3600)[-300:])
    z = np.load(out / "knn.npz")
    np.savez(OUT / "knn.npz", d=z["d"], i=z["i"])
    d = z["d"].astype(np.float32)
    B = pd.DataFrame({"pool_i": np.arange(len(d)), "k1": d[:, 0], "k5": d[:, 4], "k10": d[:, 9], "k20": d[:, 19]})
    B["rank"] = B.k10.rank(ascending=False, method="first").astype(int)
    B.to_parquet(OUT / "border.parquet")
    print(B[["k1", "k5", "k10", "k20"]].describe(percentiles=[0.5, 0.9, 0.99, 0.999]).round(4).to_string())


def cmd_outliers(a):
    P = pd.read_parquet(D.OUT / "pool.parquet")
    B = pd.read_parquet(OUT / "border.parquet")
    T = B.nsmallest(a.n, "rank").join(P, on="pool_i")
    d = Path(a.out) / "outliers"
    d.mkdir(parents=True, exist_ok=True)
    T["file"] = [str(d / f"{r.rank:04d}_{r.doc}_{r.page}.jpg") for r in T.itertuples()]
    T.to_parquet(Path(a.out) / "outliers.parquet")
    jobs = [(r.path, int(r.page), r.file) for r in T.itertuples() if not Path(r.file).exists()]
    with ProcessPoolExecutor(O.WORKERS) as ex:
        bad = [f for f, ls in ex.map(O.page_job, jobs, chunksize=4) if ls is None]
    print(
        len(T),
        "outlier candidates, k10 from",
        round(float(T.k10.min()), 4),
        "to",
        round(float(T.k10.max()), 4),
        "; render failed",
        len(bad),
    )
    print("folders:", T.folder.value_counts().head(8).to_dict())
    print("langs:", T.lang.value_counts().head(10).to_dict())
    print("->", d)


def pool_X():
    import glob

    return np.concatenate([np.load(f) for f in sorted(glob.glob(str(KF.WORK / "data" / "pool_*.npy")))])


def cmd_border(a):
    import torch

    torch.set_num_threads(O.WORKERS)
    X = torch.from_numpy(pool_X())
    n = len(X)
    L = pd.read_parquet(Path(a.out) / "outliers_labelled.parquet")
    bad = np.zeros(n, bool)
    bad[L.pool_i[~L.kept].to_numpy()] = True
    mu = torch.zeros(X.shape[1], dtype=torch.float64)
    C = torch.zeros(X.shape[1], X.shape[1], dtype=torch.float64)
    for s in range(0, n, 100_000):
        x = X[s : s + 100_000].float()
        mu += x.sum(0).double()
        C += (x.T @ x).double()
    mu /= n
    C = C / n - torch.outer(mu, mu)
    ev, V = torch.linalg.eigh(C)
    V = V[:, -NDIR:].flip(1).float()
    print(f"top {NDIR} directions: {float(ev[-NDIR:].sum() / ev.sum()):.3f} of the variance", flush=True)
    Y = torch.cat([(X[s : s + 100_000].float() - mu.float()) @ V for s in range(0, n, 100_000)]).numpy()
    Y[bad] = np.nan
    rows = []
    for j in range(NDIR):
        for end, q in (("lo", 1 - QB), ("hi", QB)):
            t = np.nanquantile(Y[:, j], q)
            i = int(np.nanargmin(np.abs(Y[:, j] - t)))
            rows.append((j, end, i, float(Y[i, j]), float(np.nanstd(Y[:, j]))))
    B = pd.DataFrame(rows, columns=["dir", "end", "pool_i", "proj", "sd"])
    S = X[np.unique(B.pool_i.to_numpy())].float() - mu.float()
    tot = float(ev.sum())
    print(f"samples: total variance {float(S.var(0).sum()) / tot:.2f} x the pool's", flush=True)
    Q, _ = torch.linalg.qr(S.T)  # orthonormal basis of the span of the samples
    VQ = V.T @ Q
    inspan = float((ev[-NDIR:].flip(0).float() * (VQ**2).sum(1)).sum()) / tot
    G = Q.T @ C.float() @ Q
    print(
        f"pool variance inside the span of the samples: {float(torch.trace(G)) / tot:.3f} (top-{NDIR}-direction part {inspan:.3f})",
        flush=True,
    )
    P = pd.read_parquet(D.OUT / "pool.parquet")
    B = B.join(P, on="pool_i")
    d = Path(a.out) / "border"
    d.mkdir(parents=True, exist_ok=True)
    B["file"] = [str(d / f"{r.dir:03d}_{r.end}_{r.doc}_{r.page}.jpg") for r in B.itertuples()]
    B.to_parquet(Path(a.out) / "border.parquet")
    U = B.drop_duplicates("pool_i")
    jobs = [(r.path, int(r.page), r.file) for r in U.itertuples() if not Path(r.file).exists()]
    with ProcessPoolExecutor(O.WORKERS) as ex:
        failed = [f for f, ls in ex.map(O.page_job, jobs, chunksize=4) if ls is None]
    print(len(B), "border samples,", len(U), "distinct pages; render failed", len(failed))
    print("folders:", U.folder.value_counts().head(10).to_dict())
    print("langs:", U.lang.value_counts().head(12).to_dict())
    print("->", d)


def cmd_same(a):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    z = np.load(OUT / "knn.npz")
    r, c = np.nonzero(z["d"] < 1e-3)
    n = len(z["d"])
    _, lab = connected_components(coo_matrix((np.ones(len(r)), (r, z["i"][r, c])), shape=(n, n)), directed=False)
    sz = np.bincount(lab)
    m = sz[lab] > 1
    P = pd.read_parquet(D.OUT / "pool.parquet")
    d = Path(a.out) / "same"
    d.mkdir(parents=True, exist_ok=True)
    G = P[m].assign(pool_i=np.flatnonzero(m), group=lab[m], gsize=sz[lab[m]])
    G["file"] = [str(d / f"{r.gsize:02d}_{r.group:07d}_{r.doc}_{r.page}.jpg") for r in G.itertuples()]
    G.to_parquet(Path(a.out) / "same.parquet")
    jobs = [(r.path, int(r.page), r.file) for r in G.itertuples() if not Path(r.file).exists()]
    with ProcessPoolExecutor(O.WORKERS) as ex:
        bad = [f for f, ls in ex.map(O.page_job, jobs, chunksize=8) if ls is None]
    print(len(G), "pages in", G.group.nunique(), "groups; render failed", len(bad), "->", d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("push")
    sp.add_parser("fetch")
    o = sp.add_parser("outliers")
    o.add_argument("--n", type=int, default=1000)
    o.add_argument("--out", default=TESTS)
    sp.add_parser("same").add_argument("--out", default=TESTS)
    sp.add_parser("border").add_argument("--out", default=TESTS)
    a = ap.parse_args()
    {"push": cmd_push, "fetch": cmd_fetch, "outliers": cmd_outliers, "same": cmd_same, "border": cmd_border}[a.cmd](a)
