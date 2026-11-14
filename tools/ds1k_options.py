"""1K page sets of the in-scope pool (tools/ds100k.py pool: fill > 0.6, pic < 0.3, no out-of-scope folder, not warped) by several selection
rules on Z (PCA-128 of the 1536-d SigLIP 2 vectors), their spread numbers, and 1440 x 1024 raw RGB npz pages. Data: Layouts/archive/info/ds100k/opt1k (OUT).

  random     uniform over the pool pages (density-proportional baseline)
  fps        farthest-point sampling (k-center greedy, Sener & Savarese 2018) over the whole pool: even spacing, reaches the edge pages
  fps_in     fps over the pool without its edge pages (distance to the K-th nearest of a REF-page sample above the Q quantile): even inside the borders
  kmeans     k-means into N cells, the page nearest each center (density-weighted representatives)
  hkm        hierarchical k-means balanced sampling (Vo et al. 2024, DINOv2 data curation): levels L, an equal share per top cluster, split
             equally down the levels, a random page at the bottom

  select   -> OUT/<option>.parquet, OUT/metrics.parquet
  render   -> <out>/pages/<doc>_<page>.npz (rgb uint8 1440 x 1024 x 3, savez_compressed, no JPEG) or .jpg (--fmt jpg, quality 92),
              <out>/<option>/ hard links (default out: OUT)
  plot     -> <out>/positions.png: the picks of every option over the pool density, on PCA 1-2 and on t-SNE (SAMPLE pool pages + the picks)

python tools/ds1k_options.py select [--n 1000] | render [--out DIR] [--fmt npz|jpg] | plot [--out DIR]"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import ds100k as D

OUT = D.OUT / "opt1k"
K, REF, Q = 10, 50_000, 0.98
LEVELS = (20_000, 2_000, 200)
H, W = 1440, 1024
WORKERS = 6
OPTIONS = ("random", "fps", "fps_in", "kmeans", "hkm")
SAMPLE = 20_000


def knn_dist(Z, R, k, block=2_000):
    """Distance of every row of Z to its k-th nearest row of R (a row of Z that is in R skips itself: k + 1 taken, the first dropped when 0)."""
    import torch
    torch.set_num_threads(WORKERS)
    Rt = torch.from_numpy(R)
    rn = (Rt ** 2).sum(1)
    out = []
    for a in range(0, len(Z), block):
        z = torch.from_numpy(Z[a:a + block])
        d = ((z ** 2).sum(1, keepdim=True) - 2 * z @ Rt.T + rn).clamp_(min=0)
        v = d.topk(k + 1, dim=1, largest=False).values
        out.append(torch.where(v[:, 0] < 1e-10, v[:, k], v[:, k - 1]).sqrt().numpy())
    return np.concatenate(out)


def kmeans(X, k, seed=0):
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(k, random_state=seed, batch_size=max(8192, 3 * k), n_init=1, max_iter=50).fit(X)
    return km.cluster_centers_.astype(np.float32), km.labels_


def hkm(Z, n, rng):
    """Balanced sampling down a k-means tree: level 0 on the pages, each next level on the centers of the one below."""
    labs, X = [], Z
    for k in LEVELS:
        X, lab = kmeans(X, k)
        labs.append(lab)
    leaf = labs[0]                                                      # page -> level-0 cluster
    paths = [leaf]
    for lab in labs[1:]:
        paths.append(lab[paths[-1]])                                    # page -> cluster at each level
    pick = []

    def take(mask, level, m):
        if m <= 0 or not mask.any():
            return
        if level < 0:
            pick.extend(rng.choice(np.flatnonzero(mask), min(m, int(mask.sum())), replace=False))
            return
        kids = np.unique(paths[level][mask])
        rng.shuffle(kids)
        share = np.full(len(kids), m // len(kids)) + (np.arange(len(kids)) < m % len(kids))
        for c, s in zip(kids, share):
            take(mask & (paths[level] == c), level - 1, int(s))

    top = paths[-1]
    for c, s in zip(*np.unique(rng.permutation(np.arange(n) % LEVELS[-1]), return_counts=True)):
        take(top == c, len(LEVELS) - 2, int(s))
    return np.array(pick[:n])


def metrics(name, idx, Z, P, edge):
    S = P.iloc[idx]
    _, cov = D.nearest(Z, Z[idx])
    nn = knn_dist(Z[idx], Z[idx], 1)
    fs = S.folder.value_counts(normalize=True)
    return dict(option=name, pages=len(S), docs=S.doc.nunique(), langs=S.lang.nunique(),
                cover_mean=cov.mean(), cover_p50=np.median(cov), cover_p99=np.quantile(cov, 0.99), cover_max=cov.max(),
                set_nn_mean=nn.mean(), set_nn_min=nn.min(), edge_share=edge[idx].mean(),
                en_share=(S.lang == "en").mean(), hw_share=(S.hw > 0).mean(), fill_mean=S.fill.mean(), pic_mean=S.pic.mean(),
                text_mean=S.text.mean(), top_folders=", ".join(f"{k} {v:.2f}" for k, v in fs.head(4).items()))


def cmd_select(a):
    P = pd.read_parquet(D.OUT / "pool.parquet")
    Z = np.load(D.OUT / "Z.npy")
    rng = np.random.default_rng(0)
    kd = knn_dist(Z, Z[np.sort(rng.choice(len(Z), REF, replace=False))], K)
    edge = kd > np.quantile(kd, Q)
    inner = np.flatnonzero(~edge)
    print(f"pool {len(Z)}, edge pages (k{K} distance > q{Q}) {int(edge.sum())}", flush=True)
    sel = {"random": rng.choice(len(Z), a.n, replace=False),
           "fps": D.fps_plain(Z, a.n),
           "fps_in": inner[D.fps_plain(Z[inner], a.n)]}
    C, _ = kmeans(Z, a.n)
    sel["kmeans"] = D.nearest(C, Z)[0]
    sel["hkm"] = hkm(Z, a.n, np.random.default_rng(0))
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, idx in sel.items():
        idx = pd.unique(idx)
        P.iloc[idx].assign(pool_i=idx, kd=kd[idx], edge=edge[idx]).to_parquet(OUT / f"{name}.parquet")
        rows.append(metrics(name, idx, Z, P, edge))
        print(name, "done", flush=True)
    M = pd.DataFrame(rows).set_index("option")
    M.to_parquet(OUT / "metrics.parquet")
    pd.set_option("display.width", 250)
    print(M.drop(columns="top_folders").round(4).to_string())
    print(M.top_folders.to_string())


def page_job(job):
    import cv2
    import diverse_ds as DD
    path, page, out = job
    try:
        im = DD.load_page(path, page, H)
        if im is None:
            return out, None
        landscape = im.shape[1] > im.shape[0]
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA if im.shape[0] * im.shape[1] > W * H else cv2.INTER_CUBIC)
        if out.endswith(".jpg"):
            cv2.imwrite(out, im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        else:
            np.savez_compressed(out, rgb=np.ascontiguousarray(im[:, :, ::-1]), path=path, page=page)
        return out, landscape
    except Exception:
        return out, None


def cmd_render(a):
    out_dir, ext = Path(a.out), "." + a.fmt
    pages = out_dir / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    S = {o: pd.read_parquet(OUT / f"{o}.parquet") for o in OPTIONS}
    U = pd.concat(S.values()).drop_duplicates(["doc", "page"])
    jobs = [(r.path, int(r.page), str(pages / f"{r.doc}_{r.page}{ext}")) for r in U.itertuples() if not (pages / f"{r.doc}_{r.page}{ext}").exists()]
    print(len(U), "pages,", len(jobs), "to render", flush=True)
    bad, land = [], 0
    with ProcessPoolExecutor(WORKERS) as ex:
        for i, (out, ls) in enumerate(ex.map(page_job, jobs, chunksize=4)):
            if ls is None:
                bad.append(out)
            land += bool(ls)
            if i % 500 == 0:
                print(i, flush=True)
    print("rendered", len(jobs) - len(bad), "failed", len(bad), "landscape (stretched)", land)
    for o, s in S.items():
        d = out_dir / o
        d.mkdir(exist_ok=True)
        n = 0
        for r in s.itertuples():
            f = pages / f"{r.doc}_{r.page}{ext}"
            if f.exists():
                if not (d / f.name).exists():
                    os.link(f, d / f.name)
                n += 1
        print(o, n, "pages ->", d)


def cmd_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    Z = np.load(D.OUT / "Z.npy", mmap_mode="r")
    S = {o: pd.read_parquet(OUT / f"{o}.parquet").pool_i.to_numpy() for o in OPTIONS}
    picks = np.unique(np.concatenate(list(S.values())))
    bg = np.setdiff1d(np.sort(np.random.default_rng(0).choice(len(Z), SAMPLE, replace=False)), picks)
    rows = np.concatenate([bg, picks])
    T = TSNE(2, init="pca", random_state=0, n_jobs=WORKERS).fit_transform(np.asarray(Z[rows], np.float32))
    tpos = dict(zip(rows, T))
    Zp = np.asarray(Z[np.sort(np.random.default_rng(1).choice(len(Z), 300_000, replace=False))][:, :2])
    fig, ax = plt.subplots(2, len(OPTIONS), figsize=(5 * len(OPTIONS), 10), squeeze=False)
    for j, o in enumerate(OPTIONS):
        p = S[o]
        a0, a1 = ax[0, j], ax[1, j]
        a0.hexbin(Zp[:, 0], Zp[:, 1], gridsize=80, bins="log", cmap="Greys")
        a0.scatter(Z[np.sort(p), 0], Z[np.sort(p), 1], s=4, c="tab:red")
        a0.set_title(f"{o}: PCA 1-2")
        a1.hexbin(T[:len(bg), 0], T[:len(bg), 1], gridsize=80, bins="log", cmap="Greys")
        tp = np.array([tpos[i] for i in p])
        a1.scatter(tp[:, 0], tp[:, 1], s=4, c="tab:red")
        a1.set_title(f"{o}: t-SNE ({len(bg)} pool pages + picks)")
    fig.tight_layout()
    out = Path(a.out) / "positions.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print("->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("select")
    s.add_argument("--n", type=int, default=1000)
    r = sp.add_parser("render")
    r.add_argument("--out", default=str(OUT))
    r.add_argument("--fmt", choices=("npz", "jpg"), default="npz")
    p = sp.add_parser("plot")
    p.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    {"select": cmd_select, "render": cmd_render, "plot": cmd_plot}[a.cmd](a)
