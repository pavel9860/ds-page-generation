"""100K equally spaced points over the in-scope page space (1536-d unit page vectors) and the real page of each: farthest-point sampling on
a Kaggle GPU over the real pages (pool of tools/ds100k.py with fill >= FILL, pic < PIC, the outliers deleted in the review left out) and
virtual points between them (normalised midpoints of k-NN pairs, ds_border knn.npz: the longest MPAIRS pairs), so the points also fall
where there is no page. Every point gets its nearest real page and the distance; a point farther than the radius from every real page is a
gap to fill. Data: ds100k/grid (OUT).

  prep     -> OUT/kaggle/aux/{mask.npy, pairs.npy} (pool rows in scope; virtual points as pool row pairs), uploaded as dataset AUX
  push     kernel KID: FPS of N points over real + virtual
  fetch    -> OUT/grid.parquet (order, virtual, a, b, nn = nearest real pool row, d = its distance, r = distance when picked)
  radius   gap counts for radius candidates (the radius that leaves 10-20K gaps)

python tools/ds_grid.py prep | push [--n 100000] | fetch | radius"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import ds100k as D
import kaggle_fps as KF

OUT = D.OUT / "grid"
WORK = OUT / "kaggle"
AUX = "pavel9860/ds100k-grid-aux"
KID = "pavel9860/ds100k-grid"
AUXW, KIDW = "pavel9860/ds100k-grid-weight", "pavel9860/ds100k-grid-w"
FILL, PIC = 0.7, 0.2
MPAIRS = 1_500_000
LABELLED = "/run/media/p/D/ML_DS/UVTM/tests/ds_border/outliers_labelled.parquet"

KERNEL = r'''
import glob, time, numpy as np, torch
X = np.concatenate([np.load(f) for f in sorted(glob.glob("/kaggle/input/**/pool_*.npy", recursive=True))])
mask = np.load(glob.glob("/kaggle/input/**/mask.npy", recursive=True)[0])
pairs = np.load(glob.glob("/kaggle/input/**/pairs.npy", recursive=True)[0])
real = np.flatnonzero(mask)
nr = len(real)
U = torch.empty((nr + len(pairs), X.shape[1]), dtype=torch.float16, device="cuda")     # candidates: real pages, then virtual points
for a in range(0, nr, 200_000):
    r = real[a:a + 200_000]
    U[a:a + len(r)] = torch.from_numpy(X[r]).cuda()
for a in range(0, len(pairs), 200_000):
    p = pairs[a:a + 200_000]
    v = torch.from_numpy(X[p[:, 0]]).cuda().float() + torch.from_numpy(X[p[:, 1]]).cuda().float()
    U[nr + a:nr + a + len(p)] = (v / v.norm(dim=1, keepdim=True)).half()
del X
wf = glob.glob("/kaggle/input/**/weight.npz", recursive=True)
if wf:                                                              # factor weights: x' = x + Q diag(s - 1) Q^T x
    z = np.load(wf[0])
    Q, s1 = torch.from_numpy(z["Q"]).cuda(), torch.from_numpy(z["s"]).cuda() - 1
    for a in range(0, len(U), 200_000):
        x = U[a:a + 200_000].float()
        U[a:a + 200_000] = (x + ((x @ Q) * s1) @ Q.T).half()
    print("weighted:", Q.shape[1], "dims", flush=True)
sq = torch.cat([(U[a:a + 200_000].float() ** 2).sum(1) for a in range(0, len(U), 200_000)])
R = U[:nr]
n = len(U)
d = torch.full((n,), float("inf"), device="cuda")
def dist2(P):
    return (sq[:, None] + sq[P][None, :] - 2 * (U @ U[P].T).float()).clamp_(min=0)
def upd(P):
    torch.minimum(d, dist2(P).min(1).values, out=d)
pick, rad, acc, r_acc = [], [], [int(np.random.default_rng(0).integers(nr))], [float("inf")]
t0 = time.time()
while True:
    pick += acc; rad += r_acc
    upd(torch.tensor(acc, device="cuda"))
    if len(pick) >= N:
        break
    v, c = torch.topk(d, 4 * B)
    acc, r_acc = [], []
    for dv, j in zip(v.tolist(), c.tolist()):
        if acc and float((sq[acc] + sq[j] - 2 * (U[acc] @ U[j]).float()).min()) < dv:
            continue
        acc.append(j); r_acc.append(dv ** 0.5)
        if len(acc) == B or len(pick) + len(acc) == N:
            break
    if len(pick) // 10000 != (len(pick) + len(acc)) // 10000:
        print(f"{len(pick) + len(acc)} picked, radius {r_acc[-1]:.4f}, {int(time.time() - t0)} s", flush=True)
pick = np.array(pick, np.int64)
P = U[torch.from_numpy(pick).cuda()]
nn, nd = [], []
del d
torch.cuda.empty_cache()
for a in range(0, len(P), 256):                                     # nearest real page of every point
    q = torch.from_numpy(pick[a:a + 256]).cuda()
    v, i = (sq[:nr][None, :] - 2 * (P[a:a + 256] @ R.T).float()).min(1)
    nn.append(real[i.cpu().numpy()]); nd.append((v + sq[q]).clamp(min=0).sqrt().cpu().numpy())
virt = pick >= nr
a_ = np.where(virt, pairs[np.clip(pick - nr, 0, None), 0], real[np.clip(pick, None, nr - 1)])
b_ = np.where(virt, pairs[np.clip(pick - nr, 0, None), 1], -1)
np.savez("/kaggle/working/grid.npz", virtual=virt, a=a_, b=b_, nn=np.concatenate(nn), d=np.concatenate(nd), r=np.array(rad, np.float32))
print("DONE", len(pick), "points,", int(virt.sum()), "virtual;", int(time.time() - t0), "s", flush=True)
'''


def cmd_prep(_):
    P = pd.read_parquet(D.OUT / "pool.parquet", columns=["fill", "pic"])
    L = pd.read_parquet(LABELLED)
    m = ((P.fill >= FILL) & (P.pic < PIC)).to_numpy().copy()
    m[L.pool_i[~L.kept].to_numpy()] = False
    z = np.load(D.OUT / "border" / "knn.npz")
    I, dd = z["i"], z["d"].astype(np.float32)
    rows = np.repeat(np.arange(len(I)), I.shape[1])
    a, b, dist = rows, I.ravel(), dd.ravel()
    k = m[a] & m[b] & (a < b) & (dist > 1e-3)
    a, b, dist = a[k], b[k], dist[k]
    top = np.argsort(-dist)[:MPAIRS]
    pairs = np.stack([a[top], b[top]], 1).astype(np.int32)
    d = WORK / "aux"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "mask.npy", m)
    np.save(d / "pairs.npy", pairs)
    print(int(m.sum()), "pages in scope;", len(pairs), "virtual pairs of", int(k.sum()), "; pair distance", round(float(dist[top].min()), 4), "-", round(float(dist[top].max()), 4))
    (d / "dataset-metadata.json").write_text(json.dumps({"title": AUX.split("/")[1], "id": AUX, "licenses": [{"name": "CC-BY-NC-SA-4.0"}]}))
    print(KF.kaggle("datasets", "create", "-p", str(d))[-300:])


def cmd_push(a):
    kid, sets = (KIDW, [KF.DS, AUX, AUXW]) if a.weighted else (KID, [KF.DS, AUX])
    d = WORK / ("kernel_w" if a.weighted else "kernel")
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.py").write_text(f"N, B = {a.n}, 100\n" + KERNEL)
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": kid, "title": kid.split("/")[1], "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "dataset_sources": sets, "kernel_sources": [],
        "competition_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4"}))
    print(KF.kaggle("kernels", "push", "-p", str(d)))


def cmd_fetch(a):
    kid, tag = (KIDW, "_w") if a.weighted else (KID, "")
    st = KF.kaggle("kernels", "status", kid)
    print(st)
    if "COMPLETE" not in st.upper():
        return
    out = WORK / f"out{tag}"
    out.mkdir(parents=True, exist_ok=True)
    print(KF.kaggle("kernels", "output", kid, "-p", str(out))[-300:])
    z = np.load(out / "grid.npz")
    G = pd.DataFrame({k: z[k] for k in ("virtual", "a", "b", "nn", "d", "r")}).reset_index(names="order")
    G.to_parquet(OUT / f"grid{tag}.parquet")
    print(len(G), "points,", int(G.virtual.sum()), "virtual; distance to the nearest real page:")
    print(G.groupby("virtual").d.describe(percentiles=[.5, .8, .9, .95, .99]).round(4).to_string())


def cmd_radius(_):
    G = pd.read_parquet(OUT / "grid.parquet")
    for q in (0.80, 0.85, 0.90, 0.95):
        r = float(G.d.quantile(q))
        print(f"radius {r:.4f}: gaps {int((G.d > r).sum())} (virtual {int((G.virtual & (G.d > r)).sum())})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("prep")
    p = sp.add_parser("push")
    p.add_argument("--n", type=int, default=100_000)
    p.add_argument("--weighted", action="store_true")
    sp.add_parser("fetch").add_argument("--weighted", action="store_true")
    sp.add_parser("radius")
    a = ap.parse_args()
    {"prep": cmd_prep, "push": cmd_push, "fetch": cmd_fetch, "radius": cmd_radius}[a.cmd](a)
