"""100K equally spaced points in the whitened page space and the pages to add. The 1536-d unit page vectors of the in-scope pool (ds_grid mask)
are centred, projected on the first NPC principal components (90 % of the variance) and whitened (every component variance 1). On a Kaggle GPU:
k nearest pages (KNN) of every page in that space; virtual points = midpoints of every page and one of its neighbours of rank RANK0..KNN (random),
so candidates also lie in holes deeper than the nearest-neighbour spacing; farthest-point sampling of N points over pages + virtual points;
nearest real page of every point. A point without a real page within the radius is a page to add. Data: ds100k/white (OUT).

  prep     -> OUT/aux/white.npz (mu, A: z = (x - mu) A), uploaded as dataset AUX
  push     kernel KID
  fetch    -> OUT/white.parquet (order, virtual, a, b, nn, d, r), OUT/scales.json (k1 / k10 / k20 medians of the pages in this space)
  gaps     points without a real page within the radius, for radius candidates

python tools/ds_white.py prep | push [--n 100000] | fetch | gaps"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

import ds100k as D
import ds_border as B
import ds_grid as G
import kaggle_fps as KF

OUT = D.OUT / "white"
AUX, KID = "pavel9860/ds100k-white-aux", "pavel9860/ds100k-white"
NPC, SAMPLE = 261, 200_000
KNN, RANK0 = 100, 10

KERNEL = r'''
import glob, json, time, numpy as np, torch
X = np.concatenate([np.load(f) for f in sorted(glob.glob("/kaggle/input/**/pool_*.npy", recursive=True))])
mask = np.load(glob.glob("/kaggle/input/**/mask.npy", recursive=True)[0])
w = np.load(glob.glob("/kaggle/input/**/white.npz", recursive=True)[0])
mu, A = torch.from_numpy(w["mu"]).cuda(), torch.from_numpy(w["A"]).cuda()
real = np.flatnonzero(mask); nr = len(real)
t0 = time.time()
Zr = torch.cat([((torch.from_numpy(X[real[a:a + 100_000]]).cuda().float() - mu) @ A).half() for a in range(0, nr, 100_000)])
del X
sq = (Zr.float() ** 2).sum(1)
kd = torch.empty((nr, KNN), dtype=torch.float16); ki = torch.empty((nr, KNN), dtype=torch.int32)
for a in range(0, nr, 1024):
    d2 = (sq[a:a + 1024, None] + sq[None] - 2 * (Zr[a:a + 1024] @ Zr.T).float()).clamp_(min=0)
    d2[torch.arange(len(d2)), torch.arange(a, a + len(d2))] = float("inf")
    v, i = d2.topk(KNN, dim=1, largest=False)
    kd[a:a + 1024] = v.sqrt().half().cpu(); ki[a:a + 1024] = i.int().cpu()
print("knn", int(time.time() - t0), "s", flush=True)
kdf = kd.float()
scales = {f"k{k}": float(kdf[:, k - 1].median()) for k in (1, 5, 10, 20, 50, 100)}
print(scales, flush=True)
rk = torch.from_numpy(np.random.default_rng(0).integers(RANK0, KNN, nr))
pb = ki[torch.arange(nr), rk].long()
pa = torch.arange(nr)
U = torch.empty((2 * nr, Zr.shape[1]), dtype=torch.float16, device="cuda")
U[:nr] = Zr
for a in range(0, nr, 200_000):
    U[nr + a:nr + a + len(pa[a:a + 200_000])] = ((Zr[pa[a:a + 200_000].cuda()].float() + Zr[pb[a:a + 200_000].cuda()].float()) / 2).half()
del Zr
sqU = (U.float() ** 2).sum(1)
n = len(U)
d = torch.full((n,), float("inf"), device="cuda")
def dist2(P):
    return (sqU[:, None] + sqU[P][None, :] - 2 * (U @ U[P].T).float()).clamp_(min=0)
pick, rad, acc, r_acc = [], [], [int(np.random.default_rng(0).integers(nr))], [float("inf")]
while True:
    pick += acc; rad += r_acc
    torch.minimum(d, dist2(torch.tensor(acc, device="cuda")).min(1).values, out=d)
    if len(pick) >= N:
        break
    v, c = torch.topk(d, 4 * B)
    acc, r_acc = [], []
    for dv, j in zip(v.tolist(), c.tolist()):
        if acc and float((sqU[acc] + sqU[j] - 2 * (U[acc] @ U[j]).float()).min()) < dv:
            continue
        acc.append(j); r_acc.append(dv ** 0.5)
        if len(acc) == B or len(pick) + len(acc) == N:
            break
    if len(pick) // 10000 != (len(pick) + len(acc)) // 10000:
        print(f"{len(pick) + len(acc)} picked, radius {r_acc[-1]:.3f}, {int(time.time() - t0)} s", flush=True)
cover = d.sqrt().cpu().numpy()
del d
torch.cuda.empty_cache()
pick = np.array(pick, np.int64)
P = U[torch.from_numpy(pick).cuda()]
nn, nd = [], []
for a in range(0, len(P), 256):
    q = torch.from_numpy(pick[a:a + 256]).cuda()
    v, i = (sqU[:nr][None, :] - 2 * (P[a:a + 256] @ U[:nr].T).float()).min(1)
    nn.append(real[i.cpu().numpy()]); nd.append((v + sqU[q]).clamp(min=0).sqrt().cpu().numpy())
virt = pick >= nr
loc = np.where(virt, pick - nr, pick)
np.savez("/kaggle/working/white.npz", virtual=virt, a=real[loc], b=np.where(virt, real[pb.numpy()[loc]], -1), nn=np.concatenate(nn),
         d=np.concatenate(nd), r=np.array(rad, np.float32), cover_real=cover[:nr], cover_virt=cover[nr:])
json.dump(scales, open("/kaggle/working/scales.json", "w"))
print("DONE", len(pick), "points,", int(virt.sum()), "virtual,", int(time.time() - t0), "s", flush=True)
'''


def cmd_prep(_):
    torch.set_num_threads(6)
    m = np.load(G.WORK / "aux" / "mask.npy")
    idx = np.sort(np.random.default_rng(0).choice(np.flatnonzero(m), SAMPLE, replace=False))
    X = torch.from_numpy(B.pool_X()[idx]).float()
    mu = X.mean(0)
    ev, V = torch.linalg.eigh(torch.cov(X.T))
    ev, V = ev.flip(0), V.flip(1)
    A = V[:, :NPC] / ev[:NPC].sqrt()
    print(f"{NPC} components: {float(ev[:NPC].sum() / ev.sum()):.3f} of the variance; component sd {float(ev[0].sqrt()):.4f} .. {float(ev[NPC - 1].sqrt()):.4f}")
    d = OUT / "aux"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "white.npz", mu=mu.numpy().astype(np.float32), A=A.numpy().astype(np.float32))
    np.save(d / "mask.npy", m)
    (d / "dataset-metadata.json").write_text(json.dumps({"title": AUX.split("/")[1], "id": AUX, "licenses": [{"name": "CC-BY-NC-SA-4.0"}]}))
    print(KF.kaggle("datasets", "create", "-p", str(d))[-200:])


def cmd_push(a):
    d = OUT / "kernel"
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.py").write_text(f"N, B, KNN, RANK0 = {a.n}, 100, {KNN}, {RANK0}\n" + KERNEL)
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": KID, "title": KID.split("/")[1], "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "dataset_sources": [KF.DS, AUX], "kernel_sources": [],
        "competition_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4"}))
    print(KF.kaggle("kernels", "push", "-p", str(d)))


def cmd_fetch(_):
    st = KF.kaggle("kernels", "status", KID)
    print(st)
    if "COMPLETE" not in st.upper():
        return
    out = OUT / "out"
    out.mkdir(parents=True, exist_ok=True)
    print(KF.kaggle("kernels", "output", KID, "-p", str(out))[-200:])
    z = np.load(out / "white.npz")
    W = pd.DataFrame({k: z[k] for k in ("virtual", "a", "b", "nn", "d", "r")}).reset_index(names="order")
    W.to_parquet(OUT / "white.parquet")
    (OUT / "scales.json").write_text((out / "scales.json").read_text())
    print(len(W), "points,", int(W.virtual.sum()), "virtual; covering radius", round(float(W.r.iloc[-1]), 3), "; scales", (out / "scales.json").read_text())
    print(W.groupby("virtual").d.describe(percentiles=[.5, .8, .9, .95, .99]).round(3).to_string())


def cmd_gaps(_):
    W = pd.read_parquet(OUT / "white.parquet")
    s = json.loads((OUT / "scales.json").read_text())
    rc = float(W.r.iloc[-1])
    for name, r in [(f"{k} median {v:.3f}", v) for k, v in s.items()] + [(f"covering radius x {f}", rc * f) for f in (0.5, 0.75, 1.0)]:
        print(f"radius {r:.3f} ({name}): pages to add {int((W.d > r).sum())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("prep")
    sp.add_parser("push").add_argument("--n", type=int, default=100_000)
    sp.add_parser("fetch")
    sp.add_parser("gaps")
    a = ap.parse_args()
    {"prep": cmd_prep, "push": cmd_push, "fetch": cmd_fetch, "gaps": cmd_gaps}[a.cmd](a)
