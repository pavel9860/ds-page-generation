"""The page space rescaled by the existing labels: the 1536-d unit page vectors whitened by the within-class covariance of the labelled pages
(LDA metric, z = (x - mu) Sw^-1/2, Sw shrunk by SHRINK), classes = the human page-kind labels (pagetype_labelled cache/pagekind_full.npz,
pages that are in the pool) and the reviewed real handwritten pages (Labels/real_hw_review/good: the page of the document with the most hw
boxes). Then farthest-point sampling of N pool pages (ds_grid mask) in that space on a Kaggle GPU. Data: ds100k/lda (OUT).

  fit      -> OUT/aux/lda.npz (mu, A), class counts; dataset AUX
  push     kernel KID
  fetch    -> OUT/select.parquet (order, pool_i, r)
  compare  label / hw / type / language shares: pool, raw 1536-d select (ds100k select.parquet), LDA select

python tools/ds_lda.py fit | push [--n 100000] | fetch | compare"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import ds100k as D
import ds_border as B
import ds_grid as G
import ds_weight as WT
import kaggle_fps as KF

OUT = D.OUT / "lda"
AUX, KID = "pavel9860/ds100k-lda-aux", "pavel9860/ds100k-lda"
SHRINK = 0.1
KINDS = "/run/media/p/D/ML_DS/UVTM/Labels/pagetype_labelled/cache/pagekind_full.npz"
HW_GOOD = Path("/run/media/p/D/ML_DS/UVTM/Labels/real_hw_review/good")

KERNEL = r'''
import glob, time, numpy as np, torch
X = np.concatenate([np.load(f) for f in sorted(glob.glob("/kaggle/input/**/pool_*.npy", recursive=True))])
mask = np.load(glob.glob("/kaggle/input/**/mask.npy", recursive=True)[0])
w = np.load(glob.glob("/kaggle/input/**/lda.npz", recursive=True)[0])
mu, A = torch.from_numpy(w["mu"]).cuda(), torch.from_numpy(w["A"]).cuda()
real = np.flatnonzero(mask); n = len(real)
U = torch.empty((n, A.shape[1]), dtype=torch.float16, device="cuda")
for a in range(0, n, 100_000):
    r = real[a:a + 100_000]
    U[a:a + len(r)] = ((torch.from_numpy(X[r]).cuda().float() - mu) @ A).half()
del X
sq = torch.cat([(U[a:a + 100_000].float() ** 2).sum(1) for a in range(0, n, 100_000)])
d = torch.full((n,), float("inf"), device="cuda")
pick, rad, acc, r_acc = [], [], [int(np.random.default_rng(0).integers(n))], [float("inf")]
t0 = time.time()
while True:
    pick += acc; rad += r_acc
    P = torch.tensor(acc, device="cuda")
    torch.minimum(d, (sq[:, None] + sq[P][None] - 2 * (U @ U[P].T).float()).clamp_(min=0).min(1).values, out=d)
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
        print(f"{len(pick) + len(acc)} picked, radius {r_acc[-1]:.3f}, {int(time.time() - t0)} s", flush=True)
np.savez("/kaggle/working/lda.npz", pick=real[np.array(pick)], r=np.array(rad, np.float32), cover=d.sqrt().cpu().numpy())
print("DONE", len(pick), int(time.time() - t0), "s", flush=True)
'''


def labelled():
    """(pool_i, class) of the labelled pool pages."""
    P = pd.read_parquet(D.OUT / "pool.parquet", columns=["h", "page", "hw"]).reset_index(names="pool_i")
    z = np.load(KINDS)
    L = pd.DataFrame({"h": z["doc"], "page": z["page"], "cls": z["label"]}).merge(P, on=["h", "page"])[["pool_i", "cls"]]
    hs = {f.name.split("_")[2] for f in HW_GOOD.glob("*.jpg")}
    H = P[P.h.isin(hs)].sort_values("hw").drop_duplicates("h", keep="last")
    L = pd.concat([L, pd.DataFrame({"pool_i": H.pool_i, "cls": "handwritten"})]).drop_duplicates("pool_i", keep="last")
    return L, len(hs)


def cmd_fit(_):
    torch.set_num_threads(6)
    L, nh = labelled()
    L = L[L.cls.map(L.cls.value_counts()) >= 20]
    print(f"labelled pool pages {len(L)} ({nh} reviewed handwritten documents):", L.cls.value_counts().to_dict())
    X = torch.from_numpy(B.pool_X()[L.pool_i.to_numpy()]).float()
    y = L.cls.to_numpy()
    Sw = torch.zeros(X.shape[1], X.shape[1])
    for c in np.unique(y):
        Xc = X[torch.from_numpy(y == c)]
        Xc = Xc - Xc.mean(0)
        Sw += Xc.T @ Xc
    Sw /= len(X)
    Sw = (1 - SHRINK) * Sw + SHRINK * torch.trace(Sw) / Sw.shape[0] * torch.eye(Sw.shape[0])
    ev, V = torch.linalg.eigh(Sw)
    A = V / ev.sqrt()
    m = np.load(G.WORK / "aux" / "mask.npy")
    mu = torch.from_numpy(B.pool_X()[np.flatnonzero(m)[::20]]).float().mean(0)
    d = OUT / "aux"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "lda.npz", mu=mu.numpy(), A=A.numpy().astype(np.float32))
    np.save(d / "mask.npy", m)
    Z = (X - mu) @ A
    cen = {c: Z[torch.from_numpy(y == c)].mean(0) for c in np.unique(y)}
    within = np.mean([float((Z[torch.from_numpy(y == c)] - cen[c]).norm(dim=1).mean()) for c in cen])
    between = np.mean([float((cen[a] - cen[b]).norm()) for a in cen for b in cen if a < b])
    Xr = X - X.mean(0)
    cr = {c: Xr[torch.from_numpy(y == c)].mean(0) for c in cen}
    print(f"mean distance to own class centre / between class centres: raw {np.mean([float((Xr[torch.from_numpy(y == c)] - cr[c]).norm(dim=1).mean()) for c in cr]):.3f} / "
          f"{np.mean([float((cr[a] - cr[b]).norm()) for a in cr for b in cr if a < b]):.3f}; rescaled {within:.2f} / {between:.2f}")
    (d / "dataset-metadata.json").write_text(json.dumps({"title": AUX.split("/")[1], "id": AUX, "licenses": [{"name": "CC-BY-NC-SA-4.0"}]}))
    print(KF.kaggle("datasets", "create", "-p", str(d))[-150:])


def cmd_push(a):
    d = OUT / "kernel"
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.py").write_text(f"N, B = {a.n}, 100\n" + KERNEL)
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
    print(KF.kaggle("kernels", "output", KID, "-p", str(out))[-150:])
    z = np.load(out / "lda.npz")
    pd.DataFrame({"pool_i": z["pick"], "r": z["r"]}).reset_index(names="order").to_parquet(OUT / "select.parquet")
    print(len(z["pick"]), "picked; covering radius", round(float(z["r"][-1]), 3))


def cmd_compare(_):
    P = pd.read_parquet(D.OUT / "pool.parquet", columns=["h", "lang", "hw", "text"])
    m = np.load(G.WORK / "aux" / "mask.npy")
    Lw, _, _, _ = WT.labels(P)
    L, _ = labelled()
    cls = pd.Series("unlabelled", index=P.index)
    cls[L.pool_i.to_numpy()] = L.cls.to_numpy()
    raw = pd.read_parquet(D.OUT / "select.parquet").pool_i.to_numpy()
    sets = {"pool": np.flatnonzero(m), "raw": raw[m[raw]], "lda": pd.read_parquet(OUT / "select.parquet").pool_i.to_numpy()}
    print("pages per set:", {k: len(v) for k, v in sets.items()})
    for name, s in (("labelled class (count)", cls), ("hw", Lw.hw), ("type", Lw.type.fillna("untyped")), ("lang", P.lang)):
        T = pd.DataFrame({k: s.iloc[v].value_counts(normalize=name != "labelled class (count)") for k, v in sets.items()}).fillna(0)
        T = T.sort_values("pool", ascending=False)
        if name == "lang":
            for k in sets:
                p = T[k][T[k] > 0]
                print(f"  lang {k}: {len(p)} languages, entropy-normalised spread {float(-(p * np.log(p)).sum() / np.log(len(p))):.3f}")
            T = T.head(12)
        if name == "labelled class (count)":
            T = T.drop(index="unlabelled")
        print(name)
        print(T.round(4).to_string(), "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("fit")
    sp.add_parser("push").add_argument("--n", type=int, default=100_000)
    sp.add_parser("fetch")
    sp.add_parser("compare")
    a = ap.parse_args()
    {"fit": cmd_fit, "push": cmd_push, "fetch": cmd_fetch, "compare": cmd_compare}[a.cmd](a)
