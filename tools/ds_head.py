"""One head: the 1536-d page vectors -> a 16-d unit space equally sensitive to page kind and handwriting. Labels: page kind = the class
folder of every hand-labelled file of Labels/pagetype_labelled (images, pages_photo out; as labelled, nothing relabelled); handwriting =
ds_metric.hw_class of the hw_det box counts (none / filled / hw) of the in-scope pages. Both supervised contrastive losses (Khosla et al.
2020) on the same output, class-balanced batches. Held out: HOLD of the labelled documents and of the in-scope pages; report: 10-NN
balanced accuracy of kind and of hw on them. -> OUT/head.pt, OUT/report.json

python tools/ds_head.py train [--steps 3000] [--unif 1.0]"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import ds100k as D
import ds_metric as DM
import ds_targets as DT
import kaggle_fps as KF

LS = Path("/run/media/p/D/ML_DS/UVTM/Labels/labelset")
OUT = D.OUT / "head16"
DIM, HOLD, BATCH, TAU = 16, 0.1, 512, 0.1


def pool_rows():
    """Row getter of the pool vectors (pool_*.npy parts, pool.parquet order)."""
    parts = [np.load(x, mmap_mode="r") for x in sorted(glob.glob(str(KF.WORK / "data" / "pool_*.npy")))]
    starts = np.cumsum([0] + [len(x) for x in parts])

    def get(r):
        q = np.searchsorted(starts, r, side="right") - 1
        return np.stack([parts[a][b - starts[a]] for a, b in zip(q, r)]).astype(np.float32)
    return get


def kinds(get):
    """(vectors, class, document) of the labelled files: the corpus vector where the file is a corpus page, else the Kaggle one."""
    I = pd.read_parquet(LS / "items_v2.parquet")
    I = I[I.path.map(lambda f: Path(f).exists())].reset_index(drop=True)
    E = np.full((len(I), 1536), np.nan, np.float32)
    a = np.flatnonzero(I.pool_i.notna())
    E[a] = get(I.pool_i.to_numpy()[a].astype(np.int64))
    at, V = {}, []
    for f in sorted(LS.glob("out*/**/ids.npy")):                     # every Kaggle embedding run of the label files
        ids = np.load(f)[:, 0]
        at.update({int(k): (len(V), i) for i, k in enumerate(ids)})
        V.append(np.load(f.parent / "embeddings.npy"))
    b = np.array([i for i, k in enumerate(I.k) if np.isnan(E[i, 0]) and int(k) in at], int)
    E[b] = np.stack([V[at[int(k)][0]][at[int(k)][1]] for k in I.k.to_numpy()[b]])
    ok = ~np.isnan(E[:, 0])
    doc = I.path.map(lambda p: (m.group(0) if (m := re.search(r"[0-9a-f]{12}", Path(p).name)) else Path(p).stem))
    return E[ok], I.cls.to_numpy()[ok], doc.to_numpy()[ok]


def supcon(z, y):
    s = z @ z.T / TAU
    s = s - torch.eye(len(z)) * 1e9
    pos = (y[:, None] == y[None]).float() - torch.eye(len(z))
    lp = s - torch.logsumexp(s, 1, keepdim=True)
    return -((lp * pos).sum(1) / pos.sum(1).clamp(min=1)).mean()


def knn_bal(Zq, yq, Zr, yr, k=10):
    pred = []
    for a in range(0, len(Zq), 2048):
        nb = torch.cdist(Zq[a:a + 2048], Zr).topk(k, largest=False).indices
        pred.append(torch.mode(yr[nb], 1).values)
    pred = torch.cat(pred).numpy()
    yq = yq.numpy()
    return float(np.mean([(pred[yq == c] == c).mean() for c in np.unique(yq)]))


def cmd_train(a):
    torch.manual_seed(0)
    torch.set_num_threads(6)
    g = np.random.default_rng(0)
    get = pool_rows()
    Ek, ck, dk = kinds(get)
    names = sorted(set(ck))
    yk = np.array([names.index(c) for c in ck])
    hk = np.isin(dk, g.choice(np.unique(dk), int(HOLD * len(np.unique(dk))), replace=False))
    R = DT.pages()
    pool_i = np.load(DT.OUT / "aux" / "Z.npz")["pool_i"]
    yh = DM.hw_class(R.hw, R.text)
    hh = R.doc.isin(g.choice(R.doc.unique(), int(HOLD * R.doc.nunique()), replace=False)).to_numpy()
    # hw training pages: every filled / hw page, an equal-size sample of none (all classes balanced in the batches anyway)
    tr_h = np.flatnonzero(~hh)
    keep = np.concatenate([tr_h[yh[tr_h] > 0], g.choice(tr_h[yh[tr_h] == 0], 60_000, replace=False)])
    Eh, yh_tr = torch.from_numpy(get(pool_i[keep])), torch.from_numpy(yh[keep]).long()
    Xk, Yk = torch.from_numpy(Ek), torch.from_numpy(yk)
    by_k = [np.flatnonzero((~hk) & (yk == c)) for c in range(len(names))]
    by_h = [np.flatnonzero(yh_tr.numpy() == c) for c in range(3)]
    print(f"kind: {len(Ek)} labelled pages, {len(names)} classes, held out {int(hk.sum())}; hw training pages {len(keep)} "
          f"{np.bincount(yh[keep]).tolist()}", flush=True)
    net = nn.Sequential(nn.Linear(1536, 512), nn.GELU(), nn.Linear(512, DIM))
    f = lambda x: F.normalize(net(x), dim=1)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=a.steps)

    def draw(groups):
        per = max(2, BATCH // len(groups))
        return np.concatenate([g.choice(x, per, replace=len(x) < per) for x in groups if len(x)])
    for s in range(a.steps):
        bk, bh = draw(by_k), draw(by_h)
        zh = f(Eh[bh])
        lk, lh = supcon(f(Xk[bk]), Yk[bk]), supcon(zh, yh_tr[bh])
        w = zh[yh_tr[bh] > 0]                                         # pages with handwriting spread over their region (Wang & Isola 2020)
        lu = torch.pdist(w).pow(2).mul(-2).exp().mean().log()
        opt.zero_grad()
        (lk + lh + a.unif * lu).backward()
        opt.step()
        sch.step()
        if s % 500 == 0 or s == a.steps - 1:
            print(f"step {s} loss kind {float(lk):.3f} hw {float(lh):.3f} spread {float(lu):.3f}", flush=True)
    net.eval()
    with torch.no_grad():
        Zk = f(Xk)
        qh = g.choice(np.flatnonzero(hh), 20_000, replace=False)
        qh = np.concatenate([qh, np.flatnonzero(hh & (yh > 0))])
        Zq = torch.cat([f(torch.from_numpy(get(pool_i[qh[x:x + 20_000]]))) for x in range(0, len(qh), 20_000)])
        Zr = torch.cat([f(Eh[x:x + 50_000]) for x in range(0, len(Eh), 50_000)])
        rep = {"kind 10-NN balanced acc (held-out labelled docs)": knn_bal(Zk[torch.from_numpy(hk)], Yk[torch.from_numpy(hk)], Zk[torch.from_numpy(~hk)], Yk[torch.from_numpy(~hk)]),
               "hw 10-NN balanced acc (held-out pages)": knn_bal(Zq, torch.from_numpy(yh[qh]), Zr, yh_tr)}
        yq = yh[qh]
        for c, nm in enumerate(("none", "filled", "hw")):              # mean distance to the class centre: the size of its region
            zc = Zq[torch.from_numpy(yq == c)]
            rep[f"spread {nm}"] = float((zc - zc.mean(0)).norm(dim=1).mean())
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), OUT / "head.pt")
    json.dump({"classes": names, **rep}, open(OUT / "report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--unif", type=float, default=1.0, help="weight of the spread loss of the pages with handwriting")
    a = ap.parse_args()
    cmd_train(a)
