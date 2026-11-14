"""A 3D-map space learned on frozen vectors (as tools/ds_metric.py for the pages): one head per factor (Conditional Similarity
Networks, Veit et al. 2017), each a unit vector of DIM dims, so every factor weighs the same in the concatenation. Input per map:
SigLIP 2 shade final + layer 4 (1536) | the 517 engineered features (z-scored). Maps: the items of maps3d_emb (generated + real
sources). Labels (the data's own fields; a map without the field: no term):
  folds    drawn fold count, capped at 3 (generated)          supervised contrastive loss (Khosla et al. 2020)
  deep     plan deep level (generated)                         supervised contrastive loss
  shallow  plan shallow creases (generated)                    supervised contrastive loss
  axis     plan bend_dir (generated)                           Rank-N-Contrast (Zha et al. 2023)
  real     source category (real)                              supervised contrastive loss
Held out: HOLD of the maps. On the held-out maps, for the raw input, every head and the concatenation: 10-NN balanced accuracy of
each class factor, 10-NN R2 of the axis. Data: tests_gen/maps3d_metric (OUT).

python tools/maps3d_metric.py [--steps N]"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

T = Path("/run/media/p/D/ML_DS/UVTM/GridPages/tests_gen")
EMB, FE, OUT = T / "maps3d_emb", T / "maps3d_fe", T / "maps3d_metric"
PLAN = Path("/run/media/p/D/ML_DS/UVTM/uv_ds_2.0.0/part6/plan.jsonl")
BLOCKS = ("base", "dog", "normgrad", "ridge", "gauss_img", "curv")
HOLD, DIM, BATCH, RNC_B = 0.1, 8, 256, 128
HEADS = ("folds", "deep", "shallow", "axis", "real")


def data():
    its = [json.loads(l) for l in open(FE / "items.jsonl")]
    plan = [json.loads(l) for l in open(PLAN)]
    S = np.hstack([np.load(EMB / f"emb_siglip2_{k}_shade.npy") for k in ("final", "l4")])
    Fe = np.hstack([np.load(FE / f"fb_{k}.npy") for k in BLOCKS]).astype(np.float64)
    Fe = np.nan_to_num((Fe - Fe.mean(0)) / (Fe.std(0) + 1e-9)).clip(-10, 10)
    S = S / np.linalg.norm(S, axis=1, keepdims=True) * np.sqrt(S.shape[1])          # both parts ~ unit variance per dim
    X = np.hstack([S, Fe]).astype(np.float32)
    gen = np.array([not i["real"] for i in its])
    lab = lambda vals: np.array(vals, np.int64)
    deeps = ["none", "light", "medium", "heavy"]
    cats = sorted({i["category"] for i in its if i["real"]})
    y = dict(folds=lab([min(i["n_folds"], 3) if not i["real"] else -1 for i in its]),
             deep=lab([deeps.index(i["deep"]) if not i["real"] else -1 for i in its]),
             shallow=lab([int(i["shallow"]) if not i["real"] else -1 for i in its]),
             real=lab([cats.index(i["category"]) if i["real"] else -1 for i in its]))
    ax = np.array([plan[int(i["path"][-11:-4])]["bend_dir"] if not i["real"] else np.nan for i in its], np.float32)
    return its, X, y, ax, gen


class Head(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = nn.ModuleList([nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, DIM)) for _ in HEADS])

    def forward(self, x):
        return [F.normalize(f(x), dim=1) for f in self.f]


def supcon(z, y, tau=0.1):
    s = z @ z.T / tau
    eye = torch.eye(len(z), dtype=torch.bool)
    s = s.masked_fill(eye, -1e9)
    pos = (y[:, None] == y[None]) & ~eye
    lp = s - torch.logsumexp(s, 1, keepdim=True)
    has = pos.any(1)
    return -(lp * pos).sum(1)[has].div(pos.sum(1)[has]).mean()


def rnc(z, y, tau=2.0):
    s = z @ z.T / tau
    d = torch.cdist(y, y)
    m = len(z)
    eye = torch.eye(m, dtype=torch.bool)
    s = s.masked_fill(eye, -1e9)
    loss = 0.0
    for i in range(m):
        neg = d[i][None, :] >= d[i][:, None]
        neg[:, i] = False
        den = torch.logsumexp(s[i][None, :].masked_fill(~neg, -1e9), 1)
        keep = ~eye[i]
        loss = loss + (den[keep] - s[i][keep]).mean()
    return loss / m


def bal_acc(pred, true):
    return float(np.mean([(pred[true == c] == c).mean() for c in np.unique(true)]))


def knn_eval(V, y, ax, tr, ho, k=10):
    V = torch.as_tensor(V, dtype=torch.float32)
    d = torch.cdist(V[ho], V[tr])
    r = {}
    for name, lab in y.items():
        q, rf = ho[lab[ho] >= 0], tr[lab[tr] >= 0]
        dd = torch.cdist(V[q], V[rf])
        nb = rf[dd.topk(k, largest=False).indices.numpy()]
        pred = np.array([np.bincount(lab[n]).argmax() for n in nb])
        r[name] = bal_acc(pred, lab[q])
    q, rf = ho[np.isfinite(ax[ho])], tr[np.isfinite(ax[tr])]
    nb = rf[torch.cdist(V[q], V[rf]).topk(k, largest=False).indices.numpy()]
    p, t = ax[nb].mean(1), ax[q]
    r["axis R2"] = float(1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum())
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(6)
    t0 = time.time()
    its, X, y, ax, gen = data()
    n = len(X)
    hold = np.random.default_rng(0).random(n) < HOLD
    tr, ho = np.flatnonzero(~hold), np.flatnonzero(hold)
    print(f"maps {n} (generated {int(gen.sum())}, real {int((~gen).sum())}); train {len(tr)}, held out {len(ho)}; input {X.shape[1]}", flush=True)
    E = torch.from_numpy(X)
    net = Head(X.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=a.steps)
    g = np.random.default_rng(1)
    groups = {h: [tr[y[h][tr] == c] for c in np.unique(y[h][tr]) if c >= 0 and (y[h][tr] == c).sum() >= 2] for h in y}
    axt = tr[np.isfinite(ax[tr])]
    axv = torch.from_numpy(ax)[:, None]

    def draw(gr):                                       # class-balanced batch
        per = max(2, BATCH // len(gr))
        return torch.from_numpy(np.concatenate([g.choice(x, per, replace=len(x) < per) for x in gr]))

    for step in range(a.steps):
        net.train()
        loss, parts = 0.0, {}
        for k, h in enumerate(HEADS):
            if h == "axis":
                b = torch.from_numpy(g.choice(axt, RNC_B, replace=False))
                l = rnc(net(E[b])[k], axv[b])
            else:
                b = draw(groups[h])
                l = supcon(net(E[b])[k], torch.from_numpy(y[h])[b])
            parts[h] = float(l.detach())
            loss = loss + l
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == a.steps - 1:
            print(f"[{int(time.time() - t0)} s] step {step}", {k: round(v, 3) for k, v in parts.items()}, flush=True)
    net.eval()
    with torch.no_grad():
        Z = [z.numpy() for z in net(E)]
    Z = [z / np.sqrt(((z[g.integers(0, n, 20000)] - z[g.integers(0, n, 20000)]) ** 2).sum(1).mean()) for z in Z]
    spaces = {"raw siglip+feat": X, **{f"head {h}": z for h, z in zip(HEADS, Z)}, f"concat {len(HEADS)} x {DIM}": np.hstack(Z)}
    res = {}
    for name, V in spaces.items():
        res[name] = knn_eval(V, y, ax, tr, ho)
        print(name, {k: round(v, 3) for k, v in res[name].items()}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), OUT / "head.pt")
    np.savez(OUT / "Z.npz", Z=np.hstack(Z).astype(np.float16), hold=hold, heads=np.array(HEADS))
    json.dump(res, open(OUT / "metric.json", "w"), indent=1)
    print(f"done {int(time.time() - t0)} s ->", OUT)


if __name__ == "__main__":
    main()
