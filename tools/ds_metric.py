"""A page space equally sensitive to handwriting, content and layout, learned on the existing 1536-d unit page vectors (frozen): one head per factor
(Conditional Similarity Networks, Veit et al. 2017: a subspace per notion of similarity), each a unit vector of DIM dims, so every factor weighs the same
in the concatenation. Pages: the in-scope pool (ds_spread aux mask). Labels (data of every page unless said):
  hw       hw class of the hw_det box counts (hw_class: none / filled / hw)   supervised contrastive loss (Khosla et al. 2020)
  content  manifest plan type (pages of _untyped documents: no term)  supervised contrastive loss
  layout   shard fill, pic, window x / y / width (scan_pct), log(1 + hw_det text boxes), z-scored   Rank-N-Contrast (Zha et al. 2023)
Held out: HOLD of the documents. On the Kaggle GPU: training, then on the held-out pages, for the raw 1536-d space, every head and the
concatenation: 10-NN balanced accuracy of hw, of plan type, of the human page-kind labels (pagetype_labelled, never trained on), 10-NN R2 of
the layout values; mean squared distance of random pairs per head; then k-center 100K in the concatenation over all in-scope pages.
Data: ds100k/metric (OUT).

python tools/ds_metric.py prep | push | test | fetch | report"""

import argparse
import json
import sqlite3

import ds100k as D
import ds_lda as LD
import ds_spread as S
import kaggle_fps as KF
import numpy as np
import pandas as pd

OUT = D.OUT / "metric"
AUX, KID = "pavel9860/ds-metric-aux", "pavel9860/ds-metric"
HOLD, DIM = 0.1, 8

KERNEL = r"""
import glob, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
t0 = time.time()
def log(*a):
    print(f"[{int(time.time() - t0)} s]", *a, flush=True)
def find(name):
    return sorted(glob.glob(f"{ROOT}/**/{name}", recursive=True))
torch.manual_seed(0)
mask = np.load(find("mask.npy")[0]); real = np.flatnonzero(mask)
L = np.load(find("labels.npz")[0])
X = np.concatenate([np.load(f) for f in find("pool_*.npy")])
rows = np.arange(len(real)) if SUB is None else np.sort(np.random.default_rng(0).choice(len(real), SUB, replace=False))
E = torch.from_numpy(X[real[rows]]).to(DEV)
del X
hw = torch.from_numpy(L["hw"][rows]).long().to(DEV)
typ = torch.from_numpy(L["type"][rows]).long().to(DEV)
lay = torch.from_numpy(L["layout"][rows]).float().to(DEV)
kind = L["kind"][rows]
hold = L["hold"][rows]
n = len(rows)
tr = np.flatnonzero(~hold)
log("pages", n, "train", len(tr), "held out", int(hold.sum()))


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.ModuleList([nn.Sequential(nn.Linear(1536, 512), nn.GELU(), nn.Linear(512, DIM)) for _ in range(3)])
    def forward(self, x):
        return [F.normalize(f(x.float()), dim=1) for f in self.f]


def supcon(z, y, tau=0.1):
    ok = y >= 0
    z, y = z[ok], y[ok]
    s = z @ z.T / tau
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    s = s.masked_fill(eye, -1e9)
    pos = (y[:, None] == y[None]) & ~eye
    lp = s - torch.logsumexp(s, 1, keepdim=True)
    has = pos.any(1)
    return -(lp * pos).sum(1)[has].div(pos.sum(1)[has]).mean()


def rnc(z, y, tau=2.0):
    s = z @ z.T / tau
    d = torch.cdist(y, y)
    m = len(z)
    eye = torch.eye(m, dtype=torch.bool, device=z.device)
    s = s.masked_fill(eye, -1e9)
    loss = 0.0
    for i in range(m):
        neg = d[i][None, :] >= d[i][:, None]
        neg[:, i] = False
        den = torch.logsumexp(s[i][None, :].masked_fill(~neg, -1e9), 1)
        keep = ~eye[i]
        loss = loss + (den[keep] - s[i][keep]).mean()
    return loss / m


net = Head().to(DEV)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=STEPS)
g = np.random.default_rng(1)
hw_np, typ_np = hw.cpu().numpy(), typ.cpu().numpy()
by_hw = [tr[hw_np[tr] == c] for c in (0, 1, 2)]                    # class-balanced batches: a third of every hw class
by_typ = [tr[typ_np[tr] == c] for c in range(int(typ_np.max()) + 1) if (typ_np[tr] == c).sum() >= 2]     # equal pages per plan type
def draw(groups, k):
    per = max(2, k // len(groups))
    return torch.from_numpy(np.concatenate([g.choice(x, per, replace=len(x) < per) for x in groups])).to(DEV)
for step in range(STEPS):
    bh, bc = draw(by_hw, BATCH), draw(by_typ, BATCH)
    bl = torch.from_numpy(g.choice(tr, RNC_B, replace=False)).to(DEV)
    lh = supcon(net(E[bh])[0], hw[bh])
    lc = supcon(net(E[bc])[1], typ[bc])
    ll = rnc(net(E[bl])[2], lay[bl])
    loss = lh + lc + ll
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 500 == 0 or step == STEPS - 1:
        log(f"step {step} loss hw {float(lh):.3f} content {float(lc):.3f} layout {float(ll):.3f}")

net.eval()
with torch.no_grad():
    Z = [torch.cat([net(E[a:a + 50_000])[k] for a in range(0, n, 50_000)]) for k in range(3)]
pi = torch.randint(0, n, (200_000,), device=DEV); pj = torch.randint(0, n, (200_000,), device=DEV)
share = [float(((z[pi] - z[pj]) ** 2).sum(1).mean()) for z in Z]
log("mean squared distance of random pairs per head before scaling (hw, content, layout):", [round(x, 3) for x in share])
Z = [z / s ** 0.5 for z, s in zip(Z, share)]                       # every head: mean squared random-pair distance 1
torch.save(net.state_dict(), f"{OUTDIR}/head.pt")
np.savez(f"{OUTDIR}/Z.npz", pool_i=real[rows], Z=torch.cat(Z, 1).half().cpu().numpy())
spaces = {"raw 1536": E, "hw head": Z[0], "content head": Z[1], "layout head": Z[2], "concat 3 x 8": torch.cat(Z, 1)}
ho = np.flatnonzero(hold)
q = torch.from_numpy(ho).to(DEV)
ref = torch.from_numpy(tr).to(DEV)


def knn(V, k=10):
    out = []
    R = V[ref].half()
    rs = torch.cat([(R[a:a + 100_000].float() ** 2).sum(1) for a in range(0, len(R), 100_000)])
    for a in range(0, len(q), 512):
        Q = V[q[a:a + 512]].half()
        d = rs[None] - 2 * (Q @ R.T).float()
        out.append(ref[d.topk(k, largest=False).indices])
    return torch.cat(out)


def bal_acc(pred, true):
    c = np.unique(true)
    return float(np.mean([(pred[true == x] == x).mean() for x in c]))


def vote(nb, y):
    yy = y[nb]
    return torch.mode(yy, 1).values


res = {}
hw_h, typ_h, lay_h = hw[q].cpu().numpy(), typ[q].cpu().numpy(), lay[q]
kind_h = kind[ho]
for name, V in spaces.items():
    nb = knn(V)
    r = {"hw balanced acc": bal_acc(vote(nb, hw).cpu().numpy(), hw_h)}
    t = typ_h >= 0
    pt = vote(nb, typ).cpu().numpy()
    r["plan type balanced acc"] = bal_acc(pt[t], typ_h[t])
    kq = kind_h != ""
    kind_all = kind
    # page-kind labels: vote among the neighbours that carry one (labelled reference pages)
    lab_ref = np.flatnonzero((kind_all != "") & ~hold)
    if len(lab_ref) > 50 and kq.sum() > 20:
        Rl = torch.from_numpy(lab_ref).to(DEV)
        dl = torch.cdist(V[q[torch.from_numpy(np.flatnonzero(kq)).to(DEV)]].float(), V[Rl].float())          # small: labelled pages only
        nl = lab_ref[dl.topk(10, largest=False).indices.cpu().numpy()]
        kp = np.array([max(set(kind_all[x].tolist()), key=kind_all[x].tolist().count) for x in nl])
        r["page-kind (human labels) balanced acc"] = bal_acc(kp, kind_h[kq])
    pl = lay[nb].mean(1)
    r["layout 10-NN R2"] = float(1 - ((pl - lay_h) ** 2).sum() / ((lay_h - lay_h.mean(0)) ** 2).sum())
    res[name] = r
    log(name, {k: round(v, 3) for k, v in r.items()})
C = torch.cat(Z, 1).half()
sq = (C.float() ** 2).sum(1)
dmin = torch.full((n,), float("inf"), device=DEV)
pick, acc, Bk = [], [0], 100
while True:
    pick += acc
    P = torch.tensor(acc, device=DEV)
    torch.minimum(dmin, (sq[:, None] + sq[P][None] - 2 * (C @ C[P].T).float()).clamp_(min=0).min(1).values, out=dmin)
    dmin[P] = -1
    if len(pick) >= NPICK:
        break
    v, c = torch.topk(dmin, 4 * Bk)
    acc = []
    for dv, j in zip(v.tolist(), c.tolist()):
        if dv < 0:
            break
        if acc and float((sq[acc] + sq[j] - 2 * (C[acc] @ C[j]).float()).min()) < dv:
            continue
        acc.append(j)
        if len(acc) == Bk or len(pick) + len(acc) == NPICK:
            break
log("k-center", len(pick), "covering radius", float(dmin.clamp(min=0).max().sqrt()))
np.savez(f"{OUTDIR}/metric.npz", pool_i=real[rows], Z=torch.cat(Z, 1).half().cpu().numpy(), pick=real[rows[np.array(pick)]])
json.dump({"knn": res, "pair_msd": share}, open(f"{OUTDIR}/metric.json", "w"), indent=1)
torch.save(net.state_dict(), f"{OUTDIR}/head.pt")
log("DONE")
"""


def hw_class(hw, text):
    """0 none, 1 filled (hw boxes > 2, text boxes > 10: filled forms, notes on text), 2 hw (hw > 10, text < 5)."""
    hw, text = np.asarray(hw), np.asarray(text)
    return np.where((hw > 10) & (text < 5), 2, np.where((hw > 2) & (text > 10), 1, 0)).astype(np.int8)


def cmd_prep(_):
    P = pd.read_parquet(D.OUT / "pool.parquet", columns=["doc", "page", "h", "hw", "text"])
    m = np.load(S.OUT / "aux" / "mask.npy")
    real = np.flatnonzero(m)
    R = P.iloc[real].reset_index(drop=True)
    sp = pd.read_parquet(
        S.M.LAYOUTS / "archive" / "info" / "scan_pct.parquet", columns=["doc", "page", "fill", "pic", "wx", "wy", "ww"]
    )
    R = R.merge(sp, on=["doc", "page"], how="left")
    assert len(R) == len(real)
    lay = np.column_stack([R.fill, R.pic, R.wx, R.wy, R.ww, np.log1p(R.text)]).astype(np.float32)
    lay = np.nan_to_num((lay - np.nanmean(lay, 0)) / np.nanstd(lay, 0))
    with sqlite3.connect(f"file:{S.M.LAYOUTS / 'manifest.sqlite'}?mode=ro", uri=True) as db:
        typ = dict(db.execute("select doc, type from plan"))
    t = R.h.map(typ).fillna("_untyped")
    types = sorted(set(t) - {"_untyped"})
    tid = np.array([types.index(x) if x != "_untyped" else -1 for x in t], np.int16)
    L, _ = LD.labelled()
    at = pd.Series(np.arange(len(real)), index=real)
    L = L[L.pool_i.isin(at.index)]
    kind = np.full(len(real), "", dtype="<U16")
    kind[at[L.pool_i].to_numpy()] = L.cls.to_numpy()
    docs = R.doc.unique()
    held = set(np.random.default_rng(0).choice(docs, int(HOLD * len(docs)), replace=False))
    hold = R.doc.isin(held).to_numpy()
    d = OUT / "aux"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "mask.npy", m)
    np.savez(
        d / "labels.npz",
        hw=hw_class(R.hw, R.text),
        type=tid,
        layout=lay.astype(np.float16),
        kind=kind,
        hold=hold,
        types=np.array(types),
    )
    print(
        f"pages {len(real)}; hw classes (none, filled, hw) {np.bincount(hw_class(R.hw, R.text), minlength=3).tolist()}; typed {float((tid >= 0).mean()):.3f} in {len(types)} types; "
        f"human page-kind labels {int((kind != '').sum())}; held-out pages {int(hold.sum())} of {len(held)} documents"
    )
    (d / "dataset-metadata.json").write_text(
        json.dumps({"title": AUX.split("/")[1], "id": AUX, "licenses": [{"name": "CC-BY-NC-SA-4.0"}]})
    )
    print(KF.kaggle("datasets", "version", "-p", str(d), "-m", "hw classes", timeout=3600)[-200:])


def code(root, outdir, dev, sub, steps, npick):
    return (
        f"ROOT, OUTDIR, DEV, SUB, DIM, STEPS, BATCH, RNC_B, NPICK = {root!r}, {outdir!r}, {dev!r}, {sub}, {DIM}, {steps}, 1024, 128, {npick}\n"
        + KERNEL
    )


def cmd_push(_):
    d = OUT / "kernel"
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.py").write_text(code("/kaggle/input", "/kaggle/working", "cuda", None, 6000, 100_000))
    (d / "kernel-metadata.json").write_text(
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
                "dataset_sources": [KF.DS, AUX],
                "kernel_sources": [],
                "competition_sources": [],
                "model_sources": [],
                "machine_shape": "NvidiaTeslaT4",
            }
        )
    )
    print(KF.kaggle("kernels", "push", "-p", str(d)))


def cmd_test(_):
    import glob

    import torch

    torch.set_num_threads(S.M.WORKERS)
    root = D.TESTS / "metric" / "test" / "in"
    root.mkdir(parents=True, exist_ok=True)
    for f in [*glob.glob(str(KF.WORK / "data" / "pool_*.npy")), *glob.glob(str(OUT / "aux" / "*.np*"))]:
        t = root / f.rsplit("/", 1)[1]
        if not t.exists():
            t.symlink_to(f)
    exec(
        compile(code(str(root), str(D.TESTS / "metric" / "test"), "cpu", 20_000, 30, 2000), "kernel", "exec"),
        {"__name__": "kernel"},
    )


def cmd_fetch(_):
    st = KF.kaggle("kernels", "status", KID)
    print(st)
    if "COMPLETE" not in st.upper():
        return
    out = OUT / "out"
    out.mkdir(parents=True, exist_ok=True)
    print(KF.kaggle("kernels", "output", KID, "-p", str(out))[-200:])


def cmd_report(_):
    P = pd.read_parquet(D.OUT / "pool.parquet", columns=["h", "lang", "hw", "text", "scan"])
    with sqlite3.connect(f"file:{S.M.LAYOUTS / 'manifest.sqlite'}?mode=ro", uri=True) as db:
        P["type"] = P.h.map(dict(db.execute("select doc, type from plan"))).fillna("_untyped")
    P["hw class"] = np.array(["none", "filled", "hw"])[hw_class(P.hw, P.text)]
    j = json.load(open(OUT / "out" / "metric.json"))
    print(pd.DataFrame(j["knn"]).T.round(3).to_string())
    print(
        "mean squared distance of random pairs per head before scaling to 1 (hw, content, layout):",
        [round(x, 3) for x in j["pair_msd"]],
    )
    m = np.load(S.OUT / "aux" / "mask.npy")
    raw = pd.read_parquet(D.OUT / "select.parquet").pool_i.to_numpy()
    sets = {
        "pool": np.flatnonzero(m),
        "raw 1536 k-center": raw[m[raw]],
        "head k-center": np.load(OUT / "out" / "metric.npz")["pick"],
    }
    for col in ("hw class", "scan", "type", "lang"):
        T = (
            pd.DataFrame({k: P[col].iloc[v].value_counts(normalize=True) for k, v in sets.items()})
            .fillna(0)
            .sort_values("pool", ascending=False)
        )
        print(f"\n{col} share")
        print(T.head(16).round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for c in ("prep", "push", "test", "fetch", "report"):
        sp.add_parser(c)
    a = ap.parse_args()
    {"prep": cmd_prep, "push": cmd_push, "test": cmd_test, "fetch": cmd_fetch, "report": cmd_report}[a.cmd](a)
