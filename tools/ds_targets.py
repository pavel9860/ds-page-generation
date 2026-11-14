"""N points (200K) equally spaced inside the border of the in-scope pages in the 16-d head space (tools/ds_head.py), most of them taking a real page and the
rest being gaps (pages to download / generate).
  prep      the heads applied locally: Z (in-scope pages), scan / hw / text boxes per page (aux dataset)
  push      aux dataset + GPU kernel; fetch brings points.npz (the points, their 16 nearest pages) and the border-aware k-center spacing
  assign    a point takes its nearest unused page: a digital-born page within FACTOR x r (r = half the covering radius), a scan only with more than
            HW_KEEP hw boxes within r (its key is FACTOR x distance); at most SCAN_CAP scans (hw, filled first), topped up with the nearest other
            scans within r; --apply writes targets.npz / targets.json (the gaps: kernel vector, 10 nearest pages per head)
  build     existing.csv (points with a page) / targets.csv (gaps: hw class vote, plan type vote, fill / pic / window / text boxes, language, 3 nearest pages)
  scans     full-resolution jpgs of the scan pages not warp-checked yet
hw class of a page (yolo hw / text box counts): hw (hw > 10, text < 5), filled (hw > 2, text > 10: filled forms, notes on text), none.
Scope: page fill (margins included) > PAGE_FILL, pic < PIC. Data: ds100k/targets (OUT). The gaps are filled by `python -m dspages.acquire`.

python tools/ds_targets.py prep | push | fetch | assign | build | scans | test   [--n 200000]"""
import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import ds100k as D
import ds_metric as DM
import ds_spread as S
import kaggle_fps as KF

OUT = D.OUT / "targets"
KID = "pavel9860/ds-targets"
AUX = "pavel9860/ds-targets-aux"
HW_KEEP = 1                                                         # scan pages with more handwriting boxes stay in the set
SCAN_CAP, FACTOR = 20_000, 1.2                                      # real scans in the set at most; a digital-born page is taken within FACTOR x the radius (and beats a scan up to FACTOR x nearer)
PAGE_FILL, PIC = 0.6, 0.2                                          # fill of the whole page (margins included), pic inside the window

KERNEL = r'''
import glob, json, time, numpy as np, torch
t0 = time.time()
def log(*a):
    print(f"[{int(time.time() - t0)} s]", *a, flush=True)
z = np.load(sorted(glob.glob(f"{ROOT}/**/Z.npz", recursive=True))[0])
Z = torch.from_numpy(z["Z"].astype(np.float32)).to(DEV)
if SUB is not None:
    Z = Z[torch.from_numpy(np.sort(np.random.default_rng(0).choice(len(Z), SUB, replace=False))).to(DEV)]
n = len(Z)
heads = [slice(0, Z.shape[1])]                                     # one unit space (ds_head)
rad = [float(Z[:, h].norm(dim=1).mean()) for h in heads]
log("pages", n, "head radii", [round(r, 3) for r in rad])


def knn(Q, R, k, skip_self=False):
    """k nearest rows of R for every row of Q (squared distances, indices)."""
    rs = (R ** 2).sum(1)
    Rh = R.half()
    D, I = [], []
    for a in range(0, len(Q), 256):
        q = Q[a:a + 256]
        d = (rs[None] + (q ** 2).sum(1)[:, None] - 2 * (q.half() @ Rh.T).float()).clamp_(min=0)
        if skip_self:
            d[torch.arange(len(q), device=DEV), torch.arange(a, a + len(q), device=DEV)] = float("inf")
        v, i = d.topk(k, largest=False)
        D.append(v); I.append(i)
    return torch.cat(D), torch.cat(I)


d10, nb = knn(Z, Z, 10, skip_self=True)
rho = d10[:, -1]                                                    # squared 10-NN distance of every page: its ball of the region
log("10-NN done")
g = torch.Generator(device=DEV).manual_seed(0)
b = nb[torch.arange(n, device=DEV), torch.randint(0, 10, (n,), device=DEV, generator=g)]
t = torch.rand(n, 1, device=DEV, generator=g)
M = (1 - t) * Z + t * Z[b]
for h, r in zip(heads, rad):
    M[:, h] = M[:, h] / M[:, h].norm(dim=1, keepdim=True) * r
inside = torch.zeros(n, dtype=torch.bool, device=DEV)
for a in range(0, n, 4096):                                         # inside: within the ball of some page among its 16 nearest
    dd_, ii_ = knn(M[a:a + 4096], Z, 16)
    inside[a:a + 4096] = (dd_ <= rho[ii_]).any(1)
log("interpolated candidates inside the region", int(inside.sum()), "of", n)
C = torch.cat([Z, M[inside]])
del M
N = len(C)
sq = (C ** 2).sum(1)
Ch = C.half()
dmin = torch.full((N,), float("inf"), device=DEV)
pick, acc, Bk, nxt = [], [int(torch.randint(0, n, (1,), generator=torch.Generator().manual_seed(0)))], 100, 10_000
while True:
    pick += acc
    P = torch.tensor(acc, device=DEV)
    torch.minimum(dmin, (sq[:, None] + sq[P][None] - 2 * (Ch @ Ch[P].T).float()).clamp_(min=0).min(1).values, out=dmin)
    dmin[P] = -1
    if len(pick) >= NPICK:
        break
    v, c = torch.topk(dmin, 4 * Bk)
    acc = []
    for dv, j in zip(v.tolist(), c.tolist()):
        if dv < 0:
            break
        if acc and float((sq[acc] + sq[j] - 2 * (C[acc] @ C[j])).min()) < dv:
            break                                                   # j moved: the rest of the batch is no longer surely the farthest
        acc.append(j)
        if len(acc) == Bk or len(pick) + len(acc) == NPICK:
            break
    if len(pick) >= nxt:
        nxt += 10_000
        log("k-center", len(pick), "radius", float(v[0].clamp(min=0).sqrt()))
cover = float(dmin.clamp(min=0).max().sqrt())
log("k-center", len(pick), "covering radius", cover)
pick = torch.tensor(pick, device=DEV)
X = C[pick].clone()
del C, Ch, sq, dmin
torch.cuda.empty_cache() if DEV == "cuda" else None
dd, ii = knn(X, Z, 16)
dd = dd.sqrt()
np.savez(f"{OUTDIR}/points.npz", pool_i=z["pool_i"] if SUB is None else np.zeros(0), pick=pick.cpu().numpy(), from_page=(pick < n).cpu().numpy(),
         X=X.half().cpu().numpy(), dd=dd.cpu().numpy().astype(np.float16), ii=ii.cpu().numpy().astype(np.int32))
json.dump({"cover": cover, "rad": rad}, open(f"{OUTDIR}/points.json", "w"))
log("DONE")
'''


def code(root, outdir, dev, sub, npick):
    return f"ROOT, OUTDIR, DEV, SUB, NPICK = {root!r}, {outdir!r}, {dev!r}, {sub}, {npick}\n" + KERNEL


def run_dir(a):
    """Outputs of one selection size: OUT/n<N>k (kernel output in out/, the lists next to it)."""
    return OUT / f"n{a.n // 1000}k"


def cmd_push(a):
    d = OUT / "kernel"
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.py").write_text(code("/kaggle/input", "/kaggle/working", "cuda", None, a.n))
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": KID, "title": KID.split("/")[1], "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "dataset_sources": [AUX], "kernel_sources": [],
        "competition_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4"}))
    print(KF.kaggle("kernels", "push", "-p", str(d)))


def cmd_test(_):
    import torch
    torch.set_num_threads(S.M.WORKERS)
    exec(compile(code(str(OUT / "aux"), str(D.TESTS / "targets" / "test"), "cpu", 20_000, 2000), "kernel", "exec"), {"__name__": "kernel"})


def assign(Pt, scan, hwn, text, factor, cap, hw_min, top_up=True):
    """Pages of the points (points.npz: X, dd, ii = the 16 nearest pages of every point), greedy by key over the pairs. A digital-born page is taken within factor x r;
    a scan only with more than hw_min - 1 handwriting boxes, within r, and its key is factor x its distance (a digital page up to factor x farther wins). At most `cap`
    scan points stay: hw, then filled pages first, then the nearer. -> (page, dist) per point, -1 / inf where no page."""
    r = float(json.load(open(Pt["dir"] / "points.json"))["cover"]) / 2
    dd, ii = Pt["dd"].astype(np.float32), Pt["ii"]
    sc = scan[ii]
    ok = np.where(sc, (hwn[ii] >= hw_min) & (dd <= r), dd <= factor * r)
    key = np.where(sc, factor * dd, dd)
    key[~ok] = np.inf
    o = np.argsort(key.flatten(), kind="stable")
    page = np.full(len(dd), -1, np.int64)
    dist = np.full(len(dd), np.inf, np.float32)
    used = np.zeros(len(scan), bool)
    for f in o[:int(np.isfinite(key).sum())]:
        p, c = divmod(f, dd.shape[1])
        j = ii[p, c]
        if page[p] < 0 and not used[j]:
            page[p], dist[p], used[j] = j, dd[p, c], True
    sp = np.flatnonzero((page >= 0) & scan[np.where(page >= 0, page, 0)])
    if len(sp) > cap:
        k = DM.hw_class(hwn[page[sp]], text[page[sp]])                  # 2 hw, 1 filled, 0 none
        drop = sp[np.lexsort((dist[sp], -k))[cap:]]
        used[page[drop]] = False
        page[drop], dist[drop] = -1, np.inf
    left = cap - int(((page >= 0) & scan[np.where(page >= 0, page, 0)]).sum())
    if top_up and left > 0:                                              # fewer scans with hw than the cap: the nearest other scans within r fill the rest
        ok2 = scan[ii] & (hwn[ii] < hw_min) & (dd <= r)
        key2 = np.where(ok2, dd, np.inf)
        for f in np.argsort(key2.flatten(), kind="stable")[:int(ok2.sum())]:
            p, c = divmod(f, dd.shape[1])
            j = ii[p, c]
            if page[p] < 0 and not used[j]:
                page[p], dist[p], used[j] = j, dd[p, c], True
                left -= 1
                if not left:
                    break
    return page, dist, r


def load_points(a):
    D_ = run_dir(a)
    P = dict(np.load(D_ / "out" / "points.npz"))
    P["dir"] = D_ / "out"
    z = np.load(OUT / "aux" / "Z.npz")
    return P, z


def cmd_assign(a):
    """The radius test and the final assignment: --factors lists the digital radius factors to compare; --apply writes targets.npz / targets.json for `build`."""
    import torch
    P, z = load_points(a)
    scan, hwn, text = z["scan"], z["hw"], z["text"]
    n = len(P["X"])
    print(f"{n} points, {len(scan)} pages in scope: {int(scan.sum())} scans, {int((scan & (hwn >= HW_KEEP + 1)).sum())} scans with hw > {HW_KEEP}")
    res = {}
    for f in a.factors:
        page, dist, r = assign(P, scan, hwn, text, f, a.cap, HW_KEEP + 1, not a.strict)
        has = page >= 0
        s = has & scan[np.where(has, page, 0)]
        k = DM.hw_class(hwn[page[s]], text[page[s]])
        res[f] = (page, dist, r)
        print(f"factor {f}: r {r:.4f}  points with a page {int(has.sum())} (digital {int((has & ~s).sum())}, scans {int(s.sum())}: hw {int((k == 2).sum())}, filled {int((k == 1).sum())})  targets {int((~has).sum())}")
    if a.apply:
        page, dist, r = res[a.factors[0]]
        tgt = np.flatnonzero(page < 0)
        Xt = torch.from_numpy(P["X"][tgt].astype(np.float32))
        Zt = torch.from_numpy(z["Z"].astype(np.float32))
        nd, nb = [], []
        for b in range(0, len(Xt), 1024):
            d = torch.cdist(Xt[b:b + 1024], Zt).topk(10, largest=False)
            nd.append(d.values.numpy()), nb.append(d.indices.numpy())
        nd, nb = np.concatenate(nd), np.concatenate(nb)
        out = {"pick": P["pick"], "from_page": P["from_page"], "page": page, "dist": dist, "target": tgt, "Xt": P["X"][tgt], "nb": nb, "nd": nd}
        for k in range(3):
            out[f"nb{k}"], out[f"nd{k}"] = nb, nd
        np.savez(P["dir"] / "targets.npz", pool_i=P["pool_i"], **out)
        json.dump({"cover": r * 2, "r": a.factors[0] * r, "with_page": int((page >= 0).sum()), "targets": len(tgt)}, open(P["dir"] / "targets.json", "w"))
        print("targets.npz written:", len(tgt), "targets")


def page_fill(sp):
    """Ink-tile area over the page area: window fill x the part of the window on the page (wx, wy, ww, wh: shares of the page)."""
    ox = np.clip(sp.wx + sp.ww, 0, 1) - np.clip(sp.wx, 0, 1)
    oy = np.clip(sp.wy + sp.wh, 0, 1) - np.clip(sp.wy, 0, 1)
    return sp.fill * ox * oy


def cmd_prep(_):
    """Scope mask (page fill > PAGE_FILL, pic < PIC, review outliers out, pages still embedded), the ds_metric heads on those pool vectors -> OUT/aux/Z.npz, dataset AUX."""
    import glob
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import ds_grid as G
    P = pd.read_parquet(D.OUT / "pool.parquet", columns=["doc", "page", "scan", "hw", "text"])
    sp = pd.read_parquet(S.M.LAYOUTS / "archive" / "info" / "scan_pct.parquet", columns=["doc", "page", "fill", "pic", "wx", "wy", "ww", "wh"])
    R = P.merge(sp, on=["doc", "page"], how="left")
    assert len(R) == len(P)
    m = ((page_fill(R) > PAGE_FILL) & (R.pic < PIC)).to_numpy().copy()
    if Path(G.LABELLED).exists():                                   # review outliers out (else the scope of the last prep stays)
        L = pd.read_parquet(G.LABELLED)
        m[L.pool_i[~L.kept].to_numpy()] = False
    else:
        last = np.load(OUT / "aux" / "Z.npz")
        n_last = int(last["n_pool"]) if "n_pool" in last.files else 1_067_635      # pool pages the last scope was made from: later ones (ds100k append) are judged by the rules alone
        m &= np.isin(np.arange(len(m)), last["pool_i"]) | (np.arange(len(m)) >= n_last)
    ids = np.load(D.EMB / "ids.npy")
    m &= np.isin(P.doc.to_numpy().astype(np.int64) * S.M.KEY + P.page.to_numpy(), np.unique(ids[:, 0].astype(np.int64) * S.M.KEY + ids[:, 1]))
    real = np.flatnonzero(m)
    print(f"pages in scope {len(real)}")
    import ds_head as H
    net = nn.Sequential(nn.Linear(1536, 512), nn.GELU(), nn.Linear(512, H.DIM))
    net.load_state_dict(torch.load(H.OUT / "head.pt", map_location="cpu"))
    get = H.pool_rows()
    torch.set_num_threads(S.M.WORKERS)
    Z = np.zeros((len(real), H.DIM), np.float16)
    with torch.no_grad():
        for a in range(0, len(real), 50_000):
            Z[a:a + 50_000] = F.normalize(net(torch.from_numpy(get(real[a:a + 50_000]))), dim=1).numpy()
    d = OUT / "aux"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "Z.npz", pool_i=real, Z=Z, n_pool=len(P), scan=R.scan.to_numpy()[real].astype(bool), hw=R.hw.fillna(0).to_numpy()[real].astype(np.int16),
             text=R.text.fillna(0).to_numpy()[real].astype(np.int16))
    (d / "dataset-metadata.json").write_text(json.dumps({"title": AUX.split("/")[1], "id": AUX, "licenses": [{"name": "CC-BY-NC-SA-4.0"}]}))
    print(KF.kaggle("datasets", "version", "-p", str(d), "-m", "head16", timeout=3600)[-200:])


def cmd_fetch(a):
    st = KF.kaggle("kernels", "status", KID)
    print(st)
    if "COMPLETE" not in st.upper():
        return
    out = run_dir(a) / "out"
    out.mkdir(parents=True, exist_ok=True)
    for f in [*out.glob("targets.*"), *out.glob("points.*")]:                                     # the cli skips a file that exists: the old run would stay
        f.unlink()
    print(KF.kaggle("kernels", "output", KID, "-p", str(out))[-200:])


def pages():
    """Rows of Z (in-scope pages) with path, page, language, scan, hw boxes, text boxes, plan type and the shard fill / pic / window."""
    P = pd.read_parquet(D.OUT / "pool.parquet", columns=["doc", "page", "path", "h", "lang", "scan", "text", "hw"])
    R = P.iloc[np.load(OUT / "aux" / "Z.npz")["pool_i"]].reset_index(drop=True)
    sp = pd.read_parquet(S.M.LAYOUTS / "archive" / "info" / "scan_pct.parquet", columns=["doc", "page", "fill", "pic", "wx", "wy", "ww"])
    n = len(R)
    R = R.merge(sp, on=["doc", "page"], how="left")
    assert len(R) == n
    with sqlite3.connect(f"file:{S.M.LAYOUTS / 'manifest.sqlite'}?mode=ro", uri=True) as db:
        R["type"] = R.h.map(dict(db.execute("select doc, type from plan"))).fillna("_untyped")
    return R


def hw_class(hw, text):
    """hw / filled / none of the hw_det box counts (ds_metric.hw_class)."""
    return np.array(["none", "filled", "hw"])[DM.hw_class(hw, text)]


def vote(v):
    """Most frequent value of a row and its share."""
    s = pd.Series(v).value_counts()
    return s.index[0], s.iloc[0] / len(v)


def cmd_build(a):
    D_ = run_dir(a)
    T = np.load(D_ / "out" / "targets.npz")
    j = json.load(open(D_ / "out" / "targets.json"))
    R = pages()
    has = T["dist"] <= j["r"]
    E = R.iloc[T["page"][has]].reset_index(drop=True)[["path", "page", "type", "lang", "scan", "hw", "text", "fill", "pic"]]
    E.insert(0, "point", np.flatnonzero(has))
    E["dist"] = T["dist"][has]
    typ, lang, hw = R.type.to_numpy(), R.lang.to_numpy(), hw_class(R.hw, R.text)
    rows = []
    for k, p in enumerate(T["target"]):
        tv = typ[T["nb1"][k]]
        tv = tv[tv != "_untyped"] if (tv != "_untyped").any() else tv
        t, ts = vote(tv)
        lg, ls = vote(lang[T["nb"][k]])
        L = R.iloc[T["nb2"][k]]
        near = R.iloc[T["nb"][k][:3]]
        hc, hs = vote(hw[T["nb0"][k]])
        rows.append({"point": p, "type": t, "type_share": ts, "hw": hc, "hw_share": hs, "fill": L.fill.mean(), "pic": L.pic.mean(),
                     "wx": L.wx.mean(), "wy": L.wy.mean(), "ww": L.ww.mean(), "text_boxes": L.text.mean(), "lang": lg, "lang_share": ls,
                     "near_dist": T["nd"][k][0], **{f"near{i + 1}": f"{a} p{b}" for i, (a, b) in enumerate(zip(near.path, near.page))}})
    G = pd.DataFrame(rows)
    E.to_csv(D_ / "existing.csv", index=False)
    G.to_csv(D_ / "targets.csv", index=False, float_format="%.3f")
    print(f"points {len(T['pick'])}, covering radius {j['cover']:.4f}, r {j['r']:.4f}; existing {len(E)}, targets {len(G)}")
    pool = R.assign(hwc=hw).groupby(["type", "hwc"]).size()
    S_ = pd.DataFrame({"targets": G.rename(columns={"hw": "hwc"}).groupby(["type", "hwc"]).size(),
                       "existing": E.assign(hwc=hw_class(E.hw, E.text)).groupby(["type", "hwc"]).size(), "pool": pool}).fillna(0).astype(int)
    print(S_.sort_values("targets", ascending=False).to_string())
    print("\ntargets: language", G.lang.value_counts().head(15).to_dict())
    print("targets: mean fill / pic / text boxes", G[["fill", "pic", "text_boxes"]].mean().round(3).to_dict(), "| existing", E[["fill", "pic", "text"]].mean().round(3).to_dict())
    print("files:", D_ / "existing.csv", D_ / "targets.csv")


def cmd_scans(a):
    """The scan pages of the existing points not checked yet (cleaned_trace: page clean / warped, document warped or moved by a cleaning
    step; folders cleaned whole) as full-resolution jpgs for the warp check (diverse_ds export format: archive/<N>k_warp_check,
    <n>_<hash>_p<page>.jpg + index.csv; tools/cleaned.py move takes the deleted ones out). Resumes: rendered files are kept."""
    import diverse_ds as DD
    from concurrent.futures import ProcessPoolExecutor
    from dspages.fillpic.paths import doc_hash
    E = pd.read_csv(run_dir(a) / "existing.csv")
    E = E[E.scan].reset_index(drop=True)
    E["h"] = E.path.map(doc_hash)
    T = pd.read_csv(S.M.LAYOUTS / "archive" / "info" / "cleaned_trace.tsv", sep="\t")                # tools/cleaned.py trace
    done = set(zip(T.doc[T.verdict.isin(["clean", "warped"])], T.page[T.verdict.isin(["clean", "warped"])].fillna(-1).astype(int)))
    checked = (np.array([k in done for k in zip(E.h, E.page)]) | E.h.isin(T.doc[T.verdict.isin(["warped_doc", "moved"])])
               | E.path.str.split("/").str[0].isin(DD.CLEAN_FOLDERS))
    print(f"{len(E)} scan pages, {int(checked.sum())} already checked by the warp cleaning (cleaned_trace / clean folders)")
    E = E[~checked].reset_index(drop=True)
    out = S.M.LAYOUTS / "archive" / f"{a.n // 1000}k_warp_check"
    out.mkdir(parents=True, exist_ok=True)
    E["file"] = [f"{i:06d}_{h}_p{p}.jpg" for i, (h, p) in enumerate(zip(E.h, E.page))]
    for x in set(f.name for f in out.glob("*.jpg")) - set(E.file):                       # renders of pages that are out now
        (out / x).unlink()
    have = set(x.name for x in out.glob("*.jpg"))
    jobs = [(r.file, r.path, r.page, out / r.file) for r in E.itertuples() if r.file not in have]
    with ProcessPoolExecutor(DD.WORKERS, initializer=DD._init, initargs=(DD.hash_index(),)) as ex:
        ok = dict(ex.map(DD.render, jobs, chunksize=16))
    E["rendered"] = E.file.map(lambda f: f in have or bool(ok.get(f)))
    E.to_csv(out / "index.csv", index=False)
    print(len(E), "scan pages,", int(E.rendered.sum()), "rendered ->", out)



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for c in ("prep", "push", "test", "fetch", "assign", "build", "scans"):
        p = sp.add_parser(c)
        p.add_argument("--n", type=int, default=100_000, help="points")
        if c == "assign":
            p.add_argument("--factors", type=float, nargs="+", default=[FACTOR])
            p.add_argument("--cap", type=int, default=SCAN_CAP)
            p.add_argument("--apply", action="store_true")
            p.add_argument("--strict", action="store_true", help="scans only with hw > 1 (no top-up to the cap)")
    a = ap.parse_args()
    {"prep": cmd_prep, "scans": cmd_scans, "assign": cmd_assign, "push": cmd_push, "test": cmd_test, "fetch": cmd_fetch, "build": cmd_build}[a.cmd](a)
