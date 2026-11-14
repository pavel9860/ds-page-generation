"""Page-kind label set, ~6K pages in 17 classes: the class folders of Labels/pagetype_labelled (images, pages_photo out).
Folder labels are soft: the hw / text box counts (yolo26s, ds_metric.hw_class) move pages between forms_blank / forms_filled / handwritten;
near duplicates inside a class keep one page; an out-of-fold model (logistic regression on the page embedding + log box
counts, pages of one document in one fold) marks the border pages (own label not the likeliest, or a small margin), which go to the review.
Iterate: fit -> review -> fit ... ; the reviewed labels win over every rule. Data: Labels/labelset (OUT).

  pack     items (every labelled file) rendered 768 x 1024 like the shards -> OUT/items.parquet, OUT/npz/pages_0000.npz, dataset pavel9860/labelset-pages-1
  push     Kaggle 2 x T4: page embedding (pagetype/embed_kaggle.py) + yolo26s boxes (kaggle_topup kernel)      fetch: -> OUT/out
  fit      labels (folder, box rules, review), dedup, out-of-fold model, ~PER pages per class -> OUT/set.parquet, OUT/border.parquet
  review   border pages in the browser (http://127.0.0.1:PORT): keys 1-9 a-i = class, 0 = drop, Left/Right; -> OUT/review.tsv

python tools/labelset.py pack | push | fetch | fit | review"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import ds100k as D
import ds_metric as DM
import form_pages as F
import kaggle_topup as KT

SRC = Path("/run/media/p/D/ML_DS/UVTM/Labels/pagetype_labelled")
OUT = SRC.parent / "labelset"
DATASET, KID = "pavel9860/labelset-pages", "pavel9860/labelset-emb-hw"
DROP = {"cache", "images", "pages_photo"}
CLASSES = sorted(c.name for c in SRC.iterdir() if c.is_dir() and c.name not in DROP)
PER, DUP, MARGIN = 350, 0.95, 0.2


def img(a):
    """(k, jpeg) of an image file, stretched to the shard size like form_pages."""
    from PIL import Image
    k, p = a
    try:
        return k, F.encode(np.asarray(Image.open(p).convert("RGB")))
    except Exception:
        return k, None


def cmd_pack(_):
    from concurrent.futures import ProcessPoolExecutor
    rows = []
    for c in sorted(p for p in SRC.iterdir() if p.is_dir() and p.name not in DROP):
        for f in sorted(c.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".pdf"):
                pg = re.search(r"_p(\d+)\.[a-z]+$", f.name, re.I)
                rows.append({"cls": c.name, "path": str(f), "page": int(pg.group(1)) if pg and f.suffix.lower() == ".pdf" else 0, "origin": "folder"})
    I = pd.DataFrame(rows).reset_index(names="k")
    pdf = I.path.str.lower().str.endswith(".pdf")
    jobs = [(p, [(k, g)]) for k, p, g in zip(I.k[pdf], I.path[pdf], I.page[pdf])]
    jpg = {k: j for k, _, j in F.run_jobs(jobs)}
    with ProcessPoolExecutor(F.WORKERS) as ex:
        jpg.update({k: j for k, j in ex.map(img, zip(I.k[~pdf], I.path[~pdf]), chunksize=32) if j})
    I["rendered"] = I.k.isin(jpg)
    (OUT / "npz").mkdir(parents=True, exist_ok=True)
    ks = sorted(jpg)
    off = np.cumsum([0] + [len(jpg[k]) for k in ks])
    np.savez(OUT / "npz" / "pages_0000.npz", ids=np.array([[k, 0] for k in ks], np.int32), offsets=off.astype(np.int64),
             jpeg=np.frombuffer(b"".join(jpg[k] for k in ks), np.uint8))
    I.to_parquet(OUT / "items.parquet")
    print(f"items {len(I)} ({I.groupby('origin').size().to_dict()}), rendered {len(ks)}, {off[-1] / 1e9:.2f} GB")
    from dspages.kaggle_api import api
    k = api()
    print(F.upload(k, 1, [OUT / "npz" / "pages_0000.npz"], DATASET, OUT))


def cmd_push(_):
    d = OUT / "kernel"
    d.mkdir(parents=True, exist_ok=True)
    kern = KT.KERNEL.replace('ds = re.search(r"pages-768x1024-(\\d+)", s).group(1)', 'ds = "0"')
    assert kern != KT.KERNEL
    (d / "run.py").write_text(f"EMBED_SRC = {KT.EMBED.read_text()!r}\n" + kern)
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": KID, "title": KID.split("/")[1], "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "keywords": ["gpu"], "dataset_sources": [f"{DATASET}-2", KT.WEIGHTS_DS],
        "kernel_sources": [], "competition_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4"}))
    print(KT.kaggle("kernels", "push", "-p", str(d)))


def cmd_fetch(_):
    st = KT.kaggle("kernels", "status", KID)
    print(st)
    if "COMPLETE" in st.upper():
        (OUT / "out").mkdir(parents=True, exist_ok=True)
        print(KT.kaggle("kernels", "output", KID, "-p", str(OUT / "out"), timeout=3600)[-300:])


def _det(part):
    """yolo26s hw / text boxes of the pages of npz2 rows `part` (one process): (ids, per-page class arrays)."""
    import io
    import torch
    from PIL import Image
    from ultralytics import YOLO
    torch.set_num_threads(1)
    z = np.load(OUT / "npz2" / "pages_0000.npz")
    m, off, jp = YOLO(str(KT.WEIGHTS)), z["offsets"], z["jpeg"]
    out = []
    for i in part:
        img = np.asarray(Image.open(io.BytesIO(jp[off[i]:off[i + 1]].tobytes())).convert("RGB"))[:, :, ::-1]
        out.append((int(z["ids"][i, 0]), m.predict(img, imgsz=1024, conf=0.25, verbose=False)[0].boxes.cls.int().numpy()))
    return out


def cmd_detect(_):
    """Box counts of the npz2 pages on the CPU (F.WORKERS processes) -> OUT/out/hw/pages_0000.npz in the kaggle_topup format."""
    from concurrent.futures import ProcessPoolExecutor
    n = len(np.load(OUT / "npz2" / "pages_0000.npz")["ids"])
    with ProcessPoolExecutor(F.WORKERS) as ex:
        res = [r for part in ex.map(_det, np.array_split(np.arange(n), F.WORKERS * 4)) for r in part]
    (OUT / "out" / "hw").mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "out" / "hw" / "pages_0000.npz", ids=np.array([[k, 0] for k, _ in res], np.int32),
             offsets=np.cumsum([0] + [len(c) for _, c in res]).astype(np.int64), cls=np.concatenate([c for _, c in res]).astype(np.uint8))
    print(n, "pages,", sum(len(c) for _, c in res), "boxes")


def data():
    """Labelled items with a page vector (n, 1536, unit) and box counts: the corpus vector and hw_det counts where the item is a corpus page
    (items_pool.parquet, pool_i), the Kaggle run's (OUT/out) for the others once it is there; items of deleted files are out."""
    import glob
    import kaggle_fps as KF
    I = pd.read_parquet(OUT / "items_pool.parquet")
    I = I[I.path.map(lambda f: Path(f).exists())].reset_index(drop=True)
    P = pd.read_parquet(D.OUT / "pool.parquet", columns=["hw", "text"])
    parts = [np.load(x, mmap_mode="r") for x in sorted(glob.glob(str(KF.WORK / "data" / "pool_*.npy")))]
    starts = np.cumsum([0] + [len(x) for x in parts])
    E = np.full((len(I), 1536), np.nan, np.float32)
    hw, tx = np.full(len(I), -1), np.full(len(I), -1)
    a = np.flatnonzero(I.pool_i.notna())
    r = I.pool_i.to_numpy()[a].astype(np.int64)
    q = np.searchsorted(starts, r, side="right") - 1
    E[a] = np.stack([parts[x][y - starts[x]] for x, y in zip(q, r)])
    hw[a], tx[a] = P.hw.to_numpy()[r], P.text.to_numpy()[r]
    if (OUT / "out").exists():
        ids = np.load(next((OUT / "out").rglob("ids.npy")))[:, 0]
        V = np.load(next((OUT / "out").rglob("embeddings.npy"))).astype(np.float32)
        z = np.load(next((OUT / "out").rglob("*pages_0000.npz")))
        own = np.repeat(z["ids"][:, 0], np.diff(z["offsets"]))
        n_hw, n_tx = np.bincount(own[z["cls"] == 0], minlength=I.k.max() + 1), np.bincount(own[z["cls"] == 1], minlength=I.k.max() + 1)
        at = {int(k): i for i, k in enumerate(ids)}
        b = np.array([i for i, k in enumerate(I.k) if np.isnan(E[i, 0]) and k in at], int)
        E[b] = V[[at[k] for k in I.k.to_numpy()[b]]]
        hw[b], tx[b] = n_hw[I.k.to_numpy()[b]], n_tx[I.k.to_numpy()[b]]
    ok = ~np.isnan(E[:, 0])
    I, E = I[ok].reset_index(drop=True).assign(hw=hw[ok], text=tx[ok]), E[ok]
    I["doc"] = I.path.map(lambda p: (m.group(0) if (m := re.search(r"[0-9a-f]{12}", Path(p).name)) else Path(p).stem))
    return I, E / np.linalg.norm(E, axis=1, keepdims=True)


def rules(I):
    """Label after the box counts (hw_class): forms_blank / forms_filled by hw, a page of handwriting is handwritten."""
    c = DM.hw_class(I.hw, I.text)
    y = I.cls.to_numpy().copy()
    y[(y == "forms_filled") & (c == 0)] = "forms_blank"
    y[(y == "forms_blank") & (c >= 1)] = "forms_filled"
    y[(c == 2) & ~np.isin(y, ["forms_filled", "music"])] = "handwritten"
    return y


def cmd_fit(_):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    I, E = data()
    y = rules(I)
    rv = OUT / "review.tsv"
    if rv.exists():
        R = pd.read_csv(rv, sep="\t", names=["k", "label"]).drop_duplicates("k", keep="last")
        y = pd.Series(y, index=I.k).where(~I.k.isin(R.k).to_numpy(), I.k.map(dict(zip(R.k, R.label))).to_numpy()).to_numpy()
    reviewed = I.k.isin(pd.read_csv(rv, sep="\t", names=["k", "label"]).k).to_numpy() if rv.exists() else np.zeros(len(I), bool)
    nohw = DM.hw_class(I.hw, I.text) == 0                             # no handwriting: never handwritten or a filled form, review or not
    y[nohw & (y == "forms_filled")] = "forms_blank"
    y[nohw & (y == "handwritten")] = "drop"
    keep = y != "drop"
    X = np.hstack([E, np.log1p(I[["hw", "text"]].to_numpy(np.float32))])
    # near duplicates in one class: the first page of every group stays
    dup = np.zeros(len(I), bool)
    for c in np.unique(y[keep]):
        ix = np.flatnonzero(keep & (y == c))
        S = E[ix] @ E[ix].T
        np.fill_diagonal(S, 0)
        for a in range(len(ix)):
            if not dup[ix[a]] and (S[a, a + 1:] > DUP).any():
                dup[ix[a + 1:][S[a, a + 1:] > DUP]] = True
    use = keep & ~dup
    iu = np.flatnonzero(use)
    P = np.zeros((len(I), len(CLASSES)), np.float32)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X[iu], y[iu], I.doc.to_numpy()[iu]):
        m = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced").fit(X[iu[tr]], y[iu[tr]])
        P[np.ix_(iu[te], [CLASSES.index(c) for c in m.classes_])] = m.predict_proba(X[iu[te]])
    own = P[np.arange(len(I)), [CLASSES.index(c) if c in CLASSES else 0 for c in y]]
    top2 = np.sort(P, 1)[:, -2:]
    pred = np.array(CLASSES)[P.argmax(1)]
    border = use & ~reviewed & ((pred != y) | (top2[:, 1] - top2[:, 0] < MARGIN))
    I = I.assign(label=y, pred=pred, p_own=own, margin=top2[:, 1] - top2[:, 0], dup=dup, border=border, reviewed=reviewed)
    # the set: per class the reviewed pages, then the surest ones, at most PER
    S = I[use & ~border].sort_values(["reviewed", "p_own"], ascending=False).groupby("label").head(PER)
    S.to_parquet(OUT / "set.parquet")
    I[border].sort_values("margin").to_parquet(OUT / "border.parquet")
    acc = (pred[iu] == y[iu]).mean()
    bal = np.mean([(pred[iu][y[iu] == c] == c).mean() for c in np.unique(y[iu])])
    print(f"items {len(I)}: dropped {int((~keep).sum())}, near duplicates {int(dup.sum())}, reviewed {int(reviewed.sum())}; out-of-fold acc {acc:.3f}, "
          f"balanced {bal:.3f}; border {int(border.sum())}; set {len(S)}")
    T = pd.DataFrame({"folder": I.cls.value_counts(), "label": I.label[use].value_counts(), "border": I.label[border].value_counts(),
                      "set": S.label.value_counts()}).fillna(0).astype(int)
    print(T.to_string())


PAGE = """<!doctype html><meta charset=utf-8><title>labelset review</title><style>body{margin:0;background:#111;color:#ddd;font:15px sans-serif;display:flex}
img{height:100vh}#s{padding:12px;line-height:1.6}b{color:#fd6}.c{display:block}</style><img id=i><div id=s></div><script>
const C=%s;let L=%s,n=0;const keys='123456789abcdefghi';
function show(){const r=L[n];document.getElementById('i').src='/img/'+r.k;document.getElementById('s').innerHTML=`${n+1}/${L.length}<br>label <b>${r.label}</b>
<br>model <b>${r.pred}</b> margin ${r.margin.toFixed(2)}<br>hw ${r.hw} text ${r.text}<br><br>`+C.map((c,i)=>`<span class=c>${keys[i]} ${c}</span>`).join('')+'<span class=c>0 drop</span>'}
document.addEventListener('keydown',e=>{let i=keys.indexOf(e.key),v=e.key=='0'?'drop':i>=0?C[i]:null;
if(v){fetch('/set?k='+L[n].k+'&v='+v);n=Math.min(n+1,L.length-1);show()}else if(e.key=='ArrowRight'){n=Math.min(n+1,L.length-1);show()}
else if(e.key=='ArrowLeft'){n=Math.max(n-1,0);show()}});show()</script>"""


def cmd_review(a):
    import http.server
    import urllib.parse
    B = pd.read_parquet(OUT / "border.parquet")
    src = dict(zip(*pd.read_parquet(OUT / "items.parquet")[["k", "path"]].to_numpy().T))
    pages = dict(zip(*pd.read_parquet(OUT / "items.parquet")[["k", "page"]].to_numpy().T))
    done = set(pd.read_csv(OUT / "review.tsv", sep="\t", names=["k", "label"]).k) if (OUT / "review.tsv").exists() else set()
    B = B[~B.k.isin(done)]
    rows = json.dumps(B[["k", "label", "pred", "margin", "hw", "text"]].to_dict("records"))
    page = (PAGE % (json.dumps(CLASSES), rows)).encode()

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            if u.path.startswith("/img/"):
                k = int(u.path[5:])                                       # the original file: an image as it is, a pdf page rendered
                f = src[k]
                body = F.render_job((f, [(k, int(pages[k]))]))[0][2] if f.lower().endswith(".pdf") else Path(f).read_bytes()
                ct = "image/jpeg"
            elif u.path == "/set":
                q = urllib.parse.parse_qs(u.query)
                with open(OUT / "review.tsv", "a") as f:
                    f.write(f"{int(q['k'][0])}\t{q['v'][0]}\n")
                body, ct = b"ok", "text/plain"
            else:
                body, ct = page, "text/html"
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.end_headers()
            self.wfile.write(body)
    print(f"{len(B)} border pages: http://127.0.0.1:{a.port}", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["pack", "push", "fetch", "detect", "fit", "review"])
    ap.add_argument("--port", type=int, default=8767)
    a = ap.parse_args()
    {"pack": cmd_pack, "push": cmd_push, "fetch": cmd_fetch, "detect": cmd_detect, "fit": cmd_fit, "review": cmd_review}[a.cmd](a)
