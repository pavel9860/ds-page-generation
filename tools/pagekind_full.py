"""All page-type classes (the 18 folders of pagetype_labelled) from the image embedding and, where the page has a usable PDF layer, the
formatting features; plus folder samples of the corpus (Layouts/<folder> = class, page 0, the corpus page embedding of emb_out) to check
those folders and to add data to the small classes.

  data     labelled pages (embedding; formatting features when matched to a PDF by tools/pagekind_sources.py) + up to --per-folder
           documents of every class folder of FOLDERS that have a page-0 corpus embedding (scientific_paper: page 0 or 1 with formulas,
           plots or tables, see science()) -> CACHE/pagekind_full.npz (features cached)
  eval     grouped CV (pages of one document in one fold): embedding logreg, formatting xgb (NaN without a PDF layer), their product;
           standard metrics per class; per source (labelled / folder) the share of pages whose label the model gives a low probability
           -> CACHE/pagekind_full_low.jsonl and review sheets CACHE/full_review/<class>_NN.jpg (lowest probability of the own label first)

python tools/pagekind_full.py data [--per-folder 300]
python tools/pagekind_full.py eval [--low 0.2]
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pymupdf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/run/media/p/D/BUISNESS/corpus_collecting")
import pagekind_eval as V                                     # noqa: E402
from dspages.hand import corpus                               # noqa: E402
from dspages.hand import pagekind as PK                       # noqa: E402
from dspages.hand.config import PATHS                         # noqa: E402

ROOT = V.UV / "pagetype_labelled"
DATA = V.CACHE / "pagekind_full.npz"
FOLDERS = {"certificates": "certificates", "flyers_clean": "flyers", "maps": "maps", "sheet_music": "music", "tables_clean": "tables",
           "blueprints": "blueprints", "receipts": "recieps", "scientific_paper": "scientific"}
MATH = set("=+−±×÷∑∫∂√∞≈≠≤≥∈∉∀∃∇∆πθλμσαβγδεφωΩΣΠ^_")
LAYER = ("text", "encoding lost", "vector / painted text")     # plan.cls with a PDF text / vector layer: formatting features exist
NF = len(PK.NAMES)
SCAN_ONLY = ("cache", "pages_photo", "handwritten", "images", "recieps")   # classes without digital-born pages: not page kinds here


def classes():
    return sorted(c.name for c in ROOT.iterdir() if c.is_dir() and c.name not in SCAN_ONLY)


def _feat(job):
    pdf, no = job
    try:
        return PK.features(pymupdf.open(pdf)[no])
    except Exception:
        return None


def science(pdf, no):
    """A page of a scientific paper that shows its science, not only running text: formulas (math symbols and Greek letters, at least
    30 and 2 % of the characters), a figure or plot (pictures or vector paths covering at least 12 % of the page) or a table grid."""
    try:
        page = pymupdf.open(pdf)[no]
        t = page.get_text()
        letters = max(sum(c.isalnum() for c in t), 1)
        math = sum(c in MATH or "\u0391" <= c <= "\u03c9" for c in t)
        area = page.rect.width * page.rect.height or 1.0
        pics = sum(pymupdf.Rect(i["bbox"]).get_area() for i in page.get_image_info()) / area
        paths = page.get_drawings()
        box = pymupdf.Rect()
        for d in paths:
            if d["rect"].get_area() < 0.9 * area:
                box |= d["rect"]
        figure = len(paths) >= 40 and box.get_area() / area >= 0.12
        f = dict(zip(PK.NAMES, PK.features(page)))
        big = max((pymupdf.Rect(i["bbox"]).get_area() for i in page.get_image_info()), default=0) / area
        if big >= 0.6:                                  # a scanned page: one picture over the page
            return False
        return (math >= 30 and math >= 0.03 * letters) or pics >= 0.12 or figure or f["grid"] >= 0.5
    except Exception:
        return False


def data(per_folder, seed=0):
    rng = random.Random(seed)
    cls = {d: c for d, c in corpus.connect().execute("select doc,cls from plan")}
    paths = corpus.pdf_paths(corpus.connect())
    pe = V.page_embeddings()
    src = {r["image"]: r for r in map(json.loads, open(V.SRC)) if r["ok"]}
    rows = []                                                   # (label, doc, source, id, embedding, pdf, page)
    import pagekind_inventory as I
    balance = json.load(open(V.CACHE / "pagekind_balance.json"))      # tools/pagekind_lang.py plan: language-balanced labelled pages
    keep = {i for ids in balance.values() for i in ids}
    for it in I.inventory():
        if it.emb is not None and it.id in keep:
            rows.append((it.cls, it.doc, "labelled", it.id, it.emb, it.pdf, it.page))
    have = {r[1] for r in rows}
    ids = np.load(V.UV / "emb_out" / "ids.npy")
    emb = np.load(V.UV / "emb_out" / "embeddings.npy", mmap_mode="r")
    man = {r["id"]: r["path"] for r in map(json.loads, open(V.UV / "emb_out" / "manifest.jsonl"))}
    row0 = {}
    for k, (d, p) in enumerate(ids):
        if p == 0 and int(d) in man:
            row0[corpus.doc_id(man[int(d)])] = k
    row1 = {}
    for k, (d, p) in enumerate(ids):
        if p == 1 and int(d) in man:
            row1[corpus.doc_id(man[int(d)])] = k
    for folder, c in FOLDERS.items():
        docs = [(d, p) for d, p in paths.items() if str(p).split("/Layouts/")[-1].split("/Refine_DS/")[-1].startswith(folder + "/") and d in row0 and d not in have]
        rng.shuffle(docs)
        n = 0
        for d, p in docs:
            if n >= per_folder:
                break
            usable = cls.get(d) in LAYER and p.exists()
            cand = [(0, row0[d])]
            if c == "scientific":                       # every embedded page (0, 1) on its own: a row when it shows formulas / plots / tables
                cand = [(q, r[d]) for q, r in ((0, row0), (1, row1)) if d in r and usable and science(p, q)]
            if not cand:
                continue
            for no, k in cand:
                rows.append((c, d, "folder", f"{folder}/{p.name}#{no}", np.asarray(emb[k], np.float32), str(p) if usable else None,
                             no if usable else -1))
            n += 1
    jobs = [(r[5], r[6]) for r in rows if r[5]]
    with ProcessPoolExecutor(6) as ex:
        feats = iter(list(ex.map(_feat, jobs, chunksize=8)))
    X = np.full((len(rows), NF), np.nan, np.float32)
    for i, r in enumerate(rows):
        if r[5]:
            f = next(feats)
            if f is not None:
                X[i] = f
    np.savez(DATA, label=np.array([r[0] for r in rows]), doc=np.array([r[1] for r in rows]), source=np.array([r[2] for r in rows]),
             id=np.array([r[3] for r in rows]), E=np.array([r[4] for r in rows], np.float32), X=X,
             pdf=np.array([r[5] or "" for r in rows]), page=np.array([r[6] for r in rows]))
    print(Counter((r[2], r[0]) for r in rows))
    print(f"{len(rows)} pages, formatting features for {int((~np.isnan(X).any(1)).sum())}")


def lang_weights(y, docs):
    """Sample weights that give every class the language mix of the digital corpus (plan.cls text) and every class the same total:
    w = corpus share of the page's language / share of that language inside its class (languages missing from a class keep weight 1;
    pages without a language label count as their own group)."""
    db = corpus.connect()
    lang = corpus.languages(db, min_conf=0.0)
    cor = Counter(lang.get(d, "?") for d, _, c, _ in db.execute("select doc,new_path,cls,pages from plan") if c == "text")
    tot = sum(cor.values())
    lg = np.array([lang.get(d, "?") or "?" for d in docs])
    w = np.ones(len(y))
    for c in np.unique(y):
        m = y == c
        inside = Counter(lg[m])
        for l, n in inside.items():
            w[m & (lg == l)] = (cor.get(l, 0) / tot or 1.0 / tot) / (n / m.sum())
        w[m] *= m.sum() / w[m].sum()                     # class total unchanged
    w *= len(y) / (len(np.unique(y)) * np.bincount(y)[y])  # balanced classes
    return w / w.mean(), lg


class Weighted:
    def __init__(self, make, w):
        self.make, self.w = make, w

    def fit(self, Z, y, idx):
        m = self.make()
        try:
            m.fit(Z, y, **({"logisticregression__sample_weight": self.w[idx]} if hasattr(m, "steps") else {"sample_weight": self.w[idx]}))
        except TypeError:
            m.fit(Z, y)
        self.m = m
        return self

    def predict_proba(self, Z):
        return self.m.predict_proba(Z) if hasattr(self.m, "predict_proba") else self.m.proba(Z)


def oof(make, Z, y, g, mask=None, seed=0, w=None):
    k = y.max() + 1
    P = np.full((len(y), k), np.nan)
    idx = np.arange(len(y)) if mask is None else np.flatnonzero(mask)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(idx, y[idx], g[idx]):
        tr, te = idx[tr], idx[te]
        present = np.unique(y[tr])
        P[te] = 0
        if w is None:
            P[np.ix_(te, present)] = make().fit(Z[tr], np.searchsorted(present, y[tr])).predict_proba(Z[te])
        else:
            P[np.ix_(te, present)] = Weighted(make, w).fit(Z[tr], np.searchsorted(present, y[tr]), tr).predict_proba(Z[te])
    return P


GROUPS = {"en": "English", "?": "no language label"}


def report(name, P, y, names, mask, w=None, lg=None):
    m = mask & ~np.isnan(P).any(1)
    pr = P[m].argmax(1)
    acc = (pr == y[m]).mean()
    line = f"\n=== {name}: {m.sum()} pages, accuracy {acc:.3f}"
    if w is not None:
        line += f", class-balanced {np.average(pr == y[m], weights=w[m]):.3f}"
        by = defaultdict(list)
        for ok, l in zip(pr == y[m], lg[m]):
            by[GROUPS.get(l, "other languages")].append(ok)
        line += " | " + ", ".join(f"{k} {np.mean(v):.3f} ({len(v)})" for k, v in sorted(by.items()))
    print(line)
    print(classification_report(y[m], pr, labels=range(len(names)), target_names=names, digits=3, zero_division=0,
                                ))
    return pr


def sheets(items, out, per=10):
    from pagetype import recheck as RC
    out.mkdir(parents=True, exist_ok=True)
    by = defaultdict(list)
    for it in items:
        by[it["label"]].append(it)
    for c, its in by.items():
        for s in range(0, len(its), per):
            paths = []
            for it in its[s:s + per]:
                if it["source"] == "labelled" and not it["id"].lower().endswith(".pdf"):
                    paths.append(ROOT / it["id"])
                else:
                    f = out / "pages" / (it["id"].replace("/", "__").replace("#", "_p") + ".jpg")
                    if not f.exists() and it["pdf"]:
                        f.parent.mkdir(exist_ok=True)
                        pg = pymupdf.open(it["pdf"])[max(it["page"], 0)]
                        z = 1024 / max(pg.rect.width, pg.rect.height)
                        pg.get_pixmap(matrix=pymupdf.Matrix(z, z)).save(str(f))
                    if not f.exists():
                        continue
                    paths.append(f)
            RC.montage(paths, out / f"{c}_{s // per:02d}.jpg", cell=400, cols=5)


def evaluate(low):
    z = np.load(DATA)
    names = classes()
    y = np.array([names.index(l) for l in z["label"]])
    g, src, E, X = z["doc"], z["source"], normalize(z["E"]), z["X"]
    lab = src == "labelled"
    has_f = ~np.isnan(X).any(1)
    print("pages per class (labelled / folder / with formatting):")
    for i, c in enumerate(names):
        print(f"  {c:14s} {int(((y == i) & lab).sum()):5d} {int(((y == i) & ~lab).sum()):5d} {int(((y == i) & has_f).sum()):5d}")
    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000))
    _, lg = lang_weights(y, g)
    w = len(y) / (len(np.unique(y)) * np.bincount(y)[y])          # balanced classes; languages are balanced in the data itself
    print("English share per class (raw):", {c: round(float((lg[y == i] == "en").mean()), 2) for i, c in enumerate(names)})
    Pe = oof(lr, E, y, g, w=w)
    Pf = oof(lambda: V.xgb(), X, y, g, mask=has_f, w=w)
    Pf_full = np.where(np.isnan(Pf), 1.0, Pf)                    # no PDF layer: the embedding alone decides
    Pp = np.log(Pe + 1e-4) + np.log(Pf_full + 1e-4)
    Pp = np.exp(Pp - Pp.max(1, keepdims=True))
    Pp /= Pp.sum(1, keepdims=True)
    report("embedding, labelled pages", Pe, y, names, lab, w, lg)
    report("formatting xgb, labelled pages with a PDF layer", Pf, y, names, lab & has_f, w, lg)
    pr = report("embedding x formatting, labelled pages", Pp, y, names, lab, w, lg)
    print("confusion (rows true, cols predicted):", names)
    print(confusion_matrix(y[lab], pr, labels=range(len(names))))
    report("embedding x formatting, folder samples (folder = label)", Pp, y, names, ~lab, w, lg)
    own = Pp[np.arange(len(y)), y]
    items = [dict(id=str(z["id"][i]), pdf=str(z["pdf"][i]), page=int(z["page"][i]), label=str(z["label"][i]), source=str(src[i]), p_own=round(float(own[i]), 3),
                  predicted=names[int(Pp[i].argmax())], p_pred=round(float(Pp[i].max()), 3)) for i in np.argsort(own) if own[i] < low]
    print(f"\npages whose own label has probability < {low}: per class (labelled / folder)")
    cnt = Counter((it["label"], it["source"]) for it in items)
    for c in names:
        n_l, n_f = int(((y == names.index(c)) & lab).sum()), int(((y == names.index(c)) & ~lab).sum())
        print(f"  {c:14s} {cnt[(c, 'labelled')]:4d}/{n_l:<5d} {cnt[(c, 'folder')]:4d}/{n_f}")
    with open(V.CACHE / "pagekind_full_low.jsonl", "w") as f:
        f.writelines(json.dumps(it) + "\n" for it in items)
    sheets(items, V.CACHE / "full_review")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("data", "eval"))
    ap.add_argument("--per-folder", type=int, default=300)
    ap.add_argument("--low", type=float, default=0.2)
    a = ap.parse_args()
    data(a.per_folder) if a.cmd == "data" else evaluate(a.low)
