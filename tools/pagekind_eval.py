"""Page kind from formatting (dspages.hand.pagekind: fast line features + reader features), from image embeddings, and from both:
grouped cross-validation (pages of one document in one fold, seeds x 5 folds) on the labelled pages matched to their PDFs
(tools/pagekind_sources.py), feature importance, the pages most often wrong.

Embeddings: `page` = the embedding of the labelled image (pagetype_labelled/cache/labelled_embeddings.npz; the labelled images without a
source PDF join the training folds of the embedding models as extra data); `doc` = the corpus embedding (emb_out) of that very page
(pages 0-1 of every document were embedded).
Features are cached next to the source list (pagekind_feats.npz); --refresh recomputes them.

python tools/pagekind_eval.py [--classes plain,poems,formatted] [--relabel FILE] [--fit] [--errors] [--importance]
"""
import argparse
import json
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pymupdf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from xgboost import XGBClassifier

from dspages.hand import corpus
from dspages.hand import pagekind as PK
from dspages.hand.config import PATHS

UV = Path(PATHS.uvtm)
CACHE = UV / "pagetype_labelled" / "cache"
SRC = CACHE / "pagekind_src.jsonl"
PY = Path(__file__).resolve().parents[1] / "src" / "dspages" / "hand" / "pagekind.py"


def _feats(job):
    pdf, no = job
    page = pymupdf.open(pdf)[no]
    t0 = time.perf_counter()
    a = PK.features(page)
    t1 = time.perf_counter()
    b = PK.reader_features(page)
    return a, b, t1 - t0, time.perf_counter() - t1


def features(recs, refresh=False):
    f = CACHE / "pagekind_feats.npz"
    keys = [f"{r['image']}|{r['page']}" for r in recs]
    if f.exists() and not refresh:
        z = np.load(f)
        if list(z["keys"]) == keys and z["X"].shape[1] == len(PK.NAMES) and z["Y"].shape[1] == len(PK.READER):
            return z["X"], z["Y"], z["t"]
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(_feats, [(r["pdf"], r["page"]) for r in recs], chunksize=4))
    X, Y = np.array([r[0] for r in res]), np.array([r[1] for r in res])
    t = np.array([(r[2], r[3]) for r in res])
    np.savez(f, keys=np.array(keys), X=X, Y=Y, t=t)
    return X, Y, t


def doc_embeddings(docs):
    ids = np.load(UV / "emb_out" / "ids.npy")
    emb = np.load(UV / "emb_out" / "embeddings.npy", mmap_mode="r")
    paths = {r["id"]: r["path"] for r in map(json.loads, open(UV / "emb_out" / "manifest.jsonl"))}     # ids are not line numbers
    want = {i for i, p in paths.items() if corpus.doc_id(p) in docs}
    return {(corpus.doc_id(paths[d]), int(p)): np.asarray(emb[k], np.float32) for k, (d, p) in enumerate(ids) if d in want}


def page_embeddings():
    """{image stem: embedding} of the labelled images, and of the pages added later (pagekind_new_embeddings.npz)."""
    lab = np.load(CACHE / "labelled_embeddings.npz")
    out = {k.split("/")[-1]: v for k, v in zip(lab["keys"], lab["X"])}
    if (CACHE / "pagekind_new_embeddings.npz").exists():
        new = np.load(CACHE / "pagekind_new_embeddings.npz")
        out.update(zip(new["keys"], new["X"]))
    return out


def load(src, classes, any_layer, refresh, relabel=None):
    recs = [r for r in map(json.loads, open(src)) if r["ok"]]
    X, Y, t = features(recs, refresh)
    if relabel:                                         # label corrections from a visual review: {"<label dir>/<file>": {old, new}}
        fix = json.load(open(relabel))
        for r in recs:
            key = f"{Path(r['image']).parent.name}/{Path(r['image']).name}"
            if key in fix:
                r["label"] = fix[key]["new"]
    cls = {d: c for d, c in corpus.connect().execute("select doc,cls from plan")}
    for r in recs:
        r["doc"] = corpus.doc_id(r["pdf"])
        r["cls"] = cls.get(r["doc"])
    keep = np.array([r["label"] in classes and (any_layer or r["cls"] == "text") for r in recs])
    recs = [r for r, k in zip(recs, keep) if k]
    X, Y, t = X[keep], Y[keep], t[keep]
    pe = page_embeddings()
    de = doc_embeddings({r["doc"] for r in recs})
    nan = np.full(768, np.nan, np.float32)
    E = np.array([pe.get(Path(r["image"]).stem, nan) for r in recs])
    D = np.array([de.get((r["doc"], r["page"]), nan) for r in recs])            # the corpus embedding of that very page
    y = np.array([classes.index(r["label"]) for r in recs])
    g = np.array([r["doc"] for r in recs])
    return X, Y, E, D, y, g, recs, t


def extra_embeddings(classes, used):
    """Labelled images of the classes that are not in the evaluation set (no source PDF, scans, no text layer): more training data for
    the embedding models. Group: the document part of the file name."""
    pe, out = page_embeddings(), []
    root = UV / "pagetype_labelled"
    for c in classes:
        for f in sorted((root / c).glob("*.jpg")):
            if str(f) not in used and f.stem in pe:
                m = re.match(r"^[a-z0-9]+_([0-9a-f]{12}|\d{4})", f.stem)
                out.append((pe[f.stem], classes.index(c), m.group(1) if m else f.stem))
    if not out:
        return None
    return normalize(np.array([o[0] for o in out])), np.array([o[1] for o in out]), np.array([o[2] for o in out])


def balanced_w(y):
    w = len(y) / (len(set(y)) * np.bincount(y, minlength=y.max() + 1))
    return w[y]


def xgb(**kw):
    p = dict(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5, min_child_weight=2, n_jobs=6, verbosity=0)
    p.update(kw)
    return XGBClassifier(**p)


class Model:
    """fit(X, y) / proba(X) with balanced class weights: xgb, or logistic regression on standardised features."""

    def __init__(self, kind, **kw):
        self.kind, self.kw = kind, kw

    def fit(self, X, y):
        if self.kind == "xgb":
            self.m = xgb(**self.kw).fit(X, y, sample_weight=balanced_w(y))
        else:
            self.m = make_pipeline(StandardScaler(), LogisticRegression(C=self.kw.get("C", 0.3), max_iter=5000, class_weight="balanced")).fit(X, y)
        return self

    def proba(self, X):
        return self.m.predict_proba(X)


def _train(Z, y, tr, te, g, extra):
    Zt, yt = Z[tr], y[tr]
    if extra is not None:
        ok = ~np.isin(extra[2], g[te])
        Zt, yt = np.vstack([Zt, extra[0][ok]]), np.r_[yt, extra[1][ok]]
    return Zt, yt


def cv_proba(make, Z, y, g, seed, extra=None):
    """Out-of-fold class probabilities; extra = (Z, y, groups) joins every training fold (without the groups of the test fold)."""
    P = np.zeros((len(y), len(set(y))))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(Z, y, g):
        Zt, yt = _train(Z, y, tr, te, g, extra)
        present = np.unique(yt)                         # a small subset can miss a class in a training fold
        remap = np.searchsorted(present, yt)
        P[np.ix_(te, present)] = make().fit(Zt, remap).proba(Z[te])
    return P


def stacked(parts, y, g, seed):
    """Second level: logistic regression on the log probabilities of the first-level models, out of fold inside every outer training
    fold (nested, so the test fold never reaches either level)."""
    P = np.zeros((len(y), len(set(y))))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed + 100).split(y, y, g):
        inner = [cv_proba(make, Z[tr], y[tr], g[tr], seed, ex) for make, Z, ex in parts]
        outer = [make().fit(*_train(Z, y, tr, te, g, ex)).proba(Z[te]) for make, Z, ex in parts]
        meta = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit(np.log(np.hstack(inner) + 1e-4), y[tr])
        P[te] = meta.predict_proba(np.log(np.hstack(outer) + 1e-4))
    return P


def report(name, Ps, y, classes):
    preds = [P.argmax(1) for P in Ps]
    acc = np.array([(p == y).mean() for p in preds])
    f1 = np.array([f1_score(y, p, average="macro") for p in preds])
    per = f1_score(y, preds[0], average=None)
    print(f"{name:46s} acc {acc.mean():.3f} +- {acc.std():.3f}  err {1 - acc.mean():.3f}  macro-F1 {f1.mean():.3f}  "
          + " ".join(f"{c}={v:.2f}" for c, v in zip(classes, per)), flush=True)


def standard(name, Ps, y, classes):
    """Precision / recall / F1 / support per class, macro and weighted averages, accuracy, one-vs-rest ROC-AUC, log loss and the
    confusion matrix (rows: true, columns: predicted) of out-of-fold predictions, seed 0; AUC and log loss averaged over seeds."""
    P = np.exp(Ps[0] - Ps[0].max(1, keepdims=True)) if Ps[0].min() < 0 else Ps[0]
    P = P / P.sum(1, keepdims=True)
    norm = [np.exp(q - q.max(1, keepdims=True)) if q.min() < 0 else q for q in Ps]
    norm = [q / q.sum(1, keepdims=True) for q in norm]
    auc = np.mean([roc_auc_score(y, q, multi_class="ovr", average="macro") for q in norm])
    ll = np.mean([log_loss(y, q, labels=range(len(classes))) for q in norm])
    print(f"\n=== {name} ===  ROC-AUC (macro, one-vs-rest) {auc:.3f}  log loss {ll:.3f}")
    print(classification_report(y, P.argmax(1), target_names=classes, digits=3))
    print("confusion (rows true, cols predicted):", classes)
    print(confusion_matrix(y, P.argmax(1)))


def importance(X, y, g, names, seeds=3):
    """Grouped permutation importance of the xgb formatting model (drop of out-of-fold accuracy when one feature is shuffled) and gain."""
    rng = np.random.default_rng(0)
    drop = np.zeros(X.shape[1])
    for s in range(seeds):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, y, g):
            m = Model("xgb").fit(X[tr], y[tr])
            base = (m.proba(X[te]).argmax(1) == y[te]).mean()
            for j in range(X.shape[1]):
                Xp = X[te].copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                drop[j] += (base - (m.proba(Xp).argmax(1) == y[te]).mean()) / (5 * seeds)
    gain = xgb().fit(X, y, sample_weight=balanced_w(y)).get_booster().get_score(importance_type="gain")
    gain = np.array([gain.get(f"f{j}", 0.0) for j in range(X.shape[1])])
    print("\nfeature importance: permutation (accuracy drop, grouped CV) | xgb gain share")
    for j in np.argsort(-drop):
        print(f"  {names[j]:16s} {drop[j]:+.4f}  {gain[j] / gain.sum():.3f}")


def write_model(m, classes):
    sc, lr = m[0], m[1]
    fmt = lambda a: "np.array([" + ", ".join(f"{v:.6g}" for v in a) + "])"
    code = (f"MODEL_KINDS = {tuple(classes)!r}\nMEAN = {fmt(sc.mean_)}\nSCALE = {fmt(sc.scale_)}\n"
            f"COEF = np.array([{', '.join(fmt(r) for r in lr.coef_)}])\nINTERCEPT = {fmt(lr.intercept_)}\n")
    s = PY.read_text()
    PY.write_text(re.sub(r"MODEL_KINDS = .*?\nINTERCEPT = [^\n]*\n", lambda _: code, s, flags=re.S))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--classes", default="plain,poems,formatted")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--errors", action="store_true")
    ap.add_argument("--importance", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--relabel", help="json of label corrections (e.g. pagetype_labelled/cache/pagekind_relabel.json)")
    ap.add_argument("--any-layer", action="store_true", help="also pages of scan-like documents (plan.cls != text)")
    a = ap.parse_args()
    classes = a.classes.split(",")
    X, Y, E, D, y, g, recs, t = load(a.src, classes, a.any_layer, a.refresh, a.relabel)
    print(f"{len(y)} pages, {len(set(g))} documents, {dict(zip(classes, np.bincount(y)))}; fast features {1000 * np.median(t[:, 0]):.0f} ms "
          f"median, reader {1000 * np.median(t[:, 1]):.0f} ms median")
    XY = np.hstack([X, Y])
    if a.importance:
        importance(XY, y, g, list(PK.NAMES) + list(PK.READER))
    Ee = extra_embeddings(classes, {r["image"] for r in recs})
    print(f"extra labelled images for the embedding models: {0 if Ee is None else len(Ee[1])}")
    En = normalize(E)
    runs = {
        "fast format, logreg": (lambda: Model("lr"), X, None),
        "fast format, xgb": (lambda: Model("xgb"), X, None),
        "fast + reader format, logreg": (lambda: Model("lr"), XY, None),
        "fast + reader format, xgb": (lambda: Model("xgb"), XY, None),
        "page emb, logreg C=0.05": (lambda: Model("lr", C=0.05), En, None),
        "page emb, logreg C=0.3": (lambda: Model("lr", C=0.3), En, None),
        "page emb + extra images, logreg C=0.3": (lambda: Model("lr", C=0.3), En, Ee),
        "format + page emb, concat logreg": (lambda: Model("lr", C=0.05), np.hstack([XY, En]), None),
    }
    P = {}
    for name, (make, Z, ex) in runs.items():
        P[name] = [cv_proba(make, Z, y, g, s, ex) for s in range(a.seeds)]
        report(name, P[name], y, classes)
    prod = [np.log(P["fast + reader format, xgb"][s] + 1e-4) + np.log(P["page emb + extra images, logreg C=0.3"][s] + 1e-4) for s in range(a.seeds)]
    report("product: format xgb x page emb (+extra)", prod, y, classes)
    stacks = {"stack: format xgb + page emb": [(lambda: Model("xgb"), XY, None), (lambda: Model("lr", C=0.3), En, None)],
              "stack: format xgb + page emb (+extra)": [(lambda: Model("xgb"), XY, None), (lambda: Model("lr", C=0.3), En, Ee)]}
    for name, parts in stacks.items():
        P[name] = [stacked(parts, y, g, s) for s in range(a.seeds)]
        report(name, P[name], y, classes)
    for name in ("fast format, logreg", "fast + reader format, xgb", "page emb, logreg C=0.05", "format + page emb, concat logreg"):
        standard(name, P[name], y, classes)
    standard("product: format xgb x page emb (+extra)", prod, y, classes)
    has_doc = ~np.isnan(D).any(1)
    if has_doc.sum() >= 50:
        Dn, ys, gs = normalize(D[has_doc]), y[has_doc], g[has_doc]
        print(f"\npages with a corpus page embedding: {has_doc.sum()}")
        for name, (make, Z) in {"corpus page emb, logreg": (lambda: Model("lr", C=0.3), Dn),
                                "format xgb (same pages)": (lambda: Model("xgb"), XY[has_doc]),
                                "format + corpus page emb, concat logreg": (lambda: Model("lr", C=0.05), np.hstack([XY[has_doc], Dn]))}.items():
            report(name, [cv_proba(make, Z, ys, gs, s) for s in range(a.seeds)], ys, classes)
    if a.errors:
        best = "stack: format xgb + page emb (+extra)"
        pr = np.array([p.argmax(1) for p in P[best]])
        print(f"\n{best}: confusion (seed 0)\n", confusion_matrix(y, pr[0]))
        wrong = (pr != y).mean(0)
        for i in np.argsort(-wrong):
            if wrong[i] < 0.5:
                break
            print(f"  {classes[y[i]]:>12} -> {classes[np.bincount(pr[:, i]).argmax()]:<12} {Path(recs[i]['image']).parent.name}/{Path(recs[i]['image']).name}")
    if a.fit:
        write_model(Model("lr").fit(X, y).m, classes)
        print("fast logistic model written to", PY)


if __name__ == "__main__":
    main()
