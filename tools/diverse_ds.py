"""Diverse page set from the corpus (dspages.curate): pool of embedded pages, scores, per-size plan of what to take, fill, check and
download. Data goes to Layouts/archive/info/diverse (OUT), never to the repo.

  pool    emb_out (SigLIP 2 page embeddings of pages 0 / 1) + doc_types + manifest plan + uv_pages/scans.jsonl (scan documents) ->
          OUT/pool.parquet (one row per page: doc, page, path, folder, scope live / rej, label, p, lang, pages, scan, c1, c2, d1)
          and OUT/X.npy (L2-normalised embeddings, float16); two-level k-means (2000 clusters, 200 groups)
  score   p_form (forms vs other pages, pagetype_labelled), p_fill (filled vs blank / other, trained on the filled-form reviews),
          tax (zero-shot type: SigLIP 2 text prompts of curate.TYPES), reviewed / real_filled from the reviews -> OUT/pool.parquet
  plan    for --size N: OUT/<N>k_fill_blanks.csv (blank digital forms to fill), <N>k_check_scans.csv (scans to check by hand: coverage
          gaps with a reserve, real filled scans, filled candidate scans), <N>k_downloads.csv (per cluster: digital pages to download in
          place of non-handwritten scans, filled forms to download; described by languages, types, example text and paths),
          <N>k_type_gaps.csv, <N>k_download_master.csv (all downloads, inflated by keep rates measured on the crawl folders),
          <N>k_handwritten.csv and <N>k_handwritten_by_lang.csv (x --hw-x for dirty sources); prints the summary
  render  full-resolution page images of a csv (columns k, path, page) for a review: OUT_DIR/batch_NNN/{<k>.jpg, in.json}

Reviews (filled forms): Layouts/archive/info/filled_review/{candidates.csv, batch_*/out.json (10-page sheets),
recheck/batch_*/out.json (full resolution)}: {"<k>": [label, bad]}; a full-resolution label wins.

python tools/diverse_ds.py pool | score
python tools/diverse_ds.py plan --size 100 [--floor 170] [--hw-x 5]
python tools/diverse_ds.py render CSV OUT_DIR [--per 50]
"""
import argparse
import glob
import json
import os
import sqlite3
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from dspages import curate as C
from dspages.hand import corpus
from dspages.hand.config import PATHS

UV = Path(PATHS.uvtm)
INFO = Path(PATHS.layouts) / "archive" / "info"
OUT = INFO / "diverse"
REVIEW = INFO / "filled_review"
MODEL = "google/siglip2-base-patch16-384"
REAL = ("filled_hand", "filled_typed")
LONG = 1500                                              # long side of a review image, px
WORKERS = 6


def docs_table():
    """One row per embedded document: doc (emb id), path, label, p, scope (live / rej), h (hash), folder, top, scan."""
    dt = pd.DataFrame([json.loads(l) for l in open(UV / "emb_out" / "doc_types.jsonl")])[["doc", "path", "label", "p"]]
    rej = dt.path.str.startswith("archive/rejected")
    dt = dt[rej | ~dt.path.str.startswith(("archive/", "_"))].copy()                 # Layouts without the archive, but its rejected
    dt["scope"] = np.where(dt.path.str.startswith("archive/rejected"), "rej", "live")
    dt["h"] = dt.path.map(corpus.doc_id)
    dt["folder"] = dt.path.map(lambda p: "/".join(p.split("/")[:2]) if p.startswith("archive") else p.split("/")[0])
    dt["top"] = dt.folder
    scans = {json.loads(l)["path"] for l in open(UV / "uv_pages" / "scans.jsonl")}
    dt["scan"] = dt.path.isin(scans) | ~dt.path.str.lower().str.endswith(".pdf")
    return dt


def cmd_pool(_):
    from sklearn.cluster import KMeans, MiniBatchKMeans
    dt = docs_table()
    db = sqlite3.connect(f"file:{PATHS.manifest}?mode=ro", uri=True)
    plan = pd.read_sql("select doc as h, lang, pages from plan", db).drop_duplicates("h").set_index("h")
    dt = dt.join(plan, on="h")
    dt["lang"] = dt.lang.fillna("?")
    dt["pages"] = dt.pages.fillna(1).clip(1, 500)
    ids = np.load(UV / "emb_out" / "ids.npy")
    E = np.load(UV / "emb_out" / "embeddings.npy", mmap_mode="r")
    P = pd.DataFrame({"row": np.arange(len(ids)), "doc": ids[:, 0], "page": ids[:, 1]}).merge(dt, on="doc")
    X = np.asarray(E[P.row.values], dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    k1 = MiniBatchKMeans(2000, random_state=0, batch_size=8192, n_init=1, max_iter=60).fit(X)
    cen = k1.cluster_centers_ / np.linalg.norm(k1.cluster_centers_, axis=1, keepdims=True)
    P["c1"] = k1.labels_
    P["d1"] = 1 - (X * cen[k1.labels_]).sum(1)
    k2 = KMeans(200, random_state=0, n_init=4).fit(cen, sample_weight=np.bincount(k1.labels_, minlength=2000))
    P["c2"] = k2.labels_[P.c1]
    OUT.mkdir(parents=True, exist_ok=True)
    P.drop(columns="row").to_parquet(OUT / "pool.parquet")
    np.save(OUT / "X.npy", X.astype(np.float16))
    print(len(P), "pages", P.doc.nunique(), "documents", "scan share", round(P.scan.mean(), 3), "->", OUT)


def reviews():
    """Filled-form reviews: doc, page, label, bad, hires (full-resolution label)."""
    cand = pd.read_csv(REVIEW / "candidates.csv")
    lab = {}
    for hires, pat in ((False, "batch_*/out.json"), (True, "recheck/batch_*/out.json")):
        for f in sorted(glob.glob(str(REVIEW / pat))):
            for k, v in json.load(open(f)).items():
                lab[k] = (v[0], int(v[1]) if len(v) > 1 and v[1] is not None else 0, hires)
    cand = cand[cand.k.isin(lab)].copy()
    cand[["label", "bad", "hires"]] = [lab[k] for k in cand.k]
    return cand[["k", "doc", "page", "lang", "label", "bad", "hires"]]


def text_types(X):
    """Zero-shot type of every page: the closest text prompt of curate.TYPES."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok, m = AutoTokenizer.from_pretrained(MODEL), AutoModel.from_pretrained(MODEL).eval()
    with torch.inference_mode():
        t = tok([C.PROMPT.format(x) for x in C.TYPES], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        o = m.get_text_features(**t)
        T = (o.pooler_output if hasattr(o, "pooler_output") else o).numpy()
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    sim = np.concatenate([X[i:i + 50000] @ T.T for i in range(0, len(X), 50000)])
    return np.array(C.TYPES)[sim.argmax(1)], sim.max(1)


def cmd_score(_):
    from sklearn.linear_model import LogisticRegression
    P = pd.read_parquet(OUT / "pool.parquet")
    X = np.load(OUT / "X.npy").astype(np.float32)
    z = np.load(UV / "pagetype_labelled" / "cache" / "pagekind_full.npz")
    lab, src, E = z["label"], z["source"], z["E"] / np.linalg.norm(z["E"], axis=1, keepdims=True)
    m = src == "labelled"
    form = LogisticRegression(C=2, max_iter=1000, class_weight="balanced").fit(E[m], np.isin(lab[m], ["forms_blank", "forms_filled"]))
    P["p_form"] = form.predict_proba(X)[:, 1]
    R = reviews()
    pos = dict(zip(zip(P.doc, P.page), range(len(P))))
    R["i"] = [pos.get(k, -1) for k in zip(R.doc, R.page)]
    R = R[R.i >= 0]
    rng = np.random.default_rng(0)
    other = P.index[(~P.label.isin(["forms_blank", "forms_filled", "handwritten"])) & (P.p > 0.8)].values
    blank = P.index[(P.label == "forms_blank") & (P.p > 0.8) & (P.folder == "forms")].values
    neg = np.r_[rng.choice(other, min(6000, len(other)), replace=False), rng.choice(blank, min(3000, len(blank)), replace=False)]
    Xt = np.vstack([X[R.i.values], X[neg]])
    yt = np.r_[R.label.isin(REAL).values, np.zeros(len(neg), bool)]
    P["p_fill"] = LogisticRegression(C=4, max_iter=2000, class_weight="balanced").fit(Xt, yt).predict_proba(X)[:, 1]
    key = pd.Series(list(zip(P.doc, P.page)), index=P.index)
    real = R[R.hires & R.label.isin(REAL) & (R.bad == 0)]                       # only full-resolution checks count as real
    P["reviewed"] = key.isin(set(zip(R.doc, R.page)))
    P["real_filled"] = key.isin(set(zip(real.doc, real.page)))
    P["tax"], P["tax_sim"] = text_types(X)
    P.to_parquet(OUT / "pool.parquet")
    print("forms", int((P.p_form > C.FORM_P).sum()), "filled-looking", int((P.p_fill > C.FILL_HI).sum()), "reviewed", int(P.reviewed.sum()),
          "real filled", int(P.real_filled.sum()))


def snippet(db, h, n=90):
    r = db.execute("select text from lang_text where doc=?", (h,)).fetchone()
    if not r or r[0] is None:
        return ""
    t = r[0]
    try:
        t = zlib.decompress(t).decode("utf8", "ignore")
    except Exception:
        t = t.decode("utf8", "ignore") if isinstance(t, bytes) else str(t)
    return " ".join(t.split())[:n]


def shares(s, k=3):
    return ", ".join(f"{a} {b:.0%}" for a, b in s.value_counts(normalize=True).head(k).items())


def describe(P, c, db):
    """Languages, types, scan share, example text and paths (the 3 pages closest to the centre) of cluster c."""
    s = P[P.c1 == c].sort_values("d1")
    ex = s.head(3)
    return dict(langs=shares(s.lang), types=shares(s.label, 2), doc_types=shares(s.tax), scan_share=round(s.scan.mean(), 2),
                pages_in_corpus=len(s), example_text=" | ".join(filter(None, (snippet(db, h) for h in ex.h))),
                example_paths=" ; ".join(ex.path))


def spread(df, by, n, seed=0):
    """Up to n[c] rows of every group c: one per document, languages in turn."""
    d = df.sample(frac=1, random_state=seed).drop_duplicates("doc")
    d = d.assign(r=d.groupby([by, "lang"]).cumcount()).sort_values([by, "r"])
    return pd.concat([d[d[by] == c].head(int(k)) for c, k in n.items() if k > 0]) if (n > 0).any() else d.head(0)


def cmd_plan(a):
    B, tag = a.size * 1000, f"{a.size}k"
    floor = a.floor if a.floor is not None else round(1000 * a.size / 600)
    P = C.keep_pages(pd.read_parquet(OUT / "pool.parquet"))
    sup = C.supply(P)
    q1 = C.allocate(sup, B)
    scans = C.scans_needed(sup, q1)
    w = C.page_weight(P, sup, q1)
    f = C.forms_plan(P, sup, q1)
    db = sqlite3.connect(f"file:{PATHS.manifest}?mode=ro", uri=True)
    keep = C.keep_rates(docs_table())
    R = reviews()
    dig = R[R.hires & ~R.doc.isin(P[P.scan].doc)]
    keep_filled = dig.label.isin(REAL).mean() * 0.9 if len(dig) else 0.2      # digital filled-looking pages that are really filled
    # 1) blank digital forms to fill
    F = P[(P.p_form > C.FORM_P) & (P.p_fill < C.FILL_LO) & ~P.scan]
    spread(F, "c1", f.fill)[["doc", "page", "path", "lang", "c1"]].to_csv(OUT / f"{tag}_fill_blanks.csv", index=False)
    # 2) scans to check by hand: coverage gaps (+50 % reserve, take = False), real filled scans, filled candidate scans
    S = P[P.scan]
    gap = spread(S, "c1", np.ceil(1.5 * scans))
    gap = gap.assign(why="coverage_gap", take=gap.groupby("c1").cumcount() < gap.c1.map(scans))
    rf = spread(S[S.real_filled], "c1", f.real)
    cs = S[(S.p_form > C.FORM_P) & (S.p_fill > C.FILL_HI) & ~S.reviewed].sort_values("p_fill", ascending=False)
    cs = cs[cs.groupby("c1").cumcount() < cs.c1.map(f.cand_scan_use).fillna(0)]
    chk = pd.concat([gap, rf.assign(why="real_filled", take=True), cs.assign(why="filled_candidate", take=True)])
    chk[["why", "take", "doc", "page", "path", "lang", "c1", "label"]].to_csv(OUT / f"{tag}_check_scans.csv", index=False)
    # 3) per cluster downloads: digital pages in place of non-handwritten scans, filled forms (no candidate, or only scan candidates)
    hw = (P.label == "handwritten").groupby(P.c1).mean()
    D = pd.DataFrame({"gap_scans": scans, "hw_scans": (scans * hw).round()}).join(
        (f.download + f.cand_scan_use).rename("filled_forms")).fillna(0)
    D["pages"] = D.gap_scans - D.hw_scans
    D = D[(D.pages > 0) | (D.filled_forms > 0) | (D.hw_scans > 0)]
    D = D.join(pd.DataFrame([describe(P, c, db) for c in D.index], index=D.index))
    D.sort_values(["pages", "filled_forms"], ascending=False).to_csv(OUT / f"{tag}_downloads.csv", index_label="cluster")
    # 4) type floors
    T = C.type_floor(P, w, floor)
    T.to_csv(OUT / f"{tag}_type_gaps.csv", index_label="type")
    # 5) master download list, inflated by keep rates
    main = D.types.str.split(" ").str[0]
    k_pages = main.map(keep).fillna(keep.get("plain", 0.3)).where(main != "forms_filled", keep_filled)
    tg = T[T.download > 0]
    M = pd.concat([D[D.pages > 0].assign(part="coverage_gap", need=D.pages, keep=k_pages),
                   D[D.filled_forms > 0].assign(part="filled_form", need=D.filled_forms, keep=keep_filled),
                   pd.DataFrame({"part": "type_gap", "need": tg.download, "keep": keep.get("plain", 0.3), "doc_types": tg.index}, index=tg.index)])
    M["download"] = C.inflate(M.need, M.keep)
    cols = ["part", "need", "keep", "download", "langs", "types", "doc_types", "scan_share", "example_text", "example_paths"]
    M[cols].sort_values(["part", "download"], ascending=[True, False]).to_csv(OUT / f"{tag}_download_master.csv", index_label="cluster_or_type")
    # 6) handwriting: type floors + the coverage shortfall of the handwritten clusters, x hw_x; split by the set's languages
    short = max(0.0, float((C.demand(sup, B) * hw).sum() - (q1 * hw).sum()))
    H = T.loc[T.index.isin(C.HW_TYPES), ["in_set", "pages", "scans", "handwritten"]].rename(columns={"handwritten": "need"})
    H.loc["handwritten (general)"] = [round((q1 * hw).sum()), int((P.label == "handwritten").sum()), int(((P.label == "handwritten") & P.scan).sum()), round(short)]
    H["collect"] = (H.need * a.hw_x).round().astype(int)
    H.to_csv(OUT / f"{tag}_handwritten.csv", index_label="type")
    L = (w.groupby(P.lang).sum() / w.sum()).sort_values(ascending=False)
    L = pd.concat([L.head(40), pd.Series({"other": L.iloc[40:].sum()})])
    HL = pd.DataFrame([dict(type=t, lang=l, collect=round(r.collect * p)) for t, r in H.iterrows() for l, p in L.items()])
    HL[HL.collect > 0].to_csv(OUT / f"{tag}_handwritten_by_lang.csv", index=False)
    print(f"{tag}: scans to check {int(scans.sum())} (+ real filled {int(f.real.sum())}, candidates {int(f.cand_scan_use.sum())});"
          f" filled forms {int(f.need.sum())} = real {int(f.real.sum())} + fill {int(f.fill.sum())} + digital candidates"
          f" {int(f.cand_dig_use.sum())} + scan candidates {int(f.cand_scan_use.sum())} + download {int(f.download.sum())}")
    print(f"  downloads: {M.groupby('part').need.sum().astype(int).to_dict()} kept pages -> {M.groupby('part').download.sum().to_dict()} documents;"
          f" handwriting need {int(H.need.sum())}, collect {int(H.collect.sum())} -> {OUT}")


def render(job):
    k, path, page, out = job
    import cv2
    import pymupdf
    try:
        p = corpus.resolve(path)
        if not p.exists():                              # moved / renamed since the embedding: find the document by its hash
            alts = sorted(HASHES.get(corpus.doc_id(path), []), key=lambda x: not x.lower().endswith(".pdf"))
            p, page = (corpus.resolve(alts[0]), page if alts[0].lower().endswith(".pdf") else 0) if alts else (p, page)
        if str(p).lower().endswith(".pdf"):
            d = pymupdf.open(p)
            pg = d[min(int(page), len(d) - 1)]
            s = LONG / max(pg.rect.width, pg.rect.height)
            pix = pg.get_pixmap(matrix=pymupdf.Matrix(s, s), colorspace=pymupdf.csRGB)
            im = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, 3)[:, :, ::-1]
        else:
            im = cv2.imread(str(p))
            s = LONG / max(im.shape[:2])
            im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return k, cv2.imwrite(str(out), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    except Exception:
        return k, False


def hash_index():
    """{document hash: [paths relative to Layouts]} of Layouts without the archive (but its rejected)."""
    idx = {}
    for root, _, files in os.walk(PATHS.layouts):
        rel = os.path.relpath(root, PATHS.layouts)
        if rel.startswith("archive") and not rel.startswith("archive/rejected"):
            continue
        for f in files:
            idx.setdefault(corpus.doc_id(f), []).append(os.path.join(rel, f))
    return idx


HASHES = {}


def _init(h):
    HASHES.update(h)


def cmd_render(a):
    df, out = pd.read_csv(a.csv), Path(a.out)
    jobs = []
    for b in range(0, len(df), a.per):
        d = out / f"batch_{b // a.per:03d}"
        d.mkdir(parents=True, exist_ok=True)
        part = df.iloc[b:b + a.per]
        (d / "in.json").write_text(json.dumps(list(part.k)))
        jobs += [(r.k, r.path, r.page, d / f"{r.k}.jpg") for r in part.itertuples()]
    with ProcessPoolExecutor(WORKERS, initializer=_init, initargs=(hash_index(),)) as ex:
        ok = dict(ex.map(render, jobs, chunksize=8))
    print(len(df), "pages", sum(ok.values()), "rendered in", -(-len(df) // a.per), "batches")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("pool")
    sp.add_parser("score")
    p = sp.add_parser("plan")
    p.add_argument("--size", type=int, required=True, help="set size in thousands of pages")
    p.add_argument("--floor", type=int, help="pages of every document type (default 1000 per 600k)")
    p.add_argument("--hw-x", type=float, default=5.0, help="handwriting to collect per page needed (dirty sources)")
    r = sp.add_parser("render")
    r.add_argument("csv")
    r.add_argument("out")
    r.add_argument("--per", type=int, default=50)
    a = ap.parse_args()
    {"pool": cmd_pool, "score": cmd_score, "plan": cmd_plan, "render": cmd_render}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
