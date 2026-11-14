"""Diverse page set from the corpus (dspages.curate): pool of embedded pages, scores, per-size plan of what to take, fill, check and
download. Data goes to Layouts/archive/info/diverse (OUT), never to the repo.

  pool    emb_out (SigLIP 2 page embeddings of pages 0 / 1, manifest.jsonl) + doc_types (labels; new documents: by embedding) +
          manifest plan (language, pages, scan class) ->
          OUT/pool.parquet (one row per page: doc, page, path, folder, scope live / rej, label, p, lang, pages, scan, c1, c2, d1)
          and OUT/X.npy (L2-normalised embeddings, float16); two-level k-means (2000 clusters, 200 groups)
  geom    page ratio (h / w) and native px of the raster of every pool page (metadata only) -> OUT/geom.parquet
  score   p_form (forms vs other pages, pagetype_labelled), p_fill (filled vs blank / other, trained on the filled-form reviews),
          tax (zero-shot type: SigLIP 2 text prompts of curate.TYPES), reviewed / real_filled from the reviews -> OUT/pool.parquet
  plan    for --size N: OUT/<N>k_fill_blanks.csv (blank digital forms to fill), <N>k_check_scans.csv (scans to check for warping: coverage
          gaps x --margin, real filled scans), <N>k_check_filled.csv (filled-looking candidates / their reviewed filled share), <N>k_downloads.csv (per cluster: digital pages to download in
          place of non-handwritten scans, filled forms to download; described by languages, types, example text and paths),
          <N>k_type_gaps.csv, <N>k_download_master.csv (all downloads, inflated by keep rates measured on the crawl folders),
          <N>k_handwritten.csv and <N>k_handwritten_by_lang.csv (x --hw-x for dirty sources); prints the summary
  export  the check lists of a plan as full-resolution jpgs: archive/<N>k_warp_check, archive/<N>k_filled_check (+ index.csv)
  render  full-resolution page images of a csv (columns k, path, page) for a review: OUT_DIR/batch_NNN/{<k>.jpg, in.json}

Reviews (filled forms): Layouts/archive/info/filled_review/{candidates.csv, batch_*/out.json (10-page sheets),
recheck/batch_*/out.json (full resolution)}: {"<k>": [label, bad]}; a full-resolution label wins.

python tools/diverse_ds.py pool | geom | score
python tools/diverse_ds.py plan --size 100 [--floor 170] [--hw-x 5] [--margin 1.5]
python tools/diverse_ds.py export --size 100
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
MODEL = "google/siglip2-base-patch16-naflex"            # text tower of the page embeddings (pooled half, [0:768])
POOLED = 768
REAL = ("filled_hand", "filled_typed")
LONG = 1500                                              # long side of a review image, px
WORKERS = 6
CLEAN_FOLDERS = ("certificates", "flyers")                 # cleaned of warped scans as a whole


def docs_table():
    """One row per embedded document (emb_out/manifest.jsonl): doc (emb id), path, h (hash), folder, scope (live / rej), label / p of
    doc_types (by path, else by hash; NaN for documents added after it), scan (manifest plan class scan-like, a pdf without a plan
    row, or a picture)."""
    dt = pd.DataFrame([json.loads(l) for l in open(UV / "emb_out" / "manifest.jsonl")]).rename(columns={"id": "doc"})
    dt = dt.drop_duplicates("path")                                   # a path embedded twice (two runs): the first id
    dt["folder"] = dt.path.map(lambda p: "/".join(p.split("/")[:2]) if p.startswith("archive/") else p.split("/")[0])
    dt = dt[(dt.folder == "archive/rejected") | ~dt.path.str.startswith(("archive/", "_"))].copy()   # Layouts without the archive, but its rejected
    dt["scope"] = np.where(dt.folder == "archive/rejected", "rej", "live")
    dt["h"] = dt.path.map(corpus.doc_id)
    old = pd.DataFrame([json.loads(l) for l in open(UV / "emb_out" / "doc_types.jsonl")])[["path", "label", "p"]]
    old["h"] = old.path.map(corpus.doc_id)
    lab = old.drop_duplicates("path").set_index("path")[["label", "p"]]
    dt = dt.join(lab, on="path")
    by_h = old.drop_duplicates("h").set_index("h")[["label", "p"]]
    miss = dt.label.isna()
    dt.loc[miss, ["label", "p"]] = by_h.reindex(dt.h[miss]).values
    db = sqlite3.connect(f"file:{PATHS.manifest}?mode=ro", uri=True)
    cls = dict(db.execute("select doc, cls from plan"))
    pdf = dt.path.str.lower().str.endswith(".pdf")
    c = dt.h.map(cls)
    dt["scan"] = ~pdf | (c == "scan-like") | c.isna()
    return dt


def unit_rows(X):
    """L2-normalised rows; 1536-d page vectors [pooled | mean patch token of a middle layer]: each half normalised, joined, / sqrt 2."""
    def n(a):
        return a / np.linalg.norm(a, axis=1, keepdims=True)
    if X.shape[1] == 2 * POOLED:
        return np.hstack([n(X[:, :POOLED]), n(X[:, POOLED:])]) / np.sqrt(2)
    return n(X)


def fill_labels(P, X):
    """Labels of the documents doc_types does not know: logistic regression of the page embedding on the known labels."""
    from sklearn.linear_model import LogisticRegression
    known = P.label.notna().values
    if known.all():
        return P
    rng = np.random.default_rng(0)
    tr = rng.choice(np.where(known)[0], min(60000, known.sum()), replace=False)
    lr = LogisticRegression(max_iter=500, C=2).fit(X[tr], P.label.values[tr])
    pr = lr.predict_proba(X[~known])
    P.loc[~known, "label"] = lr.classes_[pr.argmax(1)]
    P.loc[~known, "p"] = pr.max(1)
    P["label_src"] = np.where(known, "doc_types", "embedding")
    return P


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
    P["n_emb"] = P.groupby("doc").doc.transform("size")             # embedded pages of the document (pages beyond them: not embedded)
    X = unit_rows(np.asarray(E[P.row.values], dtype=np.float32))
    P = fill_labels(P, X)
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
    """Filled-form reviews: h (document hash), page, label, bad, hires (full-resolution label)."""
    cand = pd.read_csv(REVIEW / "candidates.csv")
    lab = {}
    for hires, pat in ((False, "batch_*/out.json"), (True, "recheck/batch_*/out.json")):
        for f in sorted(glob.glob(str(REVIEW / pat))):
            for k, v in json.load(open(f)).items():
                lab[k] = (v[0], int(v[1]) if len(v) > 1 and v[1] is not None else 0, hires)
    cand = cand[cand.k.isin(lab)].copy()
    cand[["label", "bad", "hires"]] = [lab[k] for k in cand.k]
    cand["h"] = cand.path.map(corpus.doc_id)                          # emb ids change between embedding runs; the hash does not
    return cand[["k", "h", "page", "lang", "label", "bad", "hires"]]


def text_scores(X, texts):
    """Cosine of every page embedding to SigLIP 2 text embeddings of `texts` (pages x texts)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok, m = AutoTokenizer.from_pretrained(MODEL), AutoModel.from_pretrained(MODEL).eval()
    with torch.inference_mode():
        o = m.get_text_features(**tok(list(texts), padding="max_length", max_length=64, truncation=True, return_tensors="pt"))
        T = (o.pooler_output if hasattr(o, "pooler_output") else o).numpy()
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    out = []
    for i in range(0, len(X), 50000):
        x = X[i:i + 50000, :POOLED]                                       # the pooled half is the one the text tower is aligned with
        out.append(x / np.linalg.norm(x, axis=1, keepdims=True) @ T.T)
    return np.concatenate(out)


def text_types(X):
    """Zero-shot type of every page (the closest text prompt of curate.TYPES) and the picture score (curate.PIC_TEXTS)."""
    sim = text_scores(X, [C.PROMPT.format(x) for x in C.TYPES] + list(C.PIC_POS) + list(C.PIC_NEG))
    t, n = len(C.TYPES), len(C.PIC_POS)
    pic = sim[:, t:t + n].max(1) - sim[:, t + n:].max(1)
    return np.array(C.TYPES)[sim[:, :t].argmax(1)], sim[:, :t].max(1), pic


def cmd_score(_):
    from sklearn.linear_model import LogisticRegression
    P = pd.read_parquet(OUT / "pool.parquet")
    X = np.load(OUT / "X.npy").astype(np.float32)
    sure = (P.p > 0.8).values                                        # the labels of doc_types (documents of the old embedding) that are sure
    isf = P.label.isin(["forms_blank", "forms_filled"]).values
    rng0 = np.random.default_rng(0)
    pos, neg = np.where(sure & isf)[0], np.where(sure & ~isf)[0]
    tr = np.r_[rng0.choice(pos, min(20000, len(pos)), replace=False), rng0.choice(neg, min(20000, len(neg)), replace=False)]
    form = LogisticRegression(C=2, max_iter=1000, class_weight="balanced").fit(X[tr], isf[tr])
    P["p_form"] = form.predict_proba(X)[:, 1]
    R = reviews()
    pos = dict(zip(zip(P.h, P.page), range(len(P))))
    R["i"] = [pos.get(k, -1) for k in zip(R.h, R.page)]
    R = R[R.i >= 0]
    rng = np.random.default_rng(0)
    other = P.index[(~P.label.isin(["forms_blank", "forms_filled", "handwritten"])) & (P.p > 0.8)].values
    blank = P.index[(P.label == "forms_blank") & (P.p > 0.8) & (P.folder == "forms")].values
    neg = np.r_[rng.choice(other, min(6000, len(other)), replace=False), rng.choice(blank, min(3000, len(blank)), replace=False)]
    Xt = np.vstack([X[R.i.values], X[neg]])
    yt = np.r_[R.label.isin(REAL).values, np.zeros(len(neg), bool)]
    P["p_fill"] = LogisticRegression(C=4, max_iter=2000, class_weight="balanced").fit(Xt, yt).predict_proba(X)[:, 1]
    key = pd.Series(list(zip(P.h, P.page)), index=P.index)
    real = R[R.hires & R.label.isin(REAL) & (R.bad == 0)]                       # only full-resolution checks count as real
    P["reviewed"] = key.isin(set(zip(R.h, R.page)))
    P["real_filled"] = key.isin(set(zip(real.h, real.page)))
    P["tax"], P["tax_sim"], P["pic"] = text_types(X)
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


def cleaned():
    """(warped pages {(h, page)}, warped documents {h}, checked clean pages {(h, page)}) from the cleaning trace (tools/cleaned.py trace)."""
    t = pd.read_csv(INFO / "cleaned_trace.tsv", sep="\t", dtype=str)
    w = t[(t.verdict == "warped") & t.page.notna()]
    wd = set(t.doc[(t.verdict == "warped_doc") | (t.verdict == "warped") & t.page.isna() | t.source.str.startswith("distorted_moved") & (t.extra == "whole")])
    parts = t[t.source.str.startswith("distorted_moved") & (t.extra == "part")]
    pages = set(zip(w.doc, w.page.astype(float).astype(int)))
    for h, pg in zip(parts.doc, parts.page):
        pages |= {(h, int(x)) for x in str(pg).split(",") if x.strip()}
    c = t[t.verdict == "clean"]
    return pages, wd, set(zip(c.doc, c.page.astype(float).astype(int)))


def pool_for_plan():
    """The pool pages a set may use: curate.keep_pages, page ratio / resolution / pictures (geom.parquet), near duplicates out
    (the digital, live, central page of a group kept)."""
    P = pd.read_parquet(OUT / "pool.parquet")
    g = pd.read_parquet(OUT / "geom.parquet").drop_duplicates(["h", "page"])
    P = P.merge(g, on=["h", "page"], how="left")
    wp, wd, ok = cleaned()                                            # warped pages / documents out; checked clean scans and the certificates /
    key = list(zip(P.h, P.page))                                      # flyers (cleaned whole) no longer need a warp check
    out = P.h.isin(wd) | pd.Series([k in wp for k in key], index=P.index)
    P = P[~out]                                                       # the index stays the row of the pool (X.npy)
    key = list(zip(P.h, P.page))
    P["checked"] = [k in ok for k in key] | P.folder.isin(CLEAN_FOLDERS).values
    P["scan"] = P.scan & ~P.checked
    P["page_ok"] = C.page_ok(P.ratio, P.native) & (P.pic <= np.percentile(P.pic, 100 * (1 - C.PIC_OUT)))
    X = np.load(OUT / "X.npy", mmap_mode="r")[P.index.values].astype(np.float32)
    prefer = np.lexsort((P.d1.values, P.scope.values != "live", P.scan.values))
    P["dup"] = C.near_duplicates(X, P.c1.values, prefer=prefer)
    print("pages", len(P), " out: ratio / resolution / pictures", int((~P.page_ok).sum()), " near duplicates", int(P.dup.sum()))
    return C.keep_pages(P)


def lang_floor(P, w, db, floor):
    """Languages with fewer than `floor` expected pages in the set: pages of the pool to add (digital first, one per document, spread over
    the clusters), what is still missing (to download); blank digital forms and handwriting by language to the same floor.
    -> (pages to add, per-language table, blank forms to add)."""
    uni = pd.read_sql("select lang, count(*) docs, sum(cls = 'text') digital from plan group by lang", db).set_index("lang")
    uni = uni[uni.index.str.len().isin([2, 3])]
    exp = w.groupby(P.lang).sum().reindex(uni.index, fill_value=0)
    T = uni.assign(expected=exp.round(1), pool=P.groupby("lang").size().reindex(uni.index, fill_value=0))
    add = []
    for l in T.index[np.ceil(exp) < floor]:
        s = P[P.lang == l].drop_duplicates("doc")
        s = s.assign(r=s.groupby("c1").cumcount()).sort_values(["scan", "r"])
        add.append(s.head(int(floor - np.ceil(exp[l]))).assign(why="lang_floor"))
    add = pd.concat(add) if add else P.head(0)
    T["add_pages"] = add.groupby("lang").size().reindex(T.index, fill_value=0)
    T["download_pages"] = (floor - np.ceil(T.expected) - T.add_pages).clip(lower=0).astype(int)
    blank = P[(P.p_form > C.FORM_P) & (P.p_fill < C.FILL_LO) & ~P.scan]
    T["blank_forms_pool"] = blank.groupby("lang").size().reindex(T.index, fill_value=0)
    return add, T, blank


def cmd_plan(a):
    B, tag = a.size * 1000, f"{a.size}k"
    floor = a.floor if a.floor is not None else round(1000 * a.size / 600)
    P = pool_for_plan()
    sup = C.supply(P)
    q1 = C.allocate(sup, B)
    scans = C.scans_needed(sup, q1)
    w = C.page_weight(P, sup, q1)
    f = C.forms_plan(P, sup, q1)
    db = sqlite3.connect(f"file:{PATHS.manifest}?mode=ro", uri=True)
    keep = C.keep_rates(docs_table().dropna(subset=["label"]))
    R = reviews()
    dig = R[R.hires & ~R.h.isin(P[P.scan].h)]
    keep_filled = dig.label.isin(REAL).mean() * 0.9 if len(dig) else 0.2      # digital filled-looking pages that are really filled
    ladd, LT, F = lang_floor(P, w, db, a.lang_floor)
    # 1) blank digital forms to fill (+ every language up to the floor)
    fill = spread(F, "c1", f.fill)[["doc", "page", "path", "lang", "c1"]].assign(why="coverage")
    have = fill.lang.value_counts().reindex(LT.index, fill_value=0)
    more = [spread(F[F.lang == l], "c1", pd.Series({c: 1 for c in F[F.lang == l].c1.unique()})).head(int(a.lang_floor - have[l]))[["doc", "page", "path", "lang", "c1"]]
            for l in LT.index[have < a.lang_floor] if (F.lang == l).any()]
    fill = pd.concat([fill] + [m.assign(why="lang_floor") for m in more])
    fill.to_csv(OUT / f"{tag}_fill_blanks.csv", index=False)
    LT["fill_blank_forms"] = fill.lang.value_counts().reindex(LT.index, fill_value=0)
    LT["fill_forms_download"] = (a.lang_floor - LT.fill_blank_forms).clip(lower=0)
    # 2) scans to check for warping by hand: coverage gaps x margin (take = False: the reserve), real filled scans
    S = P[P.scan]
    gap = spread(S, "c1", np.ceil(a.margin * scans))
    gap = gap.assign(why="coverage_gap", take=gap.groupby("c1").cumcount() < gap.c1.map(scans))
    rf = spread(S[S.real_filled], "c1", f.real).assign(why="real_filled", take=True)
    cols = ["why", "take", "doc", "page", "path", "h", "lang", "c1", "label"]
    lf = ladd[ladd.scan].assign(take=True)
    pd.concat([gap, rf, lf])[cols].to_csv(OUT / f"{tag}_check_scans.csv", index=False)
    # 2b) filled-looking candidates to check for filling (scans: also for warping), / the share of them that was filled in the reviews
    yd, ys = (R[R.hires & (R.h.isin(P[P.scan].h) == s)].label.isin(REAL).mean() for s in (False, True))
    cand = P[(P.p_form > C.FORM_P) & (P.p_fill > C.FILL_HI) & ~P.reviewed].sort_values("p_fill", ascending=False)
    n = np.where(cand.scan, np.ceil(cand.c1.map(f.cand_scan_use).fillna(0) / max(ys, 0.1)), np.ceil(cand.c1.map(f.cand_dig_use).fillna(0) / max(yd, 0.1)))
    cand = cand[cand.groupby(["c1", "scan"]).cumcount().values < n].assign(why="filled_candidate", take=True)
    cand[cols].to_csv(OUT / f"{tag}_check_filled.csv", index=False)
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
                   pd.DataFrame({"part": "type_gap", "need": tg.download, "keep": keep.get("plain", 0.3), "doc_types": tg.index}, index=tg.index),
                   pd.DataFrame({"part": "lang_gap", "need": LT.download_pages.values, "keep": keep.get("plain", 0.3), "langs": LT.index}, index="lang_" + LT.index)[(LT.download_pages > 0).values],
                   pd.DataFrame({"part": "lang_filled_form", "need": LT.fill_forms_download.values, "keep": keep.get("forms_blank", 0.3), "langs": LT.index,
                                 "doc_types": "blank form"}, index="form_" + LT.index)[(LT.fill_forms_download > 0).values]])
    M["keep"] = M.keep.fillna(keep.get("plain", 0.3))
    M["download"] = C.inflate(M.need.fillna(0), M.keep)
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
    gen_ok = (LT.digital > 0) & ~LT.index.isin(["ur", "ps", "ku", "ckb", "sd"])      # a usable text layer to write from
    share = (w.groupby(P.lang).sum() / w.sum()).reindex(LT.index, fill_value=0)
    LT["hw_generate"] = np.where(gen_ok, np.maximum(a.lang_floor, np.ceil(H.need.sum() * share)), 0).astype(int)
    LT.sort_values("expected").to_csv(OUT / f"{tag}_lang_floor.csv", index_label="lang")
    HL = pd.DataFrame([dict(type=t, lang=l, collect=round(r.collect * p)) for t, r in H.iterrows() for l, p in L.items()])
    HL[HL.collect > 0].to_csv(OUT / f"{tag}_handwritten_by_lang.csv", index=False)
    print(f"{tag}: scans to check {int(scans.sum())} x {a.margin} = {len(gap)} (+ real filled {len(rf)}); filled candidates to check"
          f" {len(cand)} ({int(cand.scan.sum())} scans; filled share in the reviews: digital {yd:.2f}, scans {ys:.2f})")
    print(f"  filled forms {int(f.need.sum())} = real {int(f.real.sum())} + fill {int(f.fill.sum())} + digital candidates"
          f" {int(f.cand_dig_use.sum())} + scan candidates {int(f.cand_scan_use.sum())} + download {int(f.download.sum())}")
    print(f"  downloads: {M.groupby('part').need.sum().astype(int).to_dict()} kept pages -> {M.groupby('part').download.sum().to_dict()} documents;"
          f" handwriting need {int(H.need.sum())}, collect {int(H.collect.sum())} -> {OUT}")


def resolve(path):
    """The file of a corpus path; moved / renamed since the embedding: found by its hash (a pdf first); None."""
    p = corpus.resolve(path)
    if p.exists():
        return p
    alts = sorted(HASHES.get(corpus.doc_id(path), []), key=lambda x: not x.lower().endswith(".pdf"))
    return corpus.resolve(alts[0]) if alts else None


def native_px(pg, W, H):
    """Long side px of a pdf page's raster: the picture covering most of the page scaled to it; inf without one (vector page)."""
    best = np.inf
    for i in pg.get_image_info():
        bw, bh = i["bbox"][2] - i["bbox"][0], i["bbox"][3] - i["bbox"][1]
        if bw * bh > 0.3 * W * H:
            best = min(best, max(i["width"] * W / bw, i["height"] * H / bh))
    return best


def load_page(path, page, long):
    """BGR image of the page with its long side `long` px, or None."""
    import cv2
    import pymupdf
    p = resolve(path)
    if p is None:
        return None
    if str(p).lower().endswith(".pdf"):
        pymupdf.TOOLS.mupdf_display_errors(False)
        d = pymupdf.open(p)
        pg = d[min(int(page), len(d) - 1)]
        s = long / max(pg.rect.width, pg.rect.height)
        pix = pg.get_pixmap(matrix=pymupdf.Matrix(s, s), colorspace=pymupdf.csRGB)
        return np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, 3)[:, :, ::-1]
    im = cv2.imread(str(p))
    if im is None:
        return None
    s = long / max(im.shape[:2])
    return cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def render(job):
    """Full-resolution page, cropped to the page ratio (curate.crop_box)."""
    k, path, page, out = job
    import cv2
    try:
        im = load_page(path, page, LONG)
        if im is None:
            return k, False
        b = C.crop_box(im.shape[1], im.shape[0])
        if b:
            im = im[b[1]:b[3], b[0]:b[2]]
        return k, cv2.imwrite(str(out), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
    except Exception:
        return k, False


def geom(job):
    """[(h, page, h / w, native px of the raster)] of the pages of one document from its metadata (no rendering)."""
    h, path, pages = job
    import pymupdf
    from PIL import Image
    nan = [(h, pg, np.nan, np.nan) for pg in pages]
    try:
        p = resolve(path)
        if p is None:
            return nan
        if str(p).lower().endswith(".pdf"):
            pymupdf.TOOLS.mupdf_display_errors(False)
            d = pymupdf.open(p)
            out = []
            for pg in pages:
                try:
                    q = d[min(int(pg), len(d) - 1)]
                    W, H = q.rect.width, q.rect.height
                    out.append((h, pg, H / W, native_px(q, W, H)))
                except Exception:
                    out.append((h, pg, np.nan, np.nan))
            return out
        w, hh = Image.open(p).size
        return [(h, pg, hh / w, max(w, hh)) for pg in pages]
    except Exception:
        return nan


def cmd_geom(_):
    """Page ratio and native resolution of every pool page -> OUT/geom.parquet (keys h, page; resumable, one job per document)."""
    P = pd.read_parquet(OUT / "pool.parquet", columns=["h", "path", "page"])
    f = OUT / "geom.parquet"
    done = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["h", "page", "ratio", "native"])
    seen = set(zip(done.h, done.page))
    P = P[[k not in seen for k in zip(P.h, P.page)]]
    todo = [(h, g.path.iloc[0], list(g.page)) for h, g in P.groupby("h")]
    out = []
    with ProcessPoolExecutor(WORKERS, initializer=_init, initargs=(hash_index(),)) as ex:
        for n, r in enumerate(ex.map(geom, todo, chunksize=16)):
            out += r
            if len(out) >= 50000 or n == len(todo) - 1:
                done = pd.concat([done, pd.DataFrame(out, columns=["h", "page", "ratio", "native"])])
                done.to_parquet(f)
                out = []
                print(len(done), "pages;", n + 1, "of", len(todo), "documents", flush=True)


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


def entropy(s):
    p = s.value_counts(normalize=True)
    return float(-(p * np.log(p)).sum())


def cmd_sample(a):
    """1K-page test sets by different selection methods from the filtered pool (pool_for_plan) -> archive/ds<N>_<method>/ (jpgs +
    index.csv: k, doc, page, path, lang, label, scan, scope) and OUT/sample_metrics.csv: coverage of the pool (cosine distance to the
    nearest sample: mean / p99 / max, share within 0.3), spacing between samples, clusters, label and language entropy."""
    P = pool_for_plan().reset_index(drop=True)
    full = pd.read_parquet(OUT / "pool.parquet", columns=["doc", "page"])
    key = dict(zip(zip(full.doc, full.page), range(len(full))))
    X0 = np.load(OUT / "X.npy", mmap_mode="r")
    X = np.asarray(X0[[key[k] for k in zip(P.doc, P.page)]], np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    K, rng = a.n, np.random.default_rng(0)
    ref = rng.choice(len(X), 30000, replace=False)
    dens = C.density(X, X[ref])
    live = np.where(P.scope.values == "live")[0]
    core = np.where(dens >= np.percentile(dens, 25))[0]
    sets = {
        "random": lambda: rng.choice(len(X), K, replace=False),
        "fps": lambda: C.farthest(X, K, np.array([], int), np.arange(len(X))),
        "medoids": lambda: C.medoids(X, K),
        "hybrid": lambda: C.select_hybrid(X, dens, K),
        "hybrid_lang": lambda: C.select_by_group(X, dens, P.lang.values, K),
    }
    rows = []
    for name, f in sets.items():
        sel = np.asarray(f())
        if len(sel) < K:                                              # duplicates of the medoid step: topped up with the boundary
            sel = np.concatenate([sel, C.farthest(X, K - len(sel), sel, core)])
        sel = sel[:K]
        n = C.nearest(X, X[sel])
        G = 1 - X[sel] @ X[sel].T
        np.fill_diagonal(G, 9)
        sp = G.min(1)
        s = P.iloc[sel]
        rows.append(dict(method=name, n=len(sel), cov_mean=n.mean(), cov_p99=np.percentile(n, 99), cov_max=n.max(), within_03=(n < .3).mean(),
                         spacing_p5=np.percentile(sp, 5), spacing_med=np.median(sp), spacing_cv=sp.std() / sp.mean(),
                         clusters=s.c1.nunique(), H_label=entropy(s.label), H_lang=entropy(s.lang), langs=s.lang.nunique(),
                         scan=s.scan.mean(), rejected=(s.scope == "rej").mean()))
        out = Path(PATHS.layouts) / "archive" / f"ds{K // 1000}k_{name}"
        out.mkdir(parents=True, exist_ok=True)
        df = s[["doc", "page", "path", "lang", "label", "scan", "scope", "c1"]].copy()
        df.insert(0, "k", [f"{i:04d}_{corpus.doc_id(p)}_p{g}" for i, (p, g) in enumerate(zip(df.path, df.page))])
        df["dist_nearest_sample"] = np.round(sp, 3)
        df.to_csv(OUT / f"sample_{K // 1000}k_{name}.csv", index=False)
        if not a.no_render:
            jobs = [(r.k, r.path, r.page, out / f"{r.k}.jpg") for r in df.itertuples()]
            with ProcessPoolExecutor(WORKERS, initializer=_init, initargs=(hash_index(),)) as ex:
                ok = dict(ex.map(render, jobs, chunksize=8))
            df["rendered"] = df.k.map(ok)
            df.to_csv(out / "index.csv", index=False)
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"sample_{K // 1000}k_metrics.csv", index=False)


PERSONS = ["school_cursive", "round_print", "ruled_neat", "rounded_neat", "soft_upright", "compact_print", "print_script", "backhand"]   # hand.labelled
PENS = ["ballpoint", "gel", "fineliner", "fountain_fine", "ballpoint_fine"]
STYLES = ["casual", "accurate", "educated", "professional"]
HW_LAYOUT = {"diary page": (("plain", "formatted", "poems"), ("lined", "margin", "blank")),
             "graph paper page": (("tables", "scientific", "formatted"), ("squared",)),
             "handwritten letter": (("plain", "bureaucracy"), ("blank", "lined")),
             "handwritten notes": (("formatted", "plain", "scientific"), ("lined", "dotted", "blank")),
             "lined notebook page": (("plain", "formatted", "scientific"), ("lined", "margin"))}


def cmd_hwlists(a):
    """Handwriting to make, as lists of jobs: OUT/<N>k_hw_generate.csv (per language, handwriting type, source layout = a digital page of the
    language whose structure is written by hand, paper, writer type, pen, style) and OUT/<N>k_fill_forms_hw.csv (blank digital forms to fill:
    writer type, pen, style). Counts: <N>k_lang_floor.csv hw_generate per language, split over the types by the plan's need (<N>k_handwritten.csv)."""
    tag = f"{a.size}k"
    LT = pd.read_csv(OUT / f"{tag}_lang_floor.csv").set_index("lang")
    H = pd.read_csv(OUT / f"{tag}_handwritten.csv").set_index("type")
    H = H[H.index.isin(HW_LAYOUT) & (H.need > 0)]
    P = pool_for_plan()
    P = P[~P.scan & P.label.isin(["plain", "formatted", "poems", "scientific", "tables", "bureaucracy"])]
    rng = np.random.default_rng(0)
    rows, k = [], 0
    for lang, n in LT.hw_generate[LT.hw_generate > 0].items():
        q = (H.need / H.need.sum() * n)                                    # largest remainder split over the types
        cnt = np.floor(q).astype(int)
        cnt[(q - cnt).sort_values(ascending=False).index[:n - cnt.sum()]] += 1
        S = P[P.lang == lang]
        for t, c in cnt.items():
            labels, papers = HW_LAYOUT[t]
            src = S[S.label.isin(labels)]
            src = src if len(src) else S                                    # no page of the wanted layout in this language
            src = src.drop_duplicates("doc")
            src = src.assign(r=src.groupby("c1").cumcount()).sort_values("r").sample(frac=1, random_state=k) if len(src) else src
            for i in range(c):
                s = src.iloc[i % len(src)] if len(src) else None
                rows.append(dict(lang=lang, hw_type=t, layout=None if s is None else s.label, paper=papers[i % len(papers)],
                                 persona=PERSONS[k % len(PERSONS)], pen=PENS[(k // 2) % len(PENS)], style=STYLES[rng.integers(len(STYLES))],
                                 source_doc=None if s is None else s.doc, source_page=None if s is None else s.page,
                                 source_path=None if s is None else s.path, source_reused=bool(len(src)) and i >= len(src)))
                k += 1
    G = pd.DataFrame(rows)
    G.to_csv(OUT / f"{tag}_hw_generate.csv", index=False)
    F = pd.read_csv(OUT / f"{tag}_fill_blanks.csv")
    F["persona"] = [PERSONS[i % len(PERSONS)] for i in range(len(F))]
    F["pen"] = [PENS[(i // 3) % len(PENS)] for i in range(len(F))]
    F["style"] = rng.choice(STYLES, len(F))
    F.to_csv(OUT / f"{tag}_fill_forms_hw.csv", index=False)
    print(len(G), "handwritten pages to generate in", G.lang.nunique(), "languages;", G.hw_type.value_counts().to_dict(),
          "\n  layouts", G.layout.value_counts(dropna=False).to_dict(), "\n  without a source page", int(G.source_doc.isna().sum()),
          " sources reused", int(G.source_reused.sum()), "\n ", len(F), "forms to fill in", F.lang.nunique(), "languages ->", OUT)


def cmd_export(a):
    """The pages of a plan's check lists as full-resolution jpgs: archive/<N>k_warp_check (scans: coverage gaps with the reserve,
    real filled scans) and archive/<N>k_filled_check (filled-looking candidates); file <n>_<hash>_p<page>.jpg, index.csv."""
    tag = f"{a.size}k"
    hashes = hash_index()
    lists = ((f"{tag}_check_scans.csv", f"{tag}_warp_check"), (f"{tag}_check_filled.csv", f"{tag}_filled_check"))
    for src, name in lists[:1] if a.scans_only else lists:
        df = pd.read_csv(OUT / src)
        out = Path(PATHS.layouts) / "archive" / name
        out.mkdir(parents=True, exist_ok=True)
        df["file"] = [f"{i:05d}_{h}_p{p}.jpg" for i, (h, p) in enumerate(zip(df.h, df.page))]
        jobs = [(r.file, r.path, r.page, out / r.file) for r in df.itertuples()]
        with ProcessPoolExecutor(WORKERS, initializer=_init, initargs=(hashes,)) as ex:
            ok = dict(ex.map(render, jobs, chunksize=8))
        df["rendered"] = df.file.map(ok)
        df.to_csv(out / "index.csv", index=False)
        print(name, len(df), "pages", int(df.rendered.sum()), "rendered ->", out)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("pool")
    sp.add_parser("geom")
    hl = sp.add_parser("hwlists")
    hl.add_argument("--size", type=int, required=True)
    sp.add_parser("score")
    q = sp.add_parser("sample")
    q.add_argument("--n", type=int, default=1000)
    q.add_argument("--no-render", action="store_true")
    p = sp.add_parser("plan")
    p.add_argument("--size", type=int, required=True, help="set size in thousands of pages")
    p.add_argument("--floor", type=int, help="pages of every document type (default 1000 per 600k)")
    p.add_argument("--hw-x", type=float, default=5.0, help="handwriting to collect per page needed (dirty sources)")
    p.add_argument("--margin", type=float, default=1.5, help="scans to check per scan needed (warped ones are dropped; "
                   "the last warp cleaning dropped ~11 %% of the scanned form documents and 43 %% of the pages of partly warped ones)")
    p.add_argument("--lang-floor", type=int, default=10, help="every language gets at least this many pages: in the set, blank forms to fill, handwriting to generate")
    e = sp.add_parser("export")
    e.add_argument("--size", type=int, required=True)
    e.add_argument("--scans-only", action="store_true", help="only the scans to check for warping")
    r = sp.add_parser("render")
    r.add_argument("csv")
    r.add_argument("out")
    r.add_argument("--per", type=int, default=50)
    a = ap.parse_args()
    {"pool": cmd_pool, "geom": cmd_geom, "score": cmd_score, "sample": cmd_sample, "hwlists": cmd_hwlists, "plan": cmd_plan, "export": cmd_export, "render": cmd_render}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
