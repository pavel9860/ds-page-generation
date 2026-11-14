"""Selection of a diverse page set from the corpus by coverage of the page-embedding space (tools/diverse_ds.py).

Pages are the embedded pages 0 / 1 of every corpus document (SigLIP 2, emb_out). Two-level k-means (2000 fine clusters, 200 groups);
a budget is dealt out over the groups and then over their clusters in proportion to sqrt(supply), capped by the supply, the excess
dealt out again (water filling). A cluster takes its digital pages first, then deeper pages of its digital documents, then scans
(scans need a manual warp check). Forms: half of the quota of a form cluster is filled pages: checked real filled pages, then blank
digital forms to fill, then unchecked filled-looking candidates, the rest is downloaded. Document types (zero-shot text prompts) get
a floor; what the corpus lacks is downloaded, handwriting is scanned or generated. Out of every set: pictures, photographed pages,
receipts, maps, picture-dominated pages, pages that do not crop to the A4 ratio, coarse rasters, near duplicates and pages of one
template. Everything here is pure (frames in, frames out).
"""
import numpy as np
import pandas as pd

OUT_LABELS = ("images", "pages_photo", "recieps", "maps", "blueprints")   # never in the set: pictures, photographed pages, receipts,
                                                                          # maps and blueprints (out of scope)
OUT_FOLDERS = ("pages_photo", "receipts", "maps", "blueprints", "maps_blueprints")
OUT_TYPES = frozenset({"postcard", "greeting card", "envelope", "receipt", "coupon", "lottery ticket", "ticket", "boarding pass",
                       "shipping label", "label sticker", "business card",                         # small formats: out of scope
                       "blank page", "photocopy", "fax", "dot matrix printout", "old typewritten document", "old printed book page",
                       "page with stamps and signatures",
                       "map", "floor plan", "engineering drawing", "electrical schematic"})                         # maps, drawings                                         # styles, not document types
HW_TYPES = ("handwritten letter", "handwritten notes", "homework in a notebook", "diary page", "lined notebook page",
            "graph paper page", "recipe", "filled form with handwriting")
PROMPT = "a scan of a {}"
TYPES = (
    "letter", "business letter with letterhead", "email printout", "invoice", "receipt", "bank statement", "payslip", "utility bill",
    "purchase order", "price list", "contract", "legal document with numbered clauses", "court document",
    "official government decree", "certificate", "diploma", "ID card application form", "passport application form", "tax form",
    "filled form with handwriting", "blank form with empty fields", "questionnaire", "checklist", "medical prescription",
    "medical report", "lab test results", "patient information leaflet", "product manual with diagrams", "warranty card",
    "technical datasheet", "safety data sheet", "spreadsheet printout", "financial report with tables", "timetable", "calendar",
    "school timetable", "school worksheet", "exam paper", "answer sheet with checkboxes", "homework in a notebook",
    "handwritten letter", "handwritten notes", "lined notebook page", "graph paper page", "diary page", "recipe", "restaurant menu",
    "flyer", "poster", "brochure", "newsletter", "newspaper page", "magazine page", "comic page", "children's book page",
    "novel page", "poetry book page", "textbook page", "dictionary page", "encyclopedia page", "bible page", "scientific article",
    "thesis page", "slide printout", "presentation handout", "table of contents", "index page", "bibliography", "sheet music",
    "song lyrics with chords", "crossword puzzle", "sudoku", "map", "floor plan", "engineering drawing", "electrical schematic",
    "chart and graph page", "infographic", "organizational chart", "ticket", "boarding pass", "envelope", "postcard",
    "greeting card", "business card", "label sticker", "form with barcode", "packing list", "shipping label", "delivery note",
    "meeting minutes", "agenda", "memo", "resume", "cover letter", "report cover page", "title page", "blank page",
    "page with stamps and signatures", "old typewritten document", "old printed book page", "dot matrix printout", "fax",
    "photocopy", "church bulletin", "event program", "certificate of award", "lottery ticket", "coupon", "catalog page",
    "real estate listing", "advertisement page")
# crawl folder -> the page type its documents are meant to be (keep rates: share of a folder's documents that are digital and that type)
FOLDER_TYPE = {"forms": ("forms_blank",), "filled_forms": ("forms_filled",), "tables": ("tables",), "bureaucracy": ("bureaucracy",),
               "sheet_music": ("music",), "maps": ("maps",), "blueprints": ("blueprints",), "magazines": ("magazines",),
               "flyers": ("flyers",), "certificates": ("certificates",), "text": ("plain", "formatted", "poems")}
TEXT_LABELS = ("plain", "formatted", "scientific", "poems")
FORM_P, FILL_LO, FILL_HI = 0.5, 0.3, 0.5                # p_form of a form page; p_fill below: blank, above: filled-looking
MIN_FORMS = 10                                          # a cluster with fewer form pages is not a form cluster


RATIO = 2 ** 0.5                                        # page h / w of the set (A4); other pages are cropped to it
MAX_CROP = 0.15                                         # at most this share of a side is cropped, else the page is out
MIN_NATIVE = 1000                                       # long side px of a raster page under this: too coarse
PIC_OUT = 0.10                                          # the pages with the highest picture score (this share of the pool) are out
PIC_POS = ("a page dominated by a large photograph", "a page with big colourful illustrations", "a poster with a large picture",
           "a page with a large chart or diagram", "a page with a grid of photographs", "a comic page")          # picture score = max cosine to these - max cosine to PIC_NEG
PIC_NEG = ("a page of printed text", "a scanned document page with text", "a page of handwritten text", "a form with fields",
           "a page with a table")
DUP_SIM = 0.95                                          # cosine above this (1536-d page vectors: pooled | mid-layer patch mean): the same page or template / series
                                                        # office, a listing split into files); pages of one cluster: ~0.73


def crop_box(w, h, ratio=RATIO, max_crop=MAX_CROP):
    """(x0, y0, x1, y1) of the centred crop of a w x h page to h / w = ratio, None when it is the ratio; False when the crop would
    cut more than max_crop of a side (landscape pages, strips)."""
    r = h / w
    if abs(r / ratio - 1) < 0.01:
        return None
    if r > ratio:
        nh = round(w * ratio)
        return (0, (h - nh) // 2, w, (h - nh) // 2 + nh) if 1 - nh / h <= max_crop else False
    nw = round(h / ratio)
    return ((w - nw) // 2, 0, (w - nw) // 2 + nw, h) if 1 - nw / w <= max_crop else False


def page_ok(ratio, native):
    """A page the set may use by its geometry: croppable to the page ratio, raster fine enough."""
    r = np.asarray(ratio, float)
    lo, hi = RATIO * (1 - MAX_CROP), RATIO / (1 - MAX_CROP)
    return (r >= lo) & (r <= hi) & (np.asarray(native, float) >= MIN_NATIVE)


def near_duplicates(X, c1, sim=DUP_SIM, prefer=None, block=2048):
    """Boolean mask of the pages to drop: within each cluster, a page whose embedding is within `sim` of an earlier kept page
    (pages taken in the order of `prefer`, best first: digital, live). Similarities by blocks of rows (one matmul per block)."""
    order = np.arange(len(c1)) if prefer is None else np.asarray(prefer)
    drop = np.zeros(len(c1), bool)
    c1 = np.asarray(c1)
    oc = c1[order]
    srt = np.argsort(oc, kind="stable")                                  # pages of a cluster together, in the order of prefer
    order, oc = order[srt], oc[srt]
    cuts = np.r_[0, np.flatnonzero(np.diff(oc)) + 1, len(oc)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        idx = order[a:b]
        Z = X[idx]
        alive = np.ones(len(idx), bool)
        for s in range(0, len(idx), block):
            S = (Z[s:s + block] @ Z[s:].T) > sim                           # rows s.., columns s..
            for j in range(min(block, len(idx) - s)):
                if alive[s + j]:
                    alive[s + j + 1:] &= ~S[j, j + 1:]
        drop[idx[~alive]] = True
    return drop


def water(supply, budget, weight):
    """Allocation per index of `supply` (Series): `budget` dealt in proportion to `weight`, capped by the supply, the excess dealt
    out again over the uncapped ones. Sum = min(budget, supply.sum())."""
    sup, w = supply.astype(float), weight.astype(float)
    a, left, act = sup * 0, float(budget), (sup > 0) & (w > 0)
    while left > 0.5 and act.any():
        q = left * w[act] / w[act].sum()
        add = np.minimum(q, sup[act] - a[act])
        a[act] += add
        left -= add.sum()
        act &= a < sup - 1e-9
    return a


def keep_pages(P):
    """The pages a set may use: no pictures, photographed pages, receipts, maps, out-of-scope types; with geometry columns also
    croppable to the page ratio, fine raster, no big pictures, not a near duplicate."""
    m = ~P.label.isin(OUT_LABELS) & ~P.folder.isin(OUT_FOLDERS)
    if "tax" in P:
        m &= ~P.tax.isin(OUT_TYPES)
    if "page_ok" in P:
        m &= P.page_ok & ~P.dup
    return P[m]


def supply(P):
    """Per fine cluster: n pages, digital pages, deep (0: every page of a document is a row of the pool), group."""
    g = P.groupby("c1")
    deep = P.pages * 0                                                    # every page is its own embedded row: no estimate of pages not embedded
    return pd.DataFrame({"n": g.size(), "dig": (~P.scan).groupby(P.c1).sum(), "deep": deep.groupby(P.c1).sum(), "c2": g.c2.first()})


def allocate(sup, budget, alpha=0.5):
    """Quota per fine cluster: water filling over the groups, then over the clusters of each group, weights supply ** alpha."""
    tot = sup.n + sup.deep
    t2 = tot.groupby(sup.c2).sum()
    q2 = water(t2, budget, t2 ** alpha)
    q1 = pd.concat([water(tot[sup.c2 == k], q2[k], tot[sup.c2 == k] ** alpha) for k in q2.index if q2[k] > 0])
    return q1.reindex(sup.index, fill_value=0).round()


def demand(sup, budget, alpha=0.5):
    """The quota per fine cluster without the supply cap (what coverage would ask for)."""
    tot = sup.n + sup.deep
    t2 = tot.groupby(sup.c2).sum()
    d2 = budget * t2 ** alpha / (t2 ** alpha).sum()
    w = tot ** alpha
    return d2.reindex(sup.c2).values * w / w.groupby(sup.c2).transform("sum")


def scans_needed(sup, q1):
    """Pages of a cluster's quota that only scans can give (no digital page, no deeper digital page left)."""
    rest = q1 - np.minimum(q1, sup.dig)
    return (rest - np.minimum(rest, sup.deep)).clip(lower=0)


def page_weight(P, sup, q1):
    """Expected selected pages per pool page (a cluster's quota spread over its pages)."""
    return P.c1.map(q1 / sup.n).fillna(0)


def forms_plan(P, sup, q1, filled_share=0.5):
    """Per form cluster: filled pages needed and where they come from: real (checked), fill (blank digital forms to fill),
    cand_dig / cand_scan (unchecked filled-looking pages), download (none of these left)."""
    F = P[P.p_form > FORM_P]
    blank = (F.p_fill < FILL_LO) & ~F.scan
    cand = (F.p_fill > FILL_HI) & ~F.reviewed
    f = pd.DataFrame({"forms": F.groupby("c1").size(), "real_avail": F.real_filled.groupby(F.c1).sum(),
                      "blank": blank.groupby(F.c1).sum(), "cand_dig": (cand & ~F.scan).groupby(F.c1).sum(),
                      "cand_scan": (cand & F.scan).groupby(F.c1).sum()})
    f = f[f.forms >= MIN_FORMS].join(q1.rename("q")).join(sup.n)
    f["need"] = (filled_share * f.q * f.forms / f.n).round()
    f["real"] = np.minimum(f.real_avail, f.need)
    f["fill"] = np.minimum(f.need - f.real, f.blank)
    rest = f.need - f.real - f.fill
    f["cand_dig_use"] = np.minimum(rest, f.cand_dig)
    f["cand_scan_use"] = np.minimum(rest - f.cand_dig_use, f.cand_scan)
    f["download"] = rest - f.cand_dig_use - f.cand_scan_use
    return f


def type_floor(P, w, floor):
    """Per zero-shot type: pages in the set, the floor, what to download (typed) or to scan / generate (handwriting)."""
    t = P.groupby("tax").agg(pages=("doc", "size"), scans=("scan", "sum"))
    t["in_set"] = w.groupby(P.tax).sum().round()
    short = (floor - t.in_set).clip(lower=0).round()
    hw = t.index.isin(HW_TYPES)
    t["download"] = np.where(hw, 0, short)
    t["handwritten"] = np.where(hw, short, 0)
    return t


def keep_rates(docs, dup=0.9):
    """Share of downloaded documents that end up used, per page type, measured on the corpus crawl folders:
    (1 - scan share) x share of the folder's digital documents that are the folder's type, x `dup` (duplicates, broken, language).
    docs: one row per document with folder (crawl folder), scan, label."""
    out = {}
    for folder, labels in FOLDER_TYPE.items():
        s = docs[docs.folder == folder]
        if len(s):
            r = (1 - s.scan.mean()) * s[~s.scan].label.isin(labels).mean() * dup
            for lab in labels:
                out[lab] = r
    for lab in TEXT_LABELS:
        out.setdefault(lab, out.get("plain", 0.3))
    return out


def inflate(need, keep, floor=0.15):
    """Documents to download for `need` kept pages."""
    return np.ceil(need / np.maximum(keep, floor)).astype(int)


# ---- embedding-space selection of a small set (1K pages): benchmarked in tools/diverse_ds.py bench / sample ----
def density(X, ref, r=0.2, chunk=10000):
    """Share of the `ref` embeddings within cosine distance r of every row of X (rows are L2-normalised)."""
    out = np.zeros(len(X), np.float32)
    for i in range(0, len(X), chunk):
        out[i:i + chunk] = ((X[i:i + chunk] @ ref.T) > 1 - r).mean(1)
    return out


def nearest(X, C, chunk=20000):
    """Cosine distance of every row of X to its nearest row of C."""
    out = np.empty(len(X), np.float32)
    for i in range(0, len(X), chunk):
        out[i:i + chunk] = 1 - (X[i:i + chunk] @ C.T).max(1)
    return out


def medoids(X, k, weight=None, seed=0):
    """Indices of the pages nearest to the k centres of a (weighted) spherical k-means: the bulk, k-medoid coverage."""
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(k, random_state=seed, batch_size=8192, n_init=1, max_iter=60).fit(X, sample_weight=weight)
    c = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    return np.unique(np.concatenate([(X @ c[i:i + 200].T).argmax(0) for i in range(0, len(c), 200)]))


def farthest(X, k, have, cand):
    """k indices of `cand` added one by one, each the farthest from `have` and the ones taken (k-center greedy: the boundary)."""
    Z = X[cand]
    d = nearest(Z, X[have]) if len(have) else np.full(len(Z), 2.0, np.float32)
    out = []
    for _ in range(k):
        i = int(d.argmax())
        out.append(cand[i])
        d = np.minimum(d, 1 - Z @ Z[i])
    return np.array(out, int)


def flat_weight(dens, n_ref, gamma=0.5):
    """k-means weights that flatten the density (count within r, + 1) ** -gamma; gamma 0: density-proportional, 1: uniform."""
    w = (dens * n_ref + 1.0) ** -gamma
    return w / w.mean()


def select_hybrid(X, dens, k, boundary=0.15, gamma=0.5, n_ref=30000, trim=25):
    """k pages: (1 - boundary) k density-flattened medoids (bulk coverage, no outlier picks) + boundary k farthest-point pages
    from the pages above the `trim` percentile of density (the extent of the distribution without junk). Topped up to k."""
    nb = round(boundary * k)
    core = np.where(dens >= np.percentile(dens, trim))[0]
    sel = medoids(X, k - nb, flat_weight(dens, n_ref, gamma))
    sel = np.concatenate([sel, farthest(X, k - len(sel), sel, core)])
    return sel


def select_by_group(X, dens, groups, k, alpha=0.5, boundary=0.15, **kw):
    """Hybrid selection per group (language): quota ~ pages ** alpha, at least 1, capped by the group."""
    n = pd.Series(groups).value_counts()
    q = water(n, k, n ** alpha).round().astype(int).clip(lower=1)
    sel = []
    for g, qg in q.items():
        ix = np.where(groups == g)[0]
        if qg >= len(ix):
            sel += list(ix)
        elif qg < 4:
            sel += list(ix[np.argsort(-dens[ix])[:qg]])
        else:
            sel += list(ix[select_hybrid(X[ix], dens[ix], int(qg), boundary, **kw)])
    sel = np.unique(sel)
    if len(sel) < k:
        sel = np.concatenate([sel, farthest(X, k - len(sel), sel, np.where(dens >= np.percentile(dens, 25))[0])])
    return sel
