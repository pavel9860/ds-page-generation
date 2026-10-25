"""Selection of a diverse page set from the corpus by coverage of the page-embedding space (tools/diverse_ds.py).

Pages are the embedded pages 0 / 1 of every corpus document (SigLIP 2, emb_out). Two-level k-means (2000 fine clusters, 200 groups);
a budget is dealt out over the groups and then over their clusters in proportion to sqrt(supply), capped by the supply, the excess
dealt out again (water filling). A cluster takes its digital pages first, then deeper pages of its digital documents, then scans
(scans need a manual warp check). Forms: half of the quota of a form cluster is filled pages: checked real filled pages, then blank
digital forms to fill, then unchecked filled-looking candidates, the rest is downloaded. Document types (zero-shot text prompts) get
a floor; what the corpus lacks is downloaded, handwriting is scanned or generated. Everything here is pure (frames in, frames out).
"""
import numpy as np
import pandas as pd

OUT_LABELS = ("images", "pages_photo", "recieps")          # never in the set: pictures, photographed pages, receipts
OUT_FOLDERS = ("pages_photo", "receipts")
OUT_TYPES = frozenset({"postcard", "greeting card", "envelope", "receipt", "coupon", "lottery ticket", "ticket", "boarding pass",
                       "shipping label", "label sticker", "business card",                         # small formats: out of scope
                       "blank page", "photocopy", "fax", "dot matrix printout", "old typewritten document", "old printed book page",
                       "page with stamps and signatures"})                                         # styles, not document types
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
    """The pages a set may use: no pictures, photographed pages, receipts, out-of-scope types."""
    m = ~P.label.isin(OUT_LABELS) & ~P.folder.isin(OUT_FOLDERS)
    if "tax" in P:
        m &= ~P.tax.isin(OUT_TYPES)
    return P[m]


def supply(P):
    """Per fine cluster: n pages, digital pages, deeper pages of its digital documents (pages 2..n; a document has rows for pages
    0 and 1, each row carries half), group."""
    g = P.groupby("c1")
    deep = (P.pages.clip(1, 500) - 2).clip(lower=0).where(~P.scan, 0) / 2
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
    docs: one row per document with top (crawl folder), scan, label."""
    out = {}
    for folder, labels in FOLDER_TYPE.items():
        s = docs[docs.top == folder]
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
