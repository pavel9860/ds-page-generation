"""Every digital-born labelled page of pagetype_labelled (class folders of the digital page kinds) with what it needs for training:
its embedding, its PDF page for the formatting features and its language. A jpg is only a render of a page: it counts when it is
matched to its PDF page with a usable layer; scans and pictures without a PDF are left out.

  item: the file in <class>/ — a jpg (embedding from labelled_embeddings / pagekind_new_embeddings, PDF page from pagekind_src.jsonl)
        or a one-document pdf (the corpus embedding of its page 0, emb_out; the pdf page 0 itself)
  language: the manifest language of the document; a pdf named <hash>_..._<lang>.pdf gives it by name; certificates/labels.csv;
        pagekind_lang.json (languages read by agents from the page, tools/pagekind_lang.py scan / read) last

inventory() -> [Item]
"""
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pagekind_eval as V                                     # noqa: E402
from dspages.hand import corpus                               # noqa: E402

ROOT = V.UV / "pagetype_labelled"
SCAN_ONLY = ("cache", "pages_photo", "handwritten")            # classes of scans only: not digital page kinds
NO_LANG = ("maps", "blueprints", "music")                       # language is not a target of these classes
LANG_FILE = V.CACHE / "pagekind_lang.json"
LAYER = ("text", "encoding lost", "vector / painted text")     # plan.cls with a PDF layer: formatting features exist
CODE = re.compile(r"_([a-z]{2,3})$")


@dataclass
class Item:
    cls: str
    id: str                    # <class>/<file name>
    doc: str
    emb: np.ndarray | None
    pdf: str | None            # PDF with a usable layer, else None
    page: int
    lang: str | None


def classes():
    return sorted(c.name for c in ROOT.iterdir() if c.is_dir() and c.name not in SCAN_ONLY)


def _corpus_page0():
    ids = np.load(V.UV / "emb_out" / "ids.npy")
    emb = np.load(V.UV / "emb_out" / "embeddings.npy", mmap_mode="r")
    man = {r["id"]: corpus.doc_id(r["path"]) for r in map(json.loads, open(V.UV / "emb_out" / "manifest.jsonl"))}
    rows = {man[int(d)]: k for k, (d, p) in enumerate(ids) if p == 0 and int(d) in man}
    return rows, emb


def inventory():
    db = corpus.connect()
    lang = corpus.languages(db, min_conf=0.0)
    cls = {d: c for d, c in db.execute("select doc,cls from plan")}
    pe = V.page_embeddings()
    src = {r["image"]: r for r in map(json.loads, open(V.SRC)) if r["ok"]}
    read = json.load(open(LANG_FILE)) if LANG_FILE.exists() else {}
    named = {}
    f = ROOT / "certificates" / "labels.csv"
    if f.exists():
        named = {r[0]: r[3] for r in csv.reader(open(f)) if len(r) >= 4 and r[3]}
    row0, emb = _corpus_page0()
    out = []
    for c in classes():
        for f in sorted((ROOT / c).iterdir()):
            if f.suffix.lower() == ".jpg":
                r = src.get(str(f))
                d = corpus.doc_id(r["pdf"]) if r else f.stem
                pdf = r["pdf"] if r and cls.get(d) in LAYER else None
                item = Item(c, f"{c}/{f.name}", d, pe.get(f.stem), pdf, r["page"] if pdf else -1, lang.get(d) if r else None)
            elif f.suffix.lower() == ".pdf":
                d = corpus.doc_id(f.name)
                k = row0.get(d)
                m = CODE.search(f.stem)
                item = Item(c, f"{c}/{f.name}", d, np.asarray(emb[k], np.float32) if k is not None else None,
                            str(f) if cls.get(d, "text") in LAYER else None, 0, lang.get(d) or (m.group(1) if m else None))
            else:
                continue
            if item.pdf is None:                         # digital-born pages only: a jpg without its PDF page, a scan, is not one
                continue
            item.lang = item.lang or named.get(f.name) or read.get(item.id)
            if item.lang in ("none", ""):
                item.lang = None
            out.append(item)
    return out
