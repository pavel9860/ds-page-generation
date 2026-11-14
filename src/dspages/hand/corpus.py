"""Access to the corpus catalogue: documents, their languages and page types (SQLite manifest + embedding classification)."""
import json
import os
import sqlite3
from pathlib import Path

from .config import PATHS


def connect():
    return sqlite3.connect(f"file:{PATHS.manifest}?mode=ro", uri=True)


def doc_id(path):
    """Document hash of a corpus file name (<hash>_<lang>.pdf, <hash>.latin.pdf, ...)."""
    return os.path.basename(path).split("_")[0].split(".")[0]


def page_types(min_p=0.0):
    """{document: (label, p)} of the embedding classifier, for p >= min_p."""
    out = {}
    for line in open(PATHS.doc_types):
        r = json.loads(line)
        if r["p"] >= min_p:
            out[doc_id(r["path"])] = (r["label"], r["p"])
    return out


def languages(db, min_conf=0.95, min_chars=0):
    """{document: language} of the confident language labels."""
    return {d: lg for d, lg, cf, ch in db.execute("select doc,lang,conf,chars from lang_res") if cf >= min_conf and (ch or 0) >= min_chars}


def resolve(rel):
    """Path of a corpus file given relative to Layouts: in Layouts, or in Refine_DS where folders were moved."""
    p = Path(PATHS.layouts) / rel
    if p.exists():
        return p
    q = Path(PATHS.refine) / rel
    return q if q.exists() else p


def pdf_documents(db):
    """[(document, path, pages)] of the PDFs with a digital text layer."""
    return [(d, resolve(p), n or 1) for d, p, c, n in db.execute("select doc,new_path,cls,pages from plan")
            if c == "text" and p.lower().endswith(".pdf")]


def pdf_paths(db):
    """{document: path} of all PDFs (the files were moved after earlier labels were written)."""
    return {d: resolve(p) for d, p in db.execute("select doc,new_path from plan") if p.lower().endswith(".pdf")}
