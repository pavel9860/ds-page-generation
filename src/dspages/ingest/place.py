"""Merged documents into the corpus folders. Type from the box counts of their pages (ds_metric.hw_class: hw = hw > 10 and print < 5, filled = hw > 2 and
print > 10): half or more of the pages hw -> handwritten/, filled -> filled_forms/, anything else -> untyped_tmp/. Name <sha12>_<lang>.<ext> (corpus naming); the
file is moved (never copied), emb_out manifest.jsonl, the registry and manifest.sqlite plan follow."""
import json
import os
import sqlite3

import numpy as np

from . import embstore
from .paths import ACQ_DB, EMB, LAYOUTS, MANIFEST_DB
from .registry import db

SHARE = 0.5
FOLDER = {"hw": "handwritten", "filled": "filled_forms", None: "untyped_tmp"}
TYPE = {"hw": "handwritten", "filled": "filled_forms", None: "_untyped"}


def page_class(hw, text):
    return np.where((hw > 10) & (text < 5), 2, np.where((hw > 2) & (text > 10), 1, 0))


def doc_class(rows, pages):
    """'hw' / 'filled' / None from [(hw, text)] of the pages of a document."""
    if not rows:
        return None
    k = page_class(np.array([r[0] for r in rows]), np.array([r[1] for r in rows]))
    n = max(pages, len(rows))
    return "hw" if (k == 2).sum() >= SHARE * n else "filled" if (k == 1).sum() >= SHARE * n else None


def scan_pages():
    with sqlite3.connect(f"file:{ACQ_DB}?mode=ro", uri=True) as a:
        return {f.split("/")[-1].split(".")[0]: (n, s) for f, n, s in a.execute("select file, pages, scan_pages from got where status = 'ok'")}


def place():
    """-> {folder: documents}."""
    c = db()
    embstore.backup(("manifest.jsonl",))
    scans = scan_pages()
    man = embstore.manifest()
    done, out = {}, {}
    with sqlite3.connect(MANIFEST_DB, timeout=600) as m:
        for doc, key, kind, path, pages, lang, src in c.execute("select doc, key, kind, path, pages, lang, lang_src from docs where state = 'merged'").fetchall():
            cls = doc_class(m.execute("select hw, text from hw_det where doc = ?", (doc,)).fetchall(), pages)
            ext = os.path.splitext(path)[1].lower()
            rel = f"{FOLDER[cls]}/{key}_{lang or 'unknown'}{ext}"
            dst = LAYOUTS / rel
            if not dst.exists():
                os.replace(LAYOUTS / path, dst)
            n, s = scans.get(key, (1, 1))
            scan = kind == "img" or s >= 0.5 * n
            m.execute("delete from plan where doc = ?", (key,))
            m.execute("insert into plan values (?,?,?,?,?,?,?,?,?,?,?,?)", (key, "scans" if scan else "docs", TYPE[cls], lang, f"ingest_{src}",
                      "scan-like" if scan else "text", rel, ext, pages, dst.stat().st_size / 1e6, lang, FOLDER[cls]))
            c.execute("update docs set path = ?, state = 'placed' where doc = ?", (rel, doc))
            done[doc] = rel
            out[FOLDER[cls]] = out.get(FOLDER[cls], 0) + 1
    c.commit()
    keep = {**man, **done}
    tmp = EMB / "manifest.jsonl.new"
    with open(tmp, "w") as fh:
        fh.writelines(json.dumps({"id": d, "path": p}) + "\n" for d, p in sorted(keep.items()))
    os.replace(tmp, EMB / "manifest.jsonl")
    return out
