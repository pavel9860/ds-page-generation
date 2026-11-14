"""The results of a run (WORK/out/<run>) into the corpus tables, all or nothing per table, one backup `*.bak` of each table (the state before the change):
embeddings.npy / ids.npy / manifest.jsonl of emb_out; manifest.sqlite hw_det (text / hw box counts: cls 0 hw, cls 1 print), scan_pct (+ info/scan_pct.parquet),
lang_res (doc hash, language, source, probability) and src_urls (url, origin)."""
import sqlite3

import numpy as np
import pandas as pd

from . import embstore
from .paths import MANIFEST_DB, PARQUET, WORK
from .registry import db



def counts(z):
    """(ids, hw, text) per page of a detection shard."""
    o, c = z["offsets"], np.r_[0, np.cumsum(z["cls"] == 0)]
    hw = c[o[1:]] - c[o[:-1]]
    return z["ids"], hw, np.diff(o) - hw


def load(run):
    out = WORK / "out" / run
    E, I = np.load(out / "embeddings.npy"), np.load(out / "ids.npy")
    hw = [(counts(z), str(f)) for f in sorted((out / "hw").glob("pages_*.npz")) for z in [np.load(f)]]     # the shard by its full path (found again when runs move)
    pct = [np.load(f) for f in sorted(out.glob("scan_pct_*.npz"))]
    return E, I, hw, pct


def merge(run):
    c = db()
    E, I, hw, pct = load(run)
    reg = {d: (k, path) for d, k, path in c.execute("select doc, key, path from docs")}
    unknown = set(I[:, 0].tolist()) - set(reg)
    assert not unknown, f"{len(unknown)} embedded docs are not in the registry"
    removed, added, total = embstore.merge_rows(E, I)
    embstore.backup(("manifest.jsonl",))
    docs = sorted(set(I[:, 0].tolist()))
    embstore.add_manifest([(d, reg[d][1]) for d in docs])
    c.executemany("update docs set state = 'embedded' where doc = ? and state != 'merged'", [(d,) for d in docs])
    c.commit()

    embstore.backup((MANIFEST_DB.name,), MANIFEST_DB.parent)
    if PARQUET.exists():
        embstore.backup((PARQUET.name,), PARQUET.parent)

    H = pd.concat([pd.DataFrame({"doc": ids[:, 0], "page": ids[:, 1], "text": text, "hw": h, "shard": name})
                   for (ids, h, text), name in hw], ignore_index=True)
    H["hash"] = H.doc.map(lambda d: reg[d][0])
    P = pd.concat([pd.DataFrame({"doc": z["ids"][:, 0], "page": z["ids"][:, 1], "fill": z["fill"], "pic": z["pic"],
                                 "wx": z["win"][:, 0], "wy": z["win"][:, 1], "ww": z["win"][:, 2], "wh": z["win"][:, 3]}) for z in pct], ignore_index=True)
    P = P[P.fill.notna()].drop_duplicates(["doc", "page"])
    P.insert(0, "hash", P.doc.map(lambda d: reg[d][0]))
    keep = set(map(tuple, I.tolist()))
    H = H[[(d, p) in keep for d, p in zip(H.doc, H.page)]].drop_duplicates(["doc", "page"])
    P = P[[(d, p) in keep for d, p in zip(P.doc, P.page)]]
    with sqlite3.connect(MANIFEST_DB, timeout=600) as m:
        m.executemany("insert or replace into hw_det values (?,?,?,?,?,?,?)",
                      H.assign(src_doc=H.doc)[["doc", "page", "hash", "text", "hw", "shard", "src_doc"]].astype(object).itertuples(False, None))
        m.executemany("delete from scan_pct where hash = ? and page = ?", P[["hash", "page"]].itertuples(False, None))
        m.executemany("insert into scan_pct values (?,?,?,?,?,?,?,?)", P[["hash", "page", "fill", "pic", "wx", "wy", "ww", "wh"]].astype(object).itertuples(False, None))
    if PARQUET.exists():
        old = pd.read_parquet(PARQUET)
        old = old.merge(P[["doc", "page"]].drop_duplicates().assign(_x=1), on=["doc", "page"], how="left")
        old = old[old._x.isna()].drop(columns="_x")
        pd.concat([old, P[old.columns]], ignore_index=True).to_parquet(PARQUET)
    return {"emb removed": removed, "emb added": added, "emb total": total, "docs": len(docs), "hw_det": len(H), "scan_pct": len(P)}


def finish(run):
    """Language and origin of the embedded docs -> lang_res, src_urls; docs state `merged`. Run after lang.text_langs() / lang.embed_langs()."""
    c = db()
    rows = c.execute("select key, lang, lang_src, lang_p, src, origin from docs where state = 'embedded'").fetchall()
    with sqlite3.connect(MANIFEST_DB, timeout=600) as m:
        m.executemany("delete from lang_res where doc = ?", [(r[0],) for r in rows])
        m.executemany("insert into lang_res values (?,?,?,?,?)", [(k, lg, f"ingest_{s}", p, 0) for k, lg, s, p, _, _ in rows if lg])
        m.executemany("delete from src_urls where hash = ? and run = ?", [(r[0], f"ingest_{run}") for r in rows])
        m.executemany("insert into src_urls values (?,?,?,?,?,?)", [(k, u, o, lg, "ingest", f"ingest_{run}") for k, lg, _, _, u, o in rows])
    c.execute("update docs set state = 'merged' where state = 'embedded'")
    c.commit()
    return len(rows)
