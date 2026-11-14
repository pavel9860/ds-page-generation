"""Duplicates by the 1536-d page vectors (ds_border same: pages at k-NN distance 0, grouped): documents whose every page is a page of another
document (a smaller or equal copy) go to archive/similar/dups like the copies of tools/layouts_revise.py (moves.jsonl stage dup_emb_subset,
kept = the larger document, equal size: layouts_revise.rank); of the identical pages left, one page per group stays, the others leave the
embedding rows, scan_pct and hw_det (the files stay). Data: Layouts/archive/info/emb_dedup (WORK).

  plan    -> WORK/doc_moves.parquet (src, kept, stage), WORK/page_drops.parquet (doc, page, h, path, group); counts
  apply   one backup of the state before (<file>.bak: manifest.sqlite, emb_out embeddings.npy / ids.npy / manifest.jsonl, scan_pct.parquet,
          layouts_revise dups.parquet), the files moved and added to layouts_revise WORK/dups.parquet,
          then: python tools/layouts_revise.py manifest; python tools/layouts_revise.py pages; python tools/emb_dedup.py rows
  rows    the dropped pages out of emb_out (embeddings.npy, ids.npy), scan_pct (parquet + table), hw_det; the *.before_revise copies of
          `pages` deleted (the .bak of apply is the one backup)

python tools/emb_dedup.py plan | apply | rows"""
import argparse
import itertools
import os
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import layouts_revise as LR
from dspages.hand.config import PATHS

LAYOUTS = Path(PATHS.layouts)
WORK = Path(PATHS.uvtm) / "tests" / "work" / "emb_dedup"
SAME = "/run/media/p/D/ML_DS/UVTM/tests/ds_border/same.parquet"
EMB = Path(PATHS.uvtm) / "Labels" / "embedding" / "emb_out"
BACKUP = [EMB / "embeddings.npy", EMB / "ids.npy", EMB / "manifest.jsonl", LAYOUTS / "archive" / "info" / "scan_pct.parquet", LR.WORK / "dups.parquet"]


def cmd_plan(_):
    G = pd.read_parquet(SAME)
    with sqlite3.connect(f"file:{LAYOUTS / 'manifest.sqlite'}?mode=ro", uri=True) as db:
        pg = pd.read_sql("select doc as h, pages from plan", db).drop_duplicates("h").set_index("h").pages
        mpaths = {r[0] for r in db.execute("select new_path from plan")}
    path = G.drop_duplicates("h").set_index("h").path
    pairs = {}
    for _, d in G.groupby("group"):
        for a, b in itertools.combinations(sorted(d.h.unique()), 2):
            pairs[(a, b)] = pairs.get((a, b), 0) + 1
    moves, gone = [], set()
    for (a, b), n in sorted(pairs.items(), key=lambda kv: -kv[1]):
        if a in gone or b in gone or pd.isna(pg.get(a)) or pd.isna(pg.get(b)):
            continue
        large, small = sorted((a, b), key=lambda h: (-pg[h], LR.rank(path[h], mpaths)))
        if n >= pg[small]:
            moves.append((path[small], path[large], "dup_emb_subset", small, int(pg[small]), int(pg[large])))
            gone.add(small)
    M = pd.DataFrame(moves, columns=["src", "kept", "stage", "h", "pages", "kept_pages"])
    R = G[~G.h.isin(gone)].copy()
    R["rk"] = [LR.rank(p, mpaths) for p in R.path]
    R = R.sort_values(["group", "rk", "doc", "page"])
    Dp = R[R.duplicated("group")][["doc", "page", "h", "path", "group"]]
    WORK.mkdir(parents=True, exist_ok=True)
    M.to_parquet(WORK / "doc_moves.parquet")
    Dp.to_parquet(WORK / "page_drops.parquet")
    print(f"documents to move (all pages in another document): {len(M)}  (smaller {int((M.pages < M.kept_pages).sum())}, equal size {int((M.pages == M.kept_pages).sum())})")
    print(f"pages to drop from the data (identical page kept once): {len(Dp)} of {Dp.h.nunique()} documents")


def cmd_apply(_):
    M = pd.read_parquet(WORK / "doc_moves.parquet")
    bak = LAYOUTS / "manifest.sqlite.bak"
    bak.unlink(missing_ok=True)
    with sqlite3.connect(LAYOUTS / "manifest.sqlite") as src, sqlite3.connect(bak) as dst:
        src.backup(dst)
    for f in BACKUP:
        shutil.copy(f, f"{f}.bak")
    print("backed up (.bak):", bak, *BACKUP)
    LR.apply_moves(M[["src", "kept", "stage"]])
    old = pd.read_parquet(LR.WORK / "dups.parquet")
    add = M[["src", "kept", "stage"]].assign(in_manifest=True)
    pd.concat([old, add[~add.src.isin(old.src)]]).to_parquet(LR.WORK / "dups.parquet")
    print("added to", LR.WORK / "dups.parquet:", len(add))


def cmd_rows(_):
    Dp = pd.read_parquet(WORK / "page_drops.parquet")
    man = pd.read_json(EMB / "manifest.jsonl", lines=True).rename(columns={"id": "doc"})
    hp = dict(zip(man.path.map(lambda p: p.rsplit("/", 1)[-1][:12]), man.doc))
    drop = {(hp[h], int(p)) for h, p in zip(Dp.h, Dp.page) if h in hp}
    ids = np.load(EMB / "ids.npy")
    keep = np.array([(int(d), int(p)) not in drop for d, p in ids])
    print(f"embedding rows {len(ids)} -> {int(keep.sum())} ({len(drop)} pages to drop, {int((~keep).sum())} found)")
    E = np.load(EMB / "embeddings.npy", mmap_mode="r")
    out = np.lib.format.open_memmap(EMB / "embeddings.new.npy", mode="w+", dtype=E.dtype, shape=(int(keep.sum()), E.shape[1]))
    rows = np.flatnonzero(keep)
    for a in range(0, len(rows), 50_000):
        out[a:a + 50_000] = E[rows[a:a + 50_000]]
    out.flush()
    del out, E
    os.replace(EMB / "embeddings.new.npy", EMB / "embeddings.npy")
    np.save(EMB / "ids.npy", ids[keep])
    info = LAYOUTS / "archive" / "info"
    s = pd.read_parquet(info / "scan_pct.parquet")
    ks = ~pd.Series([(int(d), int(p)) in drop for d, p in zip(s.doc, s.page)], index=s.index)
    s[ks].to_parquet(info / "scan_pct.parquet")
    hd = {(h, int(p)) for h, p in zip(Dp.h, Dp.page)}
    with sqlite3.connect(LAYOUTS / "manifest.sqlite", timeout=600) as db:
        n0 = db.execute("select count(*) from hw_det").fetchone()[0]
        db.executemany("delete from hw_det where doc=? and page=?", list(drop))
        db.executemany("delete from scan_pct where hash=? and page=?", list(hd))
        n1 = db.execute("select count(*) from hw_det").fetchone()[0]
    print(f"scan_pct parquet {len(s)} -> {int(ks.sum())}; hw_det {n0} -> {n1}")
    for f in [*(EMB / x for x in ("embeddings.npy", "ids.npy", "manifest.jsonl")), info / "scan_pct.parquet"]:
        Path(f"{f}.before_revise").unlink(missing_ok=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for c in ("plan", "apply", "rows"):
        sp.add_parser(c)
    a = ap.parse_args()
    {"plan": cmd_plan, "apply": cmd_apply, "rows": cmd_rows}[a.cmd](a)
