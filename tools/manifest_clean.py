"""Manifest clean-up of the corpus: documents whose file is gone, file names <hash>_<lang><ext>, the old names dropped from manifest.sqlite.

  python tools/manifest_clean.py dry     # counts only (disk index, gone / relocated / renames), writes WORK/plan.parquet
  python tools/manifest_clean.py apply   # 1) gone documents: rows out of every table (src_urls, the download links, stay)
                                         # 2) moved files: plan.new_path follows (the single file of the same hash and extension)
                                         # 3) renames <hash>_<lang><ext> in the file's folder, traced only in WORK/rename_log.tsv (not in moves.jsonl)
                                         # 4) the old names saved to WORK/old_names.parquet, then dropped: plan.orig_path / orig_name / new_name / src, old_meta.path; files and docs follow
                                         # 5) emb_out/manifest.jsonl paths follow; VACUUM
Rename: every document with a 12-hex id and a language value (mul / xx = not detected, kept in the name); a name already taken stays. The manifest was backed up before (manifest.sqlite.bak_before_cleanup_*)."""
import argparse
import collections
import json
import os
import re
import sqlite3
from pathlib import Path

import pandas as pd

from dspages.hand.config import PATHS

LAYOUTS = Path(PATHS.layouts)
EMB = Path(PATHS.uvtm) / "emb_out"
INFO = LAYOUTS / "archive" / "info"
WORK = Path(PATHS.uvtm) / "tests" / "work" / "manifest_clean"
MOVES = INFO / "moves.jsonl"
HEX = re.compile(r"(?<![0-9a-f])[0-9a-f]{12}(?![0-9a-f])")
NO_LANG = frozenset({""})                    # mul / xx are kept as names: "language not detected"
SKIP_INDEX = ("archive/similar/", "archive/tmp/")
DOC_TABLES = ("feats", "fonts", "fonts_done", "lang_res", "lang_text", "ft_probs", "ft_more", "lang_img", "lang_strip", "site_res", "docs", "plan")
HASH_TABLES = ("old_meta", "old_pagetype", "joined", "labels", "jpeg_grid")


def disk_index():
    """{sha12: [relative paths]} of every file under Layouts (not the moved duplicates)."""
    idx = collections.defaultdict(list)
    n = 0
    for dp, dn, fn in os.walk(LAYOUTS):
        rel = os.path.relpath(dp, LAYOUTS).replace(os.sep, "/")
        if (rel + "/").startswith(SKIP_INDEX):
            dn[:] = []
            continue
        for f in fn:
            n += 1
            for h in HEX.findall(f):
                idx[h].append(f"{rel}/{f}" if rel != "." else f)
    print(n, "files on disk,", len(idx), "hashes", flush=True)
    return idx


def target(doc, lang, ext, path):
    """New file name of a document, or None when it keeps its name."""
    if not HEX.fullmatch(doc) or (lang or "") in NO_LANG:
        return None
    new = f"{doc}_{lang}{ext}"
    return None if path.rsplit("/", 1)[-1] == new else new


def make_plan():
    db = sqlite3.connect(f"file:{LAYOUTS / 'manifest.sqlite'}?mode=ro", uri=True)
    P = pd.read_sql("select doc, lang, ext, new_path from plan", db)
    db.close()
    idx = disk_index()
    cur, status = [], []
    for doc, ext, p in zip(P.doc, P.ext, P.new_path):
        if (LAYOUTS / p).is_file():
            cur.append(p), status.append("here")
            continue
        alt = [q for q in idx.get(doc, []) if q.lower().endswith(ext.lower())]
        if len(alt) == 1:
            cur.append(alt[0]), status.append("moved")
        else:
            cur.append(None), status.append("gone" if not alt else "ambiguous")
    P["cur"], P["status"] = cur, status
    P["new_name"] = [target(d, l, e, c) if pd.notna(c) else None for d, l, e, c in zip(P.doc, P.lang, P.ext, P.cur)]
    P["taken"] = [(LAYOUTS / c.rsplit("/", 1)[0] / n).exists() if pd.notna(c) and pd.notna(n) else False for c, n in zip(P.cur, P.new_name)]
    WORK.mkdir(parents=True, exist_ok=True)
    P.to_parquet(WORK / "plan.parquet")
    print(P.status.value_counts().to_dict())
    r = P[P.new_name.notna()]
    print(len(r), "renames,", int(r.taken.sum()), "names already taken (skipped)")
    print(r.cur.str.split("/").str[0].value_counts().head(6).to_dict())
    return P


def cmd_apply(_):
    P = pd.read_parquet(WORK / "plan.parquet") if (WORK / "plan.parquet").exists() else make_plan()
    db = sqlite3.connect(LAYOUTS / "manifest.sqlite", timeout=600)
    gone = P.doc[P.status == "gone"].tolist()           # ambiguous ones (several files of the hash) stay as they are
    print(len(gone), "documents without a file: rows dropped", flush=True)
    for t in DOC_TABLES:
        db.executemany(f"delete from {t} where doc=?", [(d,) for d in gone])
    for t in HASH_TABLES:
        db.executemany(f"delete from {t} where hash=?", [(d,) for d in gone])
    db.commit()
    moved = P[P.status == "moved"]
    db.executemany("update plan set new_path=? where doc=?", list(zip(moved.cur, moved.doc)))
    db.commit()
    log = WORK / "rename_log.tsv"
    done = 0
    with open(log, "a") as lg:
        if log.stat().st_size == 0:
            lg.write("doc\told\tnew\n")
        for r in P[P.new_name.notna() & ~P.taken & P.cur.notna()].itertuples():
            folder = r.cur.rsplit("/", 1)[0]
            new = f"{folder}/{r.new_name}"
            if not (LAYOUTS / r.cur).exists() or (LAYOUTS / new).exists():
                continue
            os.rename(LAYOUTS / r.cur, LAYOUTS / new)
            lg.write(f"{r.doc}\t{r.cur}\t{new}\n")
            db.execute("update plan set new_path=? where doc=?", (new, r.doc))
            db.execute("update files set path=? where path=?", (new, r.cur))
            db.execute("update docs set paths=? where doc=?", (json.dumps([new]), r.doc))
            done += 1
            if done % 2000 == 0:
                db.commit()
    db.commit()
    print(done, "files renamed; log", log, flush=True)
    pd.read_sql("select doc as hash, orig_path, orig_name, new_name, src, orig_folder from plan", db).to_parquet(WORK / "old_names.parquet")
    pd.read_sql("select hash, path as old_path from old_meta", db).to_parquet(WORK / "old_meta_paths.parquet")
    print("old names kept in", WORK / "old_names.parquet", "and old_meta_paths.parquet", flush=True)
    for t, c in (("plan", "orig_path"), ("plan", "orig_name"), ("plan", "new_name"), ("plan", "src"), ("old_meta", "path")):
        if c in {x[1] for x in db.execute(f"pragma table_info({t})")}:
            db.execute(f"alter table {t} drop column {c}")
    db.commit()
    ex = [p for (p,) in db.execute("select path from files")]
    miss = [(p,) for p in ex if not (LAYOUTS / p).exists()]
    db.executemany("delete from files where path=?", miss)
    db.commit()
    print(len(miss), "rows of files without a file dropped", flush=True)
    cur = dict(db.execute("select doc, new_path from plan"))
    rows = [json.loads(l) for l in open(EMB / "manifest.jsonl")]
    bak = EMB / "manifest.jsonl.before_cleanup"
    if not bak.exists():
        bak.write_text("".join(json.dumps(r) + "\n" for r in rows))
    changed = 0
    for r in rows:
        # The 12-hex id in a name is not an identity of the bytes: copies in other folders, a re-saved pdf with pages dropped and crops named
        # real_b_143_<hash>_bn.jpg share it, and the embedded pages are numbered for the file of the manifest. Only a path whose file is gone moves.
        if (LAYOUTS / r["path"]).is_file():
            continue
        h = next(iter(HEX.findall(r["path"].rsplit("/", 1)[-1])), None)
        if h in cur and cur[h] != r["path"] and (LAYOUTS / cur[h]).is_file():
            r["path"], changed = cur[h], changed + 1
    (EMB / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(changed, "paths of the embedding manifest updated", flush=True)
    db.execute("pragma wal_checkpoint(truncate)")
    db.execute("vacuum")
    db.close()
    print("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dry").set_defaults(f=lambda a: make_plan())
    sub.add_parser("apply").set_defaults(f=cmd_apply)
    a = ap.parse_args()
    a.f(a)
