"""The corpus files as they are now against the manifest (Layouts/manifest.sqlite): files deleted, moved or renamed, files whose content changed
(pages removed: the 12-hex hash of the name, sha256(bytes)[:12], no longer the content's). Scope: every folder of Layouts but archive/,
plus archive/rejected. Data: Layouts/archive/info/layouts_revise (WORK).

  python tools/layouts_revise.py scan    # every file: size, content hash, name hash, pages -> WORK/scan.parquet; comparison with the manifest
                                         # -> WORK/diff.parquet + counts (read-only; resumable by path, size and mtime)
  python tools/layouts_revise.py dups [--apply]   # copies: same bytes, or a pdf whose pages all are in another file of its name hash
                                         # -> WORK/dups.parquet; --apply moves them to archive/similar/dups/<path> (logged in moves.jsonl)
  python tools/layouts_revise.py manifest  # manifest.sqlite follows the files (copies, gone, moved), links of untyped_sources -> src_urls
                                         # -> WORK/manifest_changes.parquet (the trace of every change)
  python tools/layouts_revise.py pages     # embedding rows, scan_pct, hw_det: one id per file, one row per (file, page), copies / gone out
                                         # -> WORK/page_rows.parquet (the trace of every row); old files kept as *.before_revise

6 workers."""

import argparse
import hashlib
import os
import re
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf

from dspages.hand.config import PATHS

LAYOUTS = Path(PATHS.layouts)
WORK = LAYOUTS / "archive" / "info" / "layouts_revise"
SKIP = ("archive", "_staging")
EXTRA = ("archive/rejected",)
HEX = re.compile(r"^([0-9a-f]{12})(?![0-9a-f])")
WORKERS, CHUNK = 6, 1 << 20
pymupdf.TOOLS.mupdf_display_errors(False)


def files():
    """Relative paths of every file in scope."""
    tops = [d.name for d in LAYOUTS.iterdir() if d.is_dir() and d.name not in SKIP] + list(EXTRA)
    out = []
    for t in tops:
        for root, _, names in os.walk(LAYOUTS / t):
            out += [os.path.relpath(os.path.join(root, n), LAYOUTS) for n in names]
    return out


def info(rel):
    """(path, size, mtime, content hash, pages; -1 unreadable pdf, 1 a picture)."""
    p = LAYOUTS / rel
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(CHUNK):
            h.update(b)
    st = p.stat()
    pages = 1
    if rel.lower().endswith(".pdf"):
        try:
            with pymupdf.open(p) as d:
                pages = d.page_count
        except Exception:
            pages = -1
    return rel, st.st_size, st.st_mtime, h.hexdigest()[:12], pages


def cmd_scan(_):
    WORK.mkdir(parents=True, exist_ok=True)
    have = pd.read_parquet(WORK / "scan.parquet") if (WORK / "scan.parquet").exists() else None
    t0 = time.time()
    paths = files()
    print(len(paths), "files in scope", int(time.time() - t0), "s", flush=True)
    stat = {r: os.stat(LAYOUTS / r) for r in paths}
    keep = []
    if have is not None:
        ok = have[
            [
                r in stat and stat[r].st_size == s and stat[r].st_mtime == m
                for r, s, m in zip(have.path, have.size, have.mtime)
            ]
        ]
        keep, done = [ok], set(ok.path)
        paths = [r for r in paths if r not in done]
    rows = []
    with ProcessPoolExecutor(WORKERS) as ex:
        for i, r in enumerate(ex.map(info, paths, chunksize=64)):
            rows.append(r)
            if i % 20000 == 0:
                print(f"{i}/{len(paths)} {int(time.time() - t0)} s", flush=True)
    S = pd.concat(keep + [pd.DataFrame(rows, columns=["path", "size", "mtime", "content", "pages"])], ignore_index=True)
    S["name"] = S.path.str.rsplit("/", n=1).str[-1].str.extract(HEX)[0]
    S.to_parquet(WORK / "scan.parquet")
    diff(S)


def diff(S):
    """Disk against the manifest's plan (one row per document: doc = name hash, new_path, pages)."""
    with sqlite3.connect(f"file:{LAYOUTS / 'manifest.sqlite'}?mode=ro", uri=True) as db:
        plan = pd.read_sql("select doc, new_path, pages from plan", db)
        urls = pd.read_sql("select hash, url from src_urls", db).drop_duplicates("hash")
    in_scope = plan.new_path.str.startswith(EXTRA) | ~plan.new_path.str.split("/").str[0].isin(SKIP)
    plan = plan[in_scope]
    on_disk = set(S.path)
    by_name = S.dropna(subset=["name"]).groupby("name").path.apply(list)
    plan["state"] = "same"
    gone = ~plan.new_path.isin(on_disk)
    moved = gone & plan.doc.isin(by_name.index)
    plan.loc[moved, "state"] = "moved"
    plan.loc[gone & ~moved, "state"] = "deleted"
    plan["now"] = plan.new_path.where(~gone, plan.doc.map(lambda d: by_name.get(d, [None])[0]))
    cur = S.set_index("path")
    plan["content"] = plan.now.map(cur.content)
    plan["pages_now"] = plan.now.map(cur.pages)
    plan["changed"] = plan.content.notna() & (plan.content != plan.doc)
    plan["url"] = plan.doc.isin(urls.hash)
    new = S[~S.path.isin(plan.now.dropna())]
    print("manifest documents in scope", len(plan), "|", plan.state.value_counts().to_dict())
    print(
        "content changed (name hash != content hash)",
        int(plan.changed.sum()),
        "| of those with fewer pages than the manifest",
        int((plan.changed & (plan.pages_now < plan.pages)).sum()),
    )
    print("files on disk not in the manifest", len(new), "| without a 12-hex name", int(new.name.isna().sum()))
    print(
        "one name hash, several files",
        int((by_name.str.len() > 1).sum()),
        "| one content, several files",
        int(S.content.duplicated(keep=False).sum()),
    )
    print("documents with a download link", int(plan.url.sum()))
    plan.to_parquet(WORK / "diff.parquet")
    new.to_parquet(WORK / "new_files.parquet")


DUPS = (
    "archive/similar/dups"  # where copies go (moves.jsonl stages dup_same_bytes / dup_page_subset, with the kept file)
)
RANK = ("archive/rejected", "untyped_tmp")  # a copy here goes before one in a typed folder


def rank(path, manifest_paths):
    """Sort key of the copy to keep: the manifest's own file, a typed folder, untyped_tmp, archive/rejected; then the shorter path."""
    return (
        path not in manifest_paths,
        next((i + 1 for i, r in enumerate(reversed(RANK)) if path.startswith(r)), 0),
        len(path),
        path,
    )


def page_prints(rel):
    """Fingerprint per page (text + 0.25x grey render), [] when unreadable."""
    try:
        with pymupdf.open(LAYOUTS / rel) as d:
            out = []
            for pg in d:
                pm = pg.get_pixmap(matrix=pymupdf.Matrix(0.25, 0.25), colorspace=pymupdf.csGRAY)
                out.append(hashlib.md5(pg.get_text().encode() + pm.samples).hexdigest())
            return out
    except Exception:
        return []


def cmd_dups(a):
    """Copies to move -> WORK/dups.parquet (src, kept, stage); --apply moves them and logs every move in moves.jsonl."""
    S = pd.read_parquet(WORK / "scan.parquet")
    with sqlite3.connect(f"file:{LAYOUTS / 'manifest.sqlite'}?mode=ro", uri=True) as db:
        mpaths = {r[0] for r in db.execute("select new_path from plan")}
    moves = []
    for _, g in S[S.content.duplicated(keep=False)].groupby("content"):  # 1) same bytes
        keep, *rest = sorted(g.path, key=lambda p: rank(p, mpaths))
        moves += [(p, keep, "dup_same_bytes") for p in rest]
    gone = {m[0] for m in moves}
    G = S[S.name.notna() & ~S.path.isin(gone)]
    G = G[
        G.name.duplicated(keep=False) & G.path.str.lower().str.endswith(".pdf")
    ]  # 2) one name hash, other bytes: page subsets
    with ProcessPoolExecutor(WORKERS) as ex:
        fp = dict(zip(G.path, ex.map(page_prints, G.path, chunksize=4)))
    for _, g in G.groupby("name"):
        ps = sorted(g.path, key=lambda p: (-len(fp[p]), rank(p, mpaths)))
        for i, p in enumerate(ps):
            sup = next((q for q in ps[:i] if q not in gone and fp[p] and set(fp[p]) <= set(fp[q])), None)
            if sup:
                moves.append((p, sup, "dup_page_subset"))
                gone.add(p)
    D = pd.DataFrame(moves, columns=["src", "kept", "stage"])
    D["in_manifest"] = D.src.isin(mpaths)
    D.to_parquet(WORK / "dups.parquet")
    print(D.groupby("stage").agg(files=("src", "size"), manifest_paths=("in_manifest", "sum")).to_string())
    print("by folder:", D.src.str.split("/").str[0].value_counts().head(8).to_dict())
    if a.apply:
        apply_moves(D)


def apply_moves(D):
    import json

    done = 0
    with open(LAYOUTS / "archive" / "info" / "moves.jsonl", "a") as log:
        for src, kept, stage in zip(D.src, D.kept, D.stage):
            s, t = LAYOUTS / src, LAYOUTS / DUPS / src
            if not s.exists() or not (LAYOUTS / kept).exists():
                continue
            t.parent.mkdir(parents=True, exist_ok=True)
            os.rename(s, t)
            log.write(json.dumps({"stage": stage, "src": src, "dst": f"{DUPS}/{src}", "kept": kept}) + "\n")
            done += 1
    print(done, "files moved to", DUPS)


DOC_TABLES = (
    "docs",
    "plan",
    "feats",
    "fonts",
    "fonts_done",
    "lang_res",
    "lang_text",
    "ft_probs",
    "ft_more",
    "site_res",
    "lang_img",
    "lang_strip",
)


def cmd_manifest(_):
    """manifest.sqlite in step with the files of the scope: copies moved by `dups` -> their kept file; files rows of files that are gone
    out; a document without a file (none under its path, none with its hash in the name) out of every document table; a moved file
    (same name hash, other path) -> its path; the download links of untyped_sources.jsonl added to src_urls.
    Every change -> WORK/manifest_changes.parquet (table, doc, old, new, action)."""
    import json

    in_scope = lambda p: p.startswith(EXTRA) or p.split("/")[0] not in SKIP
    present = {p for p in files()}
    by_name = {}
    for p in sorted(present):
        m = HEX.match(p.rsplit("/", 1)[-1])
        if m:
            by_name.setdefault(m.group(1), p)
    D = pd.read_parquet(WORK / "dups.parquet")
    mv = dict(zip(D.src, D.kept))
    log = []
    con = sqlite3.connect(LAYOUTS / "manifest.sqlite", timeout=60)
    with con:
        for path, h in con.execute("select path, hash from files").fetchall():
            if not in_scope(path) or path in present:
                continue
            new = mv.get(path)
            if new and not con.execute("select 1 from files where path=?", (new,)).fetchone():
                con.execute("update files set path=? where path=?", (new, path))
                log.append(("files", h, path, new, "copy -> kept file"))
            else:
                con.execute("delete from files where path=?", (path,))
                log.append(("files", h, path, None, "file gone"))
        gone = []
        for doc, path in con.execute("select doc, new_path from plan").fetchall():
            if not in_scope(path) or path in present:
                continue
            new = mv.get(path) or by_name.get(doc)
            if new:
                con.execute("update plan set new_path=? where doc=?", (new, doc))
                log.append(("plan", doc, path, new, "copy -> kept file" if path in mv else "moved / renamed"))
            else:
                gone.append(doc)
                log.append(("plan", doc, path, None, "document gone"))
        for t in DOC_TABLES:
            con.executemany(f"delete from {t} where doc=?", [(d,) for d in gone])
        for doc, paths in con.execute("select doc, paths from docs").fetchall():
            new = [q for q in (mv.get(p, p) for p in json.loads(paths or "[]")) if q in present or not in_scope(q)]
            new = list(dict.fromkeys(new)) or [
                r[0] for r in con.execute("select new_path from plan where doc=?", (doc,))
            ]
            if new != json.loads(paths or "[]"):
                con.execute("update docs set paths=?, n_files=? where doc=?", (json.dumps(new), len(new), doc))
        have = {r[0] for r in con.execute("select hash from src_urls")}
        add = []
        for r in map(json.loads, open(LAYOUTS / "archive" / "info" / "untyped_sources.jsonl")):
            h = r["file"][:12]
            if r.get("url") and h not in have:
                add.append((h, r["url"], r.get("origin"), r.get("lang"), r.get("arm"), r.get("run")))
                have.add(h)
        con.executemany("insert into src_urls (hash, url, origin, claim, arm, run) values (?,?,?,?,?,?)", add)
        log += [("src_urls", h, None, u, "link from untyped_sources") for h, u, *_ in add]
    con.close()
    L = pd.DataFrame(log, columns=["table", "doc", "old", "new", "action"])
    L.to_parquet(WORK / "manifest_changes.parquet")
    print(L.groupby(["table", "action"]).size().to_string())


def cmd_pages(_):
    """Embedding rows, fill / pic and boxes follow the files: one document id per file (the smallest id of the file's ids), one row per
    (file, page) (repeats of a page under the piece ids of a joined document are byte-identical renders), rows of files that are gone
    or were moved away as copies dropped; a file that moved keeps its rows under its new path. emb_out (embeddings.npy, ids.npy,
    manifest.jsonl; the old ones kept as *.before_revise), scan_pct (parquet + table), hw_det. Trace -> WORK/page_rows.parquet."""
    import json

    from dspages.hand.config import PATHS as P

    emb = Path(P.uvtm) / "Labels" / "embedding" / "emb_out"
    info = LAYOUTS / "archive" / "info"
    present = set(files())
    by_name = {}
    for p in sorted(present):
        m = HEX.match(p.rsplit("/", 1)[-1])
        if m:
            by_name.setdefault(m.group(1), p)
    dup_src = set(pd.read_parquet(WORK / "dups.parquet").src)
    man = pd.DataFrame([json.loads(l) for l in open(emb / "manifest.jsonl")]).rename(columns={"id": "doc"})
    in_scope = lambda p: p.startswith(EXTRA) or p.split("/")[0] not in SKIP

    def now(p):
        if p in present or not in_scope(p):
            return p
        if p in dup_src:
            return None  # a copy: its pages are in the kept file, under other page numbers
        m = HEX.match(p.rsplit("/", 1)[-1])
        return by_name.get(m.group(1)) if m else None

    man["now"] = man.path.map(now)
    ids = np.load(emb / "ids.npy")
    R = pd.DataFrame({"row": np.arange(len(ids)), "doc": ids[:, 0], "page": ids[:, 1]}).merge(
        man[["doc", "now"]], on="doc", how="left"
    )
    R["new_doc"] = R.groupby("now").doc.transform("min")
    R["keep"] = R.now.notna() & ~R.duplicated(["now", "page"])
    R["why"] = np.where(R.now.isna(), "file gone or a copy", np.where(R.keep, "kept", "repeat of a (file, page)"))
    R.drop(columns="row").to_parquet(WORK / "page_rows.parquet")
    print(R.why.value_counts().to_string())
    K = R[R.keep].sort_values(["new_doc", "page"])
    E = np.load(emb / "embeddings.npy", mmap_mode="r")
    out = np.lib.format.open_memmap(emb / "embeddings.new.npy", mode="w+", dtype=E.dtype, shape=(len(K), E.shape[1]))
    rows = K.row.to_numpy()
    for a in range(0, len(rows), 50_000):
        out[a : a + 50_000] = E[rows[a : a + 50_000]]
    out.flush()
    del out
    for f in ("embeddings.npy", "ids.npy", "manifest.jsonl"):
        os.replace(emb / f, emb / f"{f}.before_revise")
    os.replace(emb / "embeddings.new.npy", emb / "embeddings.npy")
    np.save(emb / "ids.npy", K[["new_doc", "page"]].to_numpy(np.int32))
    m = man[man.doc.isin(set(K.new_doc))].assign(path=lambda d: d.now)[["doc", "path"]].sort_values("doc")
    (emb / "manifest.jsonl").write_text(
        "".join(json.dumps({"id": int(d), "path": p}) + "\n" for d, p in zip(m.doc, m.path))
    )
    remap = K[["doc", "page", "new_doc"]]

    s = pd.read_parquet(info / "scan_pct.parquet").merge(remap, on=["doc", "page"])
    s = s.assign(
        doc=s.new_doc,
        hash=s.new_doc.map(
            dict(zip(m.doc, m.path.map(lambda p: (HEX.match(p.rsplit("/", 1)[-1]) or [None, None])[1])))
        ),
    )
    s = s.drop(columns="new_doc")
    os.replace(info / "scan_pct.parquet", info / "scan_pct.parquet.before_revise")
    s.to_parquet(info / "scan_pct.parquet")
    with sqlite3.connect(LAYOUTS / "manifest.sqlite", timeout=600) as db:
        h = pd.read_sql("select doc, page, hash, text, hw, shard, src_doc from hw_det", db).merge(
            remap, on=["doc", "page"]
        )
        h = h.assign(doc=h.new_doc, hash=h.new_doc.map(dict(zip(s.doc, s.hash)))).drop(columns="new_doc")
        db.execute("delete from hw_det")
        db.executemany(
            "insert into hw_det values (?,?,?,?,?,?,?)",
            h[["doc", "page", "hash", "text", "hw", "shard", "src_doc"]].astype(object).itertuples(False, None),
        )
        E2 = s[s.hash.notna()].drop_duplicates(["hash", "page"])
        db.execute("delete from scan_pct")
        db.executemany(
            "insert into scan_pct values (?,?,?,?,?,?,?,?)",
            E2[["hash", "page", "fill", "pic", "wx", "wy", "ww", "wh"]].astype(object).itertuples(False, None),
        )
    print(
        f"embedding rows {len(R)} -> {len(K)}; documents {man.doc.nunique()} -> {len(m)}; scan_pct {len(s)}; hw_det {len(h)}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("scan")
    d = sp.add_parser("dups")
    d.add_argument("--apply", action="store_true")
    sp.add_parser("manifest")
    sp.add_parser("pages")
    a = ap.parse_args()
    {"scan": cmd_scan, "dups": cmd_dups, "manifest": cmd_manifest, "pages": cmd_pages}[a.cmd](a)
