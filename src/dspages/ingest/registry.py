"""The documents of an ingest: one row per file (sha256[:12] of the bytes = the name hash the corpus tables key on), its embedding-table id, kind
(pdf / img), pages picked, language. Doc ids are assigned once, above the largest id of the embedding table, and never change.
Sources: the kept downloads of dspages.acquire (acquired/pdf) and the Lens images (acquired/lens/got.csv, minus `excluded`)."""

import csv
import hashlib
import os
import sqlite3
from pathlib import Path

import pymupdf

from . import embstore
from .paths import ACQ, ACQ_DB, DB, LAYOUTS, LENS, MANIFEST_DB, WORK

SCHEMA = """
create table if not exists docs (key text primary key, kind text, path text, doc int unique, pages int, picked int, src text, origin text,
                                 lang text, lang_src text, lang_p real, state text default 'new');
create table if not exists pages (doc int, page int, shard text, primary key (doc, page));
create table if not exists excluded (file text primary key, why text);
"""


def db():
    WORK.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB, timeout=60)
    c.executescript(SCHEMA)
    return c


def sha12(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:12]


def corpus_hashes():
    with sqlite3.connect(f"file:{MANIFEST_DB}?mode=ro", uri=True) as m:
        return {h for (h,) in m.execute("select doc from plan")} | {
            h for (h,) in m.execute("select hash from src_urls")
        }


def exclude_dropped(snapshot, folders):
    """Images of the snapshot got.csv that are in none of the folders any more were dropped by hand: never ingested. -> number excluded."""
    here = {f for d in folders for f in os.listdir(d)}
    c = db()
    rows = [
        (r["file"], "dropped by hand from " + Path(snapshot).parent.name)
        for r in csv.DictReader(open(snapshot))
        if os.path.basename(r["file"]) not in here
    ]
    c.executemany("insert or ignore into excluded values (?,?)", rows)
    c.commit()
    return len(rows)


def pdf_sources():
    """(key, relative path, pages, url) of the kept downloads, and of the valid pdfs of acquired/pdf that no row of the download log knows (url None)."""
    with sqlite3.connect(f"file:{ACQ_DB}?mode=ro", uri=True) as a:
        rows = [
            (Path(f).stem, f"acquired/{f}", n, u)
            for u, f, n in a.execute("select url, file, pages from got where status = 'ok' and file is not null")
        ]
        known = {f for (f,) in a.execute("select file from got where file is not null")}
    for f in sorted(os.listdir(ACQ / "pdf")):
        if f"pdf/{f}" in known:
            continue
        try:
            with pymupdf.open(ACQ / "pdf" / f) as doc:
                if 0 < doc.page_count <= 400:
                    rows.append((sha12(ACQ / "pdf" / f), f"acquired/pdf/{f}", doc.page_count, None))
        except Exception:
            pass
    return rows


def folder_images(folders):
    """(key, relative path, 1, None) of the jpg / png files of loose folders (test downloads), linked into acquired/img/<sha12>.<ext>; a file inside
    Layouts (generated/) stays where it is, under its own path."""
    (ACQ / "img").mkdir(exist_ok=True)
    out = []
    for d in folders:
        for root, _, names in os.walk(d):
            for n in sorted(names):
                ext = os.path.splitext(n)[1].lower()
                if ext not in (".jpg", ".jpeg", ".png"):
                    continue
                k = sha12(Path(root) / n)
                if Path(root).resolve().is_relative_to(LAYOUTS.resolve()):
                    out.append((k, str((Path(root) / n).resolve().relative_to(LAYOUTS.resolve())), 1, None))
                    continue
                dst = ACQ / "img" / f"{k}{ext}"
                if not dst.exists():
                    os.link(Path(root) / n, dst)
                out.append((k, f"acquired/img/{k}{ext}", 1, None))
    return out


def img_sources(c):
    """(key, relative path, 1, url) of the Lens images not excluded; the file is linked into acquired/img/<sha12>.jpg (one name per content)."""
    skip = {f for (f,) in c.execute("select file from excluded")}
    (ACQ / "img").mkdir(exist_ok=True)
    out = []
    for r in csv.DictReader(open(LENS / "got.csv")):
        if r["file"] in skip:
            continue
        src = LENS / "got" / r["file"]
        if not src.exists():
            continue
        k = sha12(src)
        dst = ACQ / "img" / f"{k}.jpg"
        if not dst.exists():
            os.link(src, dst)
        out.append((k, f"acquired/img/{k}.jpg", 1, r["url"]))
    return out


def register(folders=(), acquired=True):
    """New documents -> docs (state `new`) with ids above the largest; a content the corpus or the registry has is skipped. -> {kind: added}.
    acquired=False: only the images of the folders (not the acquire downloads and Lens images)."""
    c = db()
    have = {k for (k,) in c.execute("select key from docs")} | corpus_hashes()
    nxt = max(embstore.max_doc(), c.execute("select coalesce(max(doc), 0) from docs").fetchone()[0]) + 1
    added = {"pdf": 0, "img": 0}
    for kind, rows in (
        ("pdf", pdf_sources() if acquired else []),
        ("img", (img_sources(c) if acquired else []) + folder_images(folders)),
    ):
        for k, path, n, url in sorted(rows, key=lambda r: r[0]):
            if k in have:
                continue
            have.add(k)
            c.execute(
                "insert into docs (key, kind, path, doc, pages, src, origin) values (?,?,?,?,?,?,?)",
                (
                    k,
                    kind,
                    path,
                    nxt,
                    n,
                    url,
                    "acquire" if kind == "pdf" else "generated" if path.startswith("generated/") else "lens",
                ),
            )
            nxt += 1
            added[kind] += 1
    c.commit()
    return added
