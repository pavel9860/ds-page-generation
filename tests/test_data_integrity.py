"""Integrity of the corpus data the page sets are built from: the corpus manifest (Layouts/manifest.sqlite), the page embeddings (emb_out),
fill / pic (scan_pct: the detector on the shards, the only fill / pic source) and the text / handwriting boxes (hw_det + the detection shards). Read-only, against the real data; skipped when it is
not on this machine. Disk checks use fixed random samples.

  pytest tests/test_data_integrity.py -n 0"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf
import pytest

from dspages.hand.config import PATHS

LAYOUTS = Path(PATHS.layouts)
EMB = Path(PATHS.uvtm) / "Labels" / "embedding" / "emb_out"
HW = Path(PATHS.uvtm) / "Labels" / "hw_detection"
INFO = LAYOUTS / "archive" / "info"
DB = LAYOUTS / "manifest.sqlite"
DIM, SAMPLE = 1536, 500

pytestmark = pytest.mark.skipif(
    not (DB.exists() and (EMB / "ids.npy").exists()), reason="corpus data not on this machine"
)


def keys(doc, page):
    return np.asarray(doc, np.int64) * 100000 + np.asarray(page, np.int64)


@pytest.fixture(scope="module")
def db():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    yield con
    con.close()


@pytest.fixture(scope="module")
def man():
    return pd.DataFrame([json.loads(l) for l in open(EMB / "manifest.jsonl")]).rename(columns={"id": "doc"})


@pytest.fixture(scope="module")
def ids():
    return np.load(EMB / "ids.npy")


@pytest.fixture(scope="module")
def pages(ids, man):
    """One row per embedding row: row, doc, page, path."""
    return pd.DataFrame({"row": np.arange(len(ids)), "doc": ids[:, 0], "page": ids[:, 1]}).merge(
        man, on="doc", how="left"
    )


def sample(df, n=SAMPLE):
    return df.sample(min(n, len(df)), random_state=0)


# ---- corpus manifest ----
def test_manifest_one_row_per_document(db):
    for t in ("plan", "docs"):
        n, d = db.execute(f"select count(*), count(distinct doc) from {t}").fetchone()
        assert n == d, f"{t}: {n - d} repeated documents"
    n, p = db.execute("select count(*), count(distinct new_path) from plan").fetchone()
    assert n == p, f"plan: {n - p} documents share a file"


def test_manifest_files_exist(db):
    paths = [r[0] for r in db.execute("select new_path from plan")]
    miss = [p for p in sample(pd.Series(paths)) if not (LAYOUTS / p).is_file()]
    assert not miss, f"{len(miss)} of {SAMPLE} plan files missing, e.g. {miss[:3]}"


# ---- embeddings ----
def test_embedding_rows(ids):
    E = np.load(EMB / "embeddings.npy", mmap_mode="r")
    assert E.shape == (len(ids), DIM)
    rows = np.sort(np.random.default_rng(0).choice(len(E), 5000, replace=False))
    assert np.isfinite(np.asarray(E[rows], np.float32)).all()


def test_embedding_ids_unique(ids, man):
    k = keys(ids[:, 0], ids[:, 1])
    assert len(np.unique(k)) == len(k), "a (doc, page) is embedded twice"
    assert man.doc.is_unique
    assert np.isin(ids[:, 0], man.doc).all(), "embedding rows of documents the embedding manifest does not list"


def test_embedding_one_id_per_file(man, ids):
    m = man[man.doc.isin(np.unique(ids[:, 0]))]
    rep = m[m.path.duplicated(keep=False)]
    assert rep.empty, f"{rep.path.nunique()} files under several document ids ({len(rep)} ids), e.g. {rep.path.iloc[0]}"


def test_embedding_no_repeated_pages(pages):
    rep = pages.duplicated(["path", "page"]).sum()
    assert rep == 0, f"{rep} embedding rows repeat a (file, page) under another document id"


def test_embedding_files_and_pages_exist(pages):
    bad = []
    for path, g in sample(pages).groupby("path"):
        f = LAYOUTS / path
        if not f.is_file():
            bad.append((path, "missing"))
            continue
        n = pymupdf.open(f).page_count if path.lower().endswith(".pdf") else 1
        bad += [(path, p) for p in g.page if p >= n]
    assert not bad, f"{len(bad)} sampled rows whose file or page is gone, e.g. {bad[:3]}"


def test_embedding_matches_a_fresh_render():
    """tools/emb_check.py: a random sample rendered and embedded again; the stored vector must be the page's."""
    f = INFO / "emb_check" / "compare.parquet"
    if not f.exists():
        pytest.skip("no emb_check run")
    c = pd.read_parquet(f)
    assert c.cos.min() > 0.99, f"{int((c.cos <= 0.99).sum())} of {len(c)} pages differ from a fresh embedding"


def test_embedding_manifest_matches_corpus_manifest(man, db, ids):
    """The file of an embedded document is the corpus manifest's file of its hash."""
    plan = dict(db.execute("select doc, new_path from plan"))
    m = man[man.doc.isin(np.unique(ids[:, 0]))].copy()
    m["h"] = m.path.str.extract(r"(?<![0-9a-f])([0-9a-f]{12})(?![0-9a-f])")[0]
    m = m[m.h.isin(plan)]
    off = m[m.path != m.h.map(plan)]
    assert off.empty, (
        f"{len(off)} documents filed under another file than the manifest's, e.g. {off.path.iloc[0]} vs {plan[off.h.iloc[0]]}"
    )


# ---- fill / pic ----
def test_scan_pct_covers_the_embedding_exactly(ids):
    s = pd.read_parquet(INFO / "scan_pct.parquet", columns=["doc", "page", "fill", "pic"])
    ks, ke = keys(s.doc, s.page), keys(ids[:, 0], ids[:, 1])
    assert len(np.unique(ks)) == len(ks), "a page has two fill / pic rows"
    assert len(np.setdiff1d(ke, ks)) == 0, f"{len(np.setdiff1d(ke, ks))} embedded pages without fill / pic"
    assert len(np.setdiff1d(ks, ke)) == 0, f"{len(np.setdiff1d(ks, ke))} fill / pic rows of pages not embedded"
    assert s.fill.isna().sum() <= 10, f"{int(s.fill.isna().sum())} pages without a value"
    v = s.dropna()
    assert v.fill.between(0, 1).all() and v.pic.between(0, 1).all()


def test_scan_pct_table_matches_parquet(db):
    s = pd.read_parquet(INFO / "scan_pct.parquet", columns=["hash", "page"]).dropna().drop_duplicates()
    n = db.execute("select count(*) from scan_pct").fetchone()[0]
    assert n == len(s), f"manifest.sqlite scan_pct has {n} rows, scan_pct.parquet {len(s)} (hash, page)"


def test_scan_pct_not_older_than_the_embedding():
    assert (INFO / "scan_pct.parquet").stat().st_mtime >= (EMB / "ids.npy").stat().st_mtime, (
        "ids.npy changed after scan_pct was merged"
    )


# ---- text / handwriting boxes ----
@pytest.fixture(scope="module")
def hw(db):
    return pd.read_sql("select doc, page, text, hw, shard, src_doc from hw_det", db)


def test_hw_det_covers_the_embedding_exactly(hw, ids):
    kh, ke = keys(hw.doc, hw.page), keys(ids[:, 0], ids[:, 1])
    assert len(np.unique(kh)) == len(kh)
    assert len(np.setdiff1d(ke, kh)) == 0, f"{len(np.setdiff1d(ke, kh))} embedded pages without boxes"
    assert len(np.setdiff1d(kh, ke)) == 0, f"{len(np.setdiff1d(kh, ke))} box rows of pages not embedded"
    assert (hw.text >= 0).all() and (hw.hw >= 0).all()


def test_hw_det_matches_its_shards(hw):
    """The counts of a sample of pages, counted again from the shard each row names."""
    bad = []
    for shard, g in sample(hw, 2000).groupby("shard"):
        z = np.load(HW / shard)
        k = keys(z["ids"][:, 0], z["ids"][:, 1])
        pos = dict(zip(k.tolist(), range(len(k))))
        o, c = z["offsets"], z["cls"]
        for d, p, t, h in zip(g.src_doc, g.page, g.text, g.hw):  # src_doc: the page's id in the shard
            i = pos.get(int(d) * 100000 + int(p))
            if i is None:
                bad.append((shard, d, p, "not in shard"))
                continue
            b = c[o[i] : o[i + 1]]
            if (int((b == 0).sum()), int((b == 1).sum())) != (h, t):
                bad.append((shard, d, p, "counts"))
    assert not bad, f"{len(bad)} sampled rows disagree with their shard, e.g. {bad[:3]}"


def test_hw_index_lists_every_shard():
    idx = json.loads((HW / "index.json").read_text())["pages"]
    files = {f.name: len(np.load(f)["ids"]) for f in HW.glob("d*__pages_*.npz")}
    assert idx == files, "index.json and the shard files disagree"
