"""Does a crawl aimed at a gap region of the 100K set (tools/ds100k.py gaps) bring pages where the region is? Expected embedding = the
region's centroid; real = the embedding of the downloaded pages (same render as the shards: tools/form_pages.py, same model and vector as
the Kaggle embedding: corpus_collecting pagetype/embed_kaggle.py, on the CPU). Data: OUT/probe.

  cands     per region: other PDFs (local Common Crawl PDF index) of the hosts its documents came from (src_urls, untyped_sources);
            a baseline of random index PDFs -> OUT/probe/cands.parquet
  fetch     downloads (live site, else the Common Crawl capture), the first PAGES pages that render -> OUT/probe/pages.parquet + npz of jpegs
  embed     1536-d vectors -> PCA space of the pool -> distance to the target centroid / to the nearest pool page -> OUT/probe/result.parquet

python tools/ds_probe.py cands [--per 5] [--base 20] | fetch | embed"""

import argparse
import json
import random
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

import diverse_ds as DD
import ds100k as D
import form_pages as F
import numpy as np
import pandas as pd

from dspages import paths as _P  # noqa: E402

CC = f"{_P.CORPUS}"
sys.path.insert(0, CC)
WORK = D.OUT / "probe"
PAGES, HOST_MAX = 2, 2
HEX = re.compile(r"(?<![0-9a-f])[0-9a-f]{12}(?![0-9a-f])")


def host(url):
    return (urlsplit(url).hostname or "").removeprefix("www.")


def source_urls():
    """{document hash: source url} of the crawled documents."""
    with sqlite3.connect(f"file:{D.PATHS.manifest}?mode=ro", uri=True) as db:
        u = dict(db.execute("select hash, url from src_urls"))
    for r in map(json.loads, open(D.LAYOUTS / "archive" / "info" / "untyped_sources.jsonl")):
        if r.get("url"):
            u.setdefault(r["file"][:12], r["url"])
    return u


def cmd_cands(a):
    import duckdb

    T, _R = pd.read_parquet(D.OUT / "thin.parquet"), pd.read_parquet(D.OUT / "regions.parquet")
    urls = source_urls()
    T["url"] = T.h.map(urls)
    T = T[T.url.notna()].assign(host=lambda t: t.url.map(host))
    print(f"thin pages with a source url: {len(T)} of {T.region.nunique()} regions")
    hosts = sorted(set(T.host))
    con = duckdb.connect()
    con.execute("create temp table h as select unnest(?) as host", [hosts])
    idx = f"'{CC}/cc_pdf_index/*/*.parquet'"
    rows = con.execute(f"""select url, warc_filename f, warc_record_offset o, warc_record_length l,
                           regexp_replace(regexp_extract(url, '^https?://([^/:]+)', 1), '^www\\.', '') host
                           from {idx} where host in (select host from h)""").df()
    known = set(urls.values())
    rows = rows[~rows.url.isin(known)].drop_duplicates("url")
    random.Random(0)
    out = []
    for r, g in T.groupby("region"):
        c = rows[rows.host.isin(set(g.host))].sample(frac=1, random_state=r)
        c = c.groupby("host").head(HOST_MAX)
        out.append(c.head(a.per * 3).assign(region=r))  # 3 x: a share fails to download or render
    base = con.execute(
        f"select url, warc_filename f, warc_record_offset o, warc_record_length l from {idx} using sample {a.base * 3} rows"
    ).df()
    C = pd.concat(out + [base.assign(region=-1, host=base.url.map(host))], ignore_index=True)
    WORK.mkdir(parents=True, exist_ok=True)
    C.to_parquet(WORK / "cands.parquet")
    print(C.groupby("region").size().to_dict(), "candidates ->", WORK / "cands.parquet")


def fetch_one(r):
    from corpus import discovery

    data = None
    try:
        data = discovery.fetch_live(r["url"])
    except Exception:
        pass
    if not data:
        try:
            data = discovery.fetch(r)
        except Exception:
            return None
    try:
        import pymupdf

        with pymupdf.open(stream=data, filetype="pdf") as doc:
            out = []
            for i in range(min(doc.page_count, 6)):
                pg = doc[i]
                pm = pg.get_pixmap(
                    matrix=pymupdf.Matrix(F.W / pg.rect.width, F.H / pg.rect.height),
                    colorspace=pymupdf.csRGB,
                    alpha=False,
                )
                out.append((i, F.encode(np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, 3))))
                if len(out) == PAGES:
                    break
            return out
    except Exception:
        return None


def cmd_fetch(a):
    C = pd.read_parquet(WORK / "cands.parquet")
    need = {r: (a.per if r >= 0 else a.base) for r in C.region.unique()}
    got, rows, jpg = {r: 0 for r in need}, [], []
    with ThreadPoolExecutor(8) as ex:
        for (_, r), pages in zip(C.iterrows(), ex.map(fetch_one, [r for _, r in C.iterrows()])):
            if not pages or got[r.region] >= need[r.region]:
                continue
            got[r.region] += 1
            for no, j in pages:
                rows.append(dict(region=r.region, url=r.url, host=r.host, page=no, k=len(jpg)))
                jpg.append(j)
    pd.DataFrame(rows).to_parquet(WORK / "pages.parquet")
    np.savez(WORK / "jpegs.npz", *[np.frombuffer(j, np.uint8) for j in jpg])
    print("documents per region", got, "|", len(rows), "pages")


def cmd_embed(_):
    import torch

    sys.path.insert(0, f"{CC}/pagetype")
    import embed_kaggle as K

    torch.set_num_threads(6)
    P = pd.read_parquet(WORK / "pages.parquet")
    z = np.load(WORK / "jpegs.npz")
    raw = [torch.from_numpy(z[f"arr_{k}"].copy()) for k in P.k]
    model = K.load_model("cpu")
    V = np.concatenate([K.embed_pages(model, raw[a : a + 4], "cpu")[0] for a in range(0, len(raw), 4)])
    pca = np.load(D.OUT / "pca.npz")
    Zp = (DD.unit_rows(V) - pca["mean"]) @ pca["components"].T
    cen = np.load(D.OUT / "regions.npz")["centers"]
    R = pd.read_parquet(D.OUT / "regions.parquet")
    Zpool = np.load(D.OUT / "Z.npy")
    S = pd.read_parquet(D.OUT / "select.parquet")
    _, d_pool = D.nearest(Zp.astype(np.float32), Zpool)
    near_r, _ = D.nearest(Zp.astype(np.float32), cen)
    P["d_target"] = [np.linalg.norm(Zp[i] - cen[r]) if r >= 0 else np.nan for i, r in enumerate(P.region)]
    P["d_all"] = [np.linalg.norm(cen - Zp[i], axis=1) for i in range(len(P))]
    P["target_radius"] = P.region.map(R.radius)
    P["nearest_region"] = near_r
    P["d_pool"] = d_pool
    r_set = float(S.r.iloc[-1])
    P["novel"] = P.d_pool > r_set
    P.drop(columns="d_all").to_parquet(WORK / "result.parquet")
    np.save(WORK / "vectors.npy", V.astype(np.float16))
    t = P[P.region >= 0]
    base = P[P.region < 0]
    d_base = np.array(
        [np.linalg.norm(cen - Zp[i], axis=1) for i in base.index]
    )  # baseline page to every region centroid
    print(f"set spacing (final FPS radius) {r_set:.3f}")
    print(
        f"targeted pages {len(t)}: distance to the target centroid median {t.d_target.median():.3f}; "
        f"region radius median {t.target_radius.median():.3f}; within the radius {(t.d_target <= t.target_radius).mean():.2f}; "
        f"nearest region = target {(t.nearest_region == t.region).mean():.2f} (chance {1 / len(cen):.2f})"
    )
    print(
        f"baseline pages {len(base)}: distance to a region centroid median {np.median(d_base):.3f}; within that region's radius "
        f"{(d_base <= R.radius.values[None]).mean():.3f}"
    )
    print(
        f"novel (farther than the set spacing from every pool page): targeted {t.novel.mean():.2f}, baseline {base.novel.mean():.2f}"
    )
    print(
        t.assign(hit=t.nearest_region == t.region)
        .groupby("region")
        .agg(
            pages=("url", "size"),
            d_target=("d_target", "median"),
            radius=("target_radius", "first"),
            hit=("hit", "mean"),
        )
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for n in ("cands", "fetch"):
        p = sp.add_parser(n)
        p.add_argument("--per", type=int, default=5)
        p.add_argument("--base", type=int, default=20)
    sp.add_parser("embed")
    a = ap.parse_args()
    {"cands": cmd_cands, "fetch": cmd_fetch, "embed": cmd_embed}[a.cmd](a)
