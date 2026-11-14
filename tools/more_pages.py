"""More pages per document for the embedding set: which pages of the embedded documents to embed next, their 384 px shards, the Kaggle run
and the merge into the embedding files. Data goes to Layouts/archive/info/more_pages (WORK), never to the repo.

Embedded so far: pages 0 / 1 of every document (UVTM/emb_out: embeddings.npy float16 (n, 768), ids.npy [doc, page], manifest.jsonl).
Wanted: every page of a document with fewer than SHORT pages; of the others the BUDGET = max(5 + 10 % of the pages, 30) most different
pages (greedy farthest-first on cheap formatting features, seeded with the pages already embedded), never an embedded page and never an
empty one. Features: PDF text layout (blocks, lines, fonts, alignment, columns, density grid, pictures, rules) for digital pages; for
scans a thumbnail proxy (ink grid, row / column profiles, edges; a JPEG scan is decoded at 1/8 scale, so it costs a few ms).

  python tools/more_pages.py check              # page counts of the files against the manifest (pages deleted from scans?)
  python tools/more_pages.py plan               # features + selection -> WORK/{feats.npz, select.parquet}
  python tools/more_pages.py shards             # 384 px thumbnails of the selected pages -> WORK/npz/pages_*.npz (+ manifest.jsonl)
  python tools/more_pages.py upload [--per N]   # Kaggle datasets, N shards (~600 MB each) per dataset
  python tools/more_pages.py run                # script kernel on 2 x T4 (pagetype embed_kaggle), wait, download -> WORK/out
  python tools/more_pages.py merge              # WORK/out* -> emb_out/{embeddings,ids}.npy

Local jobs run with 6 workers at most; every long step takes a timeout from the caller (watchdog)."""
import argparse
import io
import json
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf

from dspages.hand.config import PATHS
from dspages.ingest.litefeat import farthest_first  # noqa: F401  (pages_topup / pages_extra: M.farthest_first)
from dspages.kaggle_api import api  # noqa: F401

sys.path.insert(0, "/run/media/p/D/BUISNESS/corpus_collecting")          # pagetype.thumbs / pack_npz (the embedding side's code)

UV = Path(PATHS.uvtm)
LAYOUTS = Path(PATHS.layouts)
EMB = UV / "emb_out"
WORK = LAYOUTS / "archive" / "info" / "more_pages"
POOL = LAYOUTS / "archive" / "info" / "diverse" / "pool.parquet"
SHORT, MIN_BUDGET, BASE_PAGES, EXTRA = 5, 30, 5, 0.10
WORKERS = 6
KEY = 100_000                                              # (doc, page) -> doc * KEY + page (pages of a file stay below it)
MAX_CANDIDATES = 400                                       # pages of one document looked at (strided over the rest of a longer one)
DOC_SECONDS = 120                                          # a document that takes longer is dropped from the features
TEXT_MIN_CHARS = 20                                        # fewer characters: the page is an image (the picture proxy is added)
EMPTY_INK = 0.002                                          # picture proxy: share of dark pixels below this and no text = empty page

pymupdf.TOOLS.mupdf_display_errors(False)


def original(name):
    """The embedding file as it was before the first merge (the backup once it exists): the merge always starts from it, so it can be repeated."""
    b = EMB / (name + ".before_more_pages")
    return b if b.exists() else EMB / name


def budget(pages):
    """Number of pages to embed in all (embedded ones included) for a document of `pages` pages."""
    return pages if pages < SHORT else int(max(BASE_PAGES + np.ceil(EXTRA * pages), MIN_BUDGET))


# ---------------------------------------------------------------------------------------------------------------- features
GRID = (8, 6)                                              # rows x columns of the density grids


def _grid(mask):
    h, w = mask.shape
    r, c = GRID
    return np.array([[mask[i * h // r:(i + 1) * h // r, j * w // c:(j + 1) * w // c].mean() for j in range(c)] for i in range(r)]).ravel()


def text_feats(pg):
    """Formatting features of the text, pictures and rules of a PDF page (no rendering); (vector, number of characters)."""
    W, H = pg.rect.width, pg.rect.height
    d = pg.get_text("dict", flags=pymupdf.TEXTFLAGS_DICT)
    sizes, bold, ital, mono, fonts, lines = [], [], [], [], set(), []
    n_img, img_area = 0, 0.0
    cells = np.zeros(GRID[0] * GRID[1])
    for b in d["blocks"]:
        if b["type"] == 1:
            n_img += 1
            x0, y0, x1, y1 = b["bbox"]
            img_area += max(0, x1 - x0) * max(0, y1 - y0) / (W * H)
            continue
        for ln in b["lines"]:
            x0, y0, x1, y1 = ln["bbox"]
            n = sum(len(s["text"].strip()) for s in ln["spans"])
            if not n:
                continue
            lines.append((x0 / W, y0 / H, x1 / W, y1 / H, n, sum(c.isdigit() for s in ln["spans"] for c in s["text"]),
                          sum(c.isupper() for s in ln["spans"] for c in s["text"])))
            for s in ln["spans"]:
                k = len(s["text"].strip())
                if k:
                    sizes += [(s["size"], k)]
                    bold += [(bool(s["flags"] & 16), k)]
                    ital += [(bool(s["flags"] & 2), k)]
                    mono += [(bool(s["flags"] & 8), k)]
                    fonts.add(s["font"])
            gx0, gx1 = int(np.clip(x0 / W, 0, .999) * GRID[1]), int(np.clip(x1 / W, 0, .999) * GRID[1])
            gy = int(np.clip((y0 + y1) / 2 / H, 0, .999) * GRID[0])
            cells[gy * GRID[1] + gx0:gy * GRID[1] + gx1 + 1] += n / (gx1 - gx0 + 1)
    chars = int(sum(l[4] for l in lines))
    f = np.zeros(40)
    f[0], f[1] = np.log1p(W / H * 10), n_img
    f[2] = min(img_area, 1)
    try:
        dr = pg.get_cdrawings()
    except Exception:
        dr = []
    f[3] = np.log1p(len(dr))
    if dr:
        wh = np.array([[min(r["rect"][2] - r["rect"][0], 1e4) / W, min(r["rect"][3] - r["rect"][1], 1e4) / H] for r in dr[:3000]])
        f[4] = (wh.min(1) < 0.004).mean()                  # thin rules / lines
        f[5] = ((wh > .5).all(1)).mean()                   # page-sized fills and frames
        f[6] = np.mean([r.get("fill") is not None for r in dr[:3000]])
    if not lines:
        return np.concatenate([f, cells]), 0
    L = np.array(lines)
    w = L[:, 2] - L[:, 0]
    sz = np.array(sizes)
    ssz = np.average(sz[:, 0], weights=sz[:, 1])
    f[7], f[8], f[9] = np.log1p(len(L)), np.log1p(chars), np.log1p(len(d["blocks"]))
    f[10] = L[:, 0].min()
    f[11], f[12], f[13] = L[:, 1].min(), L[:, 2].max(), L[:, 3].max()      # content box
    f[14] = (w * (L[:, 3] - L[:, 1])).sum()                                 # text area share
    f[15], f[16] = w.mean(), w.std()
    f[17], f[18] = ssz, np.sqrt(np.average((sz[:, 0] - ssz) ** 2, weights=sz[:, 1]))
    f[19], f[20] = sz[:, 0].max() / max(ssz, 1), sz[:, 0].min() / max(ssz, 1)
    f[21] = len(set(np.round(sz[:, 0])))
    f[22] = len(fonts)
    for i, v in ((23, bold), (24, ital), (25, mono)):
        a = np.array(v, float)
        f[i] = np.average(a[:, 0], weights=a[:, 1])
    f[26] = L[:, 5].sum() / max(chars, 1)                                  # digits
    f[27] = L[:, 6].sum() / max(chars, 1)                                  # capitals
    left = np.round(L[:, 0] * 50)                                          # left edges in 2 % steps
    u, cnt = np.unique(left, return_counts=True)
    p = cnt / cnt.sum()
    f[28], f[29] = len(u), -(p * np.log(p)).sum()
    mid = (L[:, 0] + L[:, 2]) / 2
    f[30] = (np.abs(mid - .5) < .02).mean()                                # centred lines
    f[31] = (L[:, 2] > .93).mean() if len(L) else 0                        # reach the right margin (justified)
    cols = np.histogram(mid, bins=10, range=(0, 1))[0] > max(2, 0.08 * len(L))
    f[32] = cols.sum()
    y = np.sort((L[:, 1] + L[:, 3]) / 2)
    gaps = np.diff(y)
    if len(gaps):
        f[33], f[34] = gaps.mean(), gaps.std()
        f[35] = (gaps > 3 * np.median(gaps) + 1e-3).mean() if len(gaps) > 2 else 0      # paragraph breaks
    f[36] = (L[:, 4] < 12).mean()                                          # short lines (tables, forms, titles)
    f[37] = L[:, 4].mean() / 100
    f[38] = (w < .15).mean()
    f[39] = len(set(np.round(L[:, 0] * 100))) / max(len(L), 1)
    cells = cells / max(cells.sum(), 1) * 10
    return np.concatenate([f, cells]), chars


def proxy_feats(g):
    """Features of a gray thumbnail (uint8, long side ~ 96 px): ink grid, profiles, intensity histogram, edges; (vector, ink share)."""
    g = g.astype(np.float32)
    bg = np.median(g)
    ink = (bg - g) > 40
    hh, ww = g.shape
    rows, cols = ink.mean(1), ink.mean(0)
    gx, gy = np.abs(np.diff(g, axis=1)).mean(), np.abs(np.diff(g, axis=0)).mean()
    hist = np.histogram(g, bins=8, range=(0, 255))[0] / g.size
    f = [ink.mean(), g.mean() / 255, g.std() / 255, bg / 255, gx / 255, gy / 255, ww / hh,
         rows.std(), cols.std(), (rows > .02).mean(), (cols > .02).mean(),
         np.abs(np.diff(rows)).mean(), np.abs(np.diff(cols)).mean(),
         ink[:, :ww // 8 + 1].mean(), ink[:, -ww // 8 - 1:].mean(), ink[:hh // 8 + 1].mean(), ink[-hh // 8 - 1:].mean(),
         g[:, :ww // 16 + 1].mean() / 255, g[:, -ww // 16 - 1:].mean() / 255]
    rp = np.add.reduceat(rows, np.linspace(0, hh, 17, dtype=int)[:-1]) / (hh / 16)
    return np.concatenate([f, hist, _grid(ink) * 4, rp]).astype(np.float32), float(ink.mean())


def scan_thumb(doc, pg, side=96):
    """Gray thumbnail of a page: a single JPEG picture decoded at a reduced scale, else a small render."""
    from PIL import Image
    imgs = pg.get_images(full=True)
    if len(imgs) == 1 and not imgs[0][1] and doc.xref_get_key(imgs[0][0], "Filter")[1] == "/DCTDecode" and not pg.rotation:
        try:
            with Image.open(io.BytesIO(doc.xref_stream_raw(imgs[0][0]))) as jpg:
                if jpg.mode in ("L", "RGB"):
                    jpg.draft("L", (side * 2, side * 2))
                    a = np.asarray(jpg.convert("L"))
                    z = side / max(a.shape)
                    import cv2
                    return cv2.resize(a, (max(8, round(a.shape[1] * z)), max(8, round(a.shape[0] * z))), interpolation=cv2.INTER_AREA)
        except Exception:
            pass
    z = side / max(pg.rect.width, pg.rect.height)
    pm = pg.get_pixmap(matrix=pymupdf.Matrix(z, z), colorspace=pymupdf.csGRAY, alpha=False)
    return np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w)


NT, NP = 40 + GRID[0] * GRID[1], 19 + 8 + GRID[0] * GRID[1] + 16
DIM = NT + NP + 2                                          # text block | picture proxy block | has text, has proxy


class Timeout(Exception):
    pass


def _alarm(*_):
    raise Timeout


def doc_feats(args):
    """(doc, path, scan, pages to look at) -> (doc, {page: vector}, error): features of the listed pages of one PDF."""
    doc_id, path, scan, pages = args
    out, err = {}, ""
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(DOC_SECONDS)
    try:
        with pymupdf.open(path) as doc:
            for p in pages:
                if p >= len(doc):
                    continue
                pg = doc[p]
                v = np.zeros(DIM, np.float32)
                try:
                    t, chars = text_feats(pg)
                    v[:NT], v[-2] = t, chars >= TEXT_MIN_CHARS
                except Exception as e:
                    chars, err = 0, f"text {type(e).__name__}"
                if scan or chars < TEXT_MIN_CHARS:
                    try:
                        q, ink = proxy_feats(scan_thumb(doc, pg))
                        v[NT:NT + NP], v[-1] = q, 1
                        if chars < TEXT_MIN_CHARS and ink < EMPTY_INK:
                            continue                                 # an empty page: not a candidate
                    except Exception as e:
                        err = f"proxy {type(e).__name__}"
                        if chars < TEXT_MIN_CHARS:
                            continue
                out[p] = v
    except BaseException as e:                                       # Timeout, a broken file
        err = f"{type(e).__name__}: {e}"[:200]
    finally:
        signal.alarm(0)
    return doc_id, out, err


# ---------------------------------------------------------------------------------------------------------------- selection




# ---------------------------------------------------------------------------------------------------------------- documents
def _pages_of(path):
    try:
        if not path.lower().endswith(".pdf"):
            return 1
        with pymupdf.open(LAYOUTS / path) as d:
            return len(d)
    except Exception:
        return -1


def load_docs():
    """One row per embedded document: doc (emb id), path, scan, pages (manifest), actual (pages of the file now, -1 unreadable),
    emb (embedded pages), redo (the embedded pages are not pages 0 / 1 of the file as it is now: they are embedded again)."""
    P = pd.read_parquet(POOL, columns=["doc", "path", "scan", "pages", "folder"]).drop_duplicates("doc")
    ids = np.load(original("ids.npy"))
    emb = pd.DataFrame({"doc": ids[:, 0], "page": ids[:, 1]}).groupby("doc").page.apply(lambda s: tuple(sorted(s)))
    P = P.join(emb.rename("emb"), on="doc")
    f = WORK / "counts.parquet"
    if f.exists():
        P["actual"] = pd.read_parquet(f).set_index("doc").actual.reindex(P.doc).values
    else:
        with ProcessPoolExecutor(WORKERS) as ex:
            P["actual"] = list(ex.map(_pages_of, P.path, chunksize=200))
        WORK.mkdir(parents=True, exist_ok=True)
        P[["doc", "actual"]].to_parquet(f)
    P["emb"] = P.emb.map(lambda t: t if isinstance(t, tuple) else ())
    want = P.actual.map(lambda n: tuple(range(min(max(n, 0), 2))))
    P["redo"] = (P.emb != want) | ((P.pages != P.actual) & (P.pages < 500))
    return P


def cmd_check(_):
    P = load_docs()
    pdf = P.path.str.lower().str.endswith(".pdf")
    print(len(P), "documents;", (P.actual < 0).sum(), "unreadable;", int(P.redo.sum()), "with another page count than at the embedding")
    print(pd.crosstab(P.scan, P.redo))
    n = P.actual.clip(lower=0)
    print("embedded pages beyond the file's end:", int(sum(any(p >= a for p in e) for e, a in zip(P.emb, n))))
    print("pdf docs", int(pdf.sum()))


def candidates(r):
    """Pages of document row r to choose from (not embedded, unless the embedding is replaced), strided when there are many."""
    done = set() if r.redo else set(r.emb)
    pool = [p for p in range(int(r.actual)) if p not in done]
    return pool, sorted(done)


# ---------------------------------------------------------------------------------------------------------------- plan
FEAT_CHUNK = 400


def cmd_plan(a):
    P = load_docs()
    P = P[(P.actual > 0) & P.path.str.lower().str.endswith(".pdf")]
    rows, jobs = [], []                                             # rows: chosen (doc, page); jobs: documents that need features
    for r in P.itertuples():
        pool, done = candidates(r)
        k = budget(r.actual) if r.actual >= SHORT else len(pool)
        if len(pool) <= k:
            rows += [(r.doc, p) for p in pool]
            continue
        if len(pool) > MAX_CANDIDATES:
            pool = [pool[i] for i in np.unique(np.linspace(0, len(pool) - 1, MAX_CANDIDATES).round().astype(int))]
        jobs.append((r.doc, str(LAYOUTS / r.path), bool(r.scan), done + pool))
    print(len(rows), "pages of the documents taken whole;", len(jobs), "documents to choose pages in,",
          sum(len(j[3]) for j in jobs), "pages to look at", flush=True)
    fd = WORK / "feats"
    fd.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS) as ex:
        for c in range(0, len(jobs), FEAT_CHUNK):
            f = fd / f"{c // FEAT_CHUNK:04d}.npz"
            if f.exists():
                continue
            res = list(ex.map(doc_feats, jobs[c:c + FEAT_CHUNK], chunksize=4))
            key = [d * KEY + p for d, o, _ in res for p in o]
            V = np.stack([v for _, o, _ in res for v in o.values()]) if key else np.zeros((0, DIM), np.float32)
            err = {int(d): e for d, _, e in res if e}
            np.savez(f"{f}.tmp.npz", key=np.array(key, np.int64), V=V, err=json.dumps(err))
            f.with_name(f.name + ".tmp.npz").rename(f)
            print(f"features {c + len(res)}/{len(jobs)} docs, {int(time.time() - t0)} s, {len(err)} with errors", flush=True)
    select(P, jobs, rows)


def select(P, jobs, rows):
    keys, Vs, err = [], [], {}
    for f in sorted((WORK / "feats").glob("*.npz")):
        z = np.load(f)
        keys.append(z["key"]), Vs.append(z["V"])
        err.update(json.loads(str(z["err"])))
    key, V = np.concatenate(keys), np.concatenate(Vs)
    V = V.copy()
    for lo, hi, flag, wts in ((0, NT, DIM - 2, np.r_[np.ones(40), np.full(GRID[0] * GRID[1], .4)]),
                              (NT, NT + NP, DIM - 1, np.r_[np.ones(27), np.full(GRID[0] * GRID[1], .4), np.full(16, .4)])):
        m = V[:, flag] > 0
        mu, sd = V[m, lo:hi].mean(0), V[m, lo:hi].std(0) + 1e-6
        V[:, lo:hi] = np.where(m[:, None], np.clip((V[:, lo:hi] - mu) / sd, -5, 5) * wts, 0)
    V[:, -2:] *= 3                                                 # a page with text and a page that is a picture are far apart
    at = {int(k): i for i, k in enumerate(key)}
    info = P.set_index("doc")
    fallback = 0
    for doc, path, scan, look in jobs:
        r = info.loc[doc]
        done = set() if r.redo else set(r.emb)
        k = budget(r.actual)
        have = [p for p in look if doc * KEY + p in at and p not in done]
        seeds = [V[at[doc * KEY + p]] for p in done if doc * KEY + p in at]
        if len(have) <= k:                                         # features failed (timeout, broken file): evenly spread pages
            if len(have) < k:
                fallback += 1
                have = look if not have else have
                have = [have[i] for i in np.unique(np.linspace(0, len(have) - 1, min(k, len(have))).round().astype(int))]
            rows += [(doc, p) for p in have]
            continue
        M = V[[at[doc * KEY + p] for p in have]]
        rows += [(doc, have[i]) for i in farthest_first(M, seeds, k)]
    S = pd.DataFrame(rows, columns=["doc", "page"]).sort_values(["doc", "page"])
    S = S.join(P.set_index("doc")[["path", "scan", "redo", "actual"]], on="doc")
    S.to_parquet(WORK / "select.parquet")
    print(len(S), "pages of", S.doc.nunique(), "documents;", int(S.redo.sum()), "pages of documents embedded again;",
          fallback, "documents without usable features;", len(err), "errors", "->", WORK / "select.parquet")


# ---------------------------------------------------------------------------------------------------------------- shards
SHARD_PAGES = 20_000                                       # pages of one pages_NNNN.npz (one sequential read on Kaggle)
NPZ = WORK / "npz"
DATASET, KERNEL = "pavel9860/more-pages-npz", "pavel9860/more-pages-embed"


def render_job(args):
    """(path, [(doc, page)]) -> [(doc, page, jpeg)]: 384 px thumbnails like pagetype.thumbs (scan pictures decoded, others rendered)."""
    from pagetype import thumbs as T
    path, items = args
    out = []
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(DOC_SECONDS * 3)
    try:
        with pymupdf.open(path) as doc:
            for d, p in items:
                try:
                    pg = doc[p]
                    rgb = T.scan_image(doc, pg)
                    if rgb is None:
                        z = T.WORK / max(pg.rect.width, pg.rect.height)
                        pm = pg.get_pixmap(matrix=pymupdf.Matrix(z, z), colorspace=pymupdf.csRGB, alpha=False)
                        rgb = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, 3)
                    out.append((d, p, T.encode_jpeg(T.fit(rgb))))
                except Timeout:
                    raise
                except Exception:
                    pass
    except BaseException:
        pass
    finally:
        signal.alarm(0)
    return out


def run_jobs(jobs):
    """render_job over jobs on WORKERS processes; a crash of the pool (a file that kills MuPDF) splits the jobs until the file is found
    and skipped."""
    from concurrent.futures.process import BrokenProcessPool
    try:
        with ProcessPoolExecutor(WORKERS if len(jobs) > 1 else 1) as ex:
            return [x for r in ex.map(render_job, jobs, chunksize=8) for x in r]
    except BrokenProcessPool:
        if len(jobs) == 1:
            print("skipped (crash):", jobs[0][0], flush=True)
            return []
        h = len(jobs) // 2
        return run_jobs(jobs[:h]) + run_jobs(jobs[h:])


def cmd_shards(a):
    from pagetype.pack_npz import write_shard
    S = pd.read_parquet(WORK / "select.parquet").reset_index(drop=True)
    NPZ.mkdir(parents=True, exist_ok=True)
    n_sh = -(-len(S) // SHARD_PAGES)
    t0 = time.time()
    if True:
        for s in range(n_sh):
            f = NPZ / f"pages_{s:04d}.npz"
            if f.exists():
                continue
            part = S.iloc[s * SHARD_PAGES:(s + 1) * SHARD_PAGES]
            jobs = [(str(LAYOUTS / g.path.iloc[0]), list(zip(g.doc, g.page))) for _, g in part.groupby("doc", sort=True)]
            got = run_jobs(jobs)
            got.sort(key=lambda x: (x[0], x[1]))
            write_shard(f, [(d, p) for d, p, _ in got], [j for *_, j in got])
            print(f"{f.name}: {len(got)}/{len(part)} pages, {round(sum(len(j) for *_, j in got) / 1e9, 2)} GB, {int(time.time() - t0)} s", flush=True)
    sizes = [f.stat().st_size for f in NPZ.glob("pages_*.npz")]
    print(len(sizes), "shards", round(sum(sizes) / 1e9, 1), "GB")


# ---------------------------------------------------------------------------------------------------------------- Kaggle


def parts(per_part):
    files = sorted(NPZ.glob("pages_*.npz"))
    return [files[i:i + per_part] for i in range(0, len(files), per_part)]


def cmd_upload(a):
    """One dataset per part (a.per shards): <DATASET>-<k>. Existing ones with all their files are left alone."""
    import shutil
    k = api()
    for i, files in enumerate(parts(a.per)):
        ref = f"{DATASET}-{i}"
        folder = WORK / f"upload_{i}"
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir()
        for f in files:
            (folder / f.name).hardlink_to(f)
        (folder / "dataset-metadata.json").write_text(json.dumps({"title": f"more-pages-npz-{i}", "id": ref, "licenses": [{"name": "CC0-1.0"}],
                                                                  "isPrivate": True}))
        try:
            have = {x.name for x in k.dataset_list_files(ref).files}
        except Exception:
            have = set()
        if {f.name for f in files} <= have:
            print(ref, "already there", flush=True)
            continue
        t0 = time.time()
        try:
            k.dataset_status(ref)
            k.dataset_create_version(str(folder), version_notes=time.strftime("%Y-%m-%d"), dir_mode="skip", quiet=False)
        except Exception:
            k.dataset_create_new(str(folder), dir_mode="skip", quiet=False)
        for _ in range(90):                                        # the status endpoint answers 403 for a while after an upload
            try:
                if {f.name for f in files} <= {x.name for x in k.dataset_list_files(ref).files}:
                    break
            except Exception:
                pass
            time.sleep(20)
        else:
            raise SystemExit(f"{ref}: files not listed after the upload")
        print(ref, len(files), "shards uploaded in", int(time.time() - t0), "s", flush=True)
        shutil.rmtree(folder)


def cmd_test(a):
    """10 pages of the first shard -> a test dataset, a kernel run on it, the output checked (shape, finite)."""
    import shutil
    z = np.load(NPZ / "pages_0000.npz")
    off = z["offsets"]
    d = WORK / "upload_test"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir()
    from pagetype.pack_npz import write_shard
    write_shard(d / "pages_0000.npz", z["ids"][:10], [z["jpeg"][off[i]:off[i + 1]].tobytes() for i in range(10)])
    ref = "pavel9860/more-pages-test"
    (d / "dataset-metadata.json").write_text(json.dumps({"title": "more-pages-test", "id": ref, "licenses": [{"name": "CC0-1.0"}], "isPrivate": True}))
    k = api()
    try:
        k.dataset_status(ref)
        k.dataset_create_version(str(d), version_notes="test", dir_mode="skip", quiet=True)
    except Exception:
        k.dataset_create_new(str(d), dir_mode="skip", quiet=True)
    for _ in range(90):
        try:
            if k.dataset_list_files(ref).files:
                break
        except Exception:
            pass
        time.sleep(20)
    run_kernel([ref], "pavel9860/more-pages-test-embed", WORK / "out_test")
    v, i = np.load(WORK / "out_test" / "embeddings.npy"), np.load(WORK / "out_test" / "ids.npy")
    print("test output", v.shape, i.shape, np.isfinite(v.astype(np.float32)).all(), i[:3].tolist(), z["ids"][:3].tolist())


def run_kernel(datasets, kernel, out):
    import shutil
    k, folder = api(), WORK / "kernel"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir()
    shutil.copy("/run/media/p/D/BUISNESS/corpus_collecting/pagetype/embed_kaggle.py", folder / "embed_kaggle.py")
    (folder / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel, "title": kernel.split("/")[1], "code_file": "embed_kaggle.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "keywords": ["gpu"],
        "dataset_sources": datasets, "kernel_sources": [], "competition_sources": [], "model_sources": [],
        "machine_shape": "NvidiaTeslaT4"}))
    k.kernels_push(str(folder))
    t0 = time.time()
    while True:
        st = str(k.kernels_status(kernel).status).lower()
        print(f"[{int(time.time() - t0)} s] {st}", flush=True)
        if "complete" in st or "error" in st or "cancel" in st:
            break
        time.sleep(60)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    k.kernels_output(kernel, str(out))
    if "complete" not in st or not (out / "embeddings.npy").exists():
        raise SystemExit(f"kernel {st}: see {out}")


def cmd_run(a):
    run_kernel([f"{DATASET}-{i}" for i in range(len(parts(a.per)))], KERNEL, WORK / "out")


def cmd_local(a):
    """Pages of the selection that are in no shard (a render failed, a file crashed MuPDF) -> thumbnails -> SigLIP 2 on the CPU ->
    WORK/out_local/{embeddings,ids}.npy (merged with WORK/out by `merge`)."""
    import cv2
    import torch
    sys.path.insert(0, "/run/media/p/D/BUISNESS/corpus_collecting/pagetype")
    import embed_kaggle as EK
    S = pd.read_parquet(WORK / "select.parquet")
    have = np.concatenate([np.load(f)["ids"] for f in sorted(NPZ.glob("pages_*.npz"))]).astype(np.int64)
    miss = S[~np.isin(S.doc.astype(np.int64) * KEY + S.page, have[:, 0] * KEY + have[:, 1])]
    print(len(miss), "pages in no shard,", miss.doc.nunique(), "documents", flush=True)
    got = run_jobs([(str(LAYOUTS / g.path.iloc[0]), list(zip(g.doc, g.page))) for _, g in miss.groupby("doc")])
    done = {(d, p) for d, p, _ in got}
    import subprocess
    import tempfile
    from pagetype import thumbs as T
    for r in miss.itertuples():                                    # MuPDF crashes on these files: poppler renders them
        if (r.doc, r.page) in done:
            continue
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(["pdftoppm", "-f", str(r.page + 1), "-l", str(r.page + 1), "-scale-to", str(T.WORK), "-png",
                            str(LAYOUTS / r.path), f"{td}/p"], capture_output=True, timeout=120)
            im = next(Path(td).glob("p*.png"), None)
            rgb = cv2.imread(str(im)) if im is not None else None
            if rgb is not None:
                rgb = rgb[:, :, ::-1]
                got.append((r.doc, r.page, T.encode_jpeg(T.fit(np.ascontiguousarray(rgb)))))
    model = EK.load_model("cpu")
    vec = []
    with torch.inference_mode():
        for i in range(0, len(got), 16):
            x = np.stack([cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1] for *_, j in got[i:i + 16]])
            x = (torch.from_numpy(x.copy()).permute(0, 3, 1, 2).float() / 255 - 0.5) / 0.5
            o = model.get_image_features(pixel_values=x)
            vec.append(getattr(o, "pooler_output", o).numpy())
    out = WORK / "out_local"
    out.mkdir(exist_ok=True)
    np.save(out / "embeddings.npy", np.concatenate(vec).astype(np.float16) if vec else np.zeros((0, 768), np.float16))
    np.save(out / "ids.npy", np.array([(d, p) for d, p, _ in got], np.int32).reshape(-1, 2))
    print(len(got), "pages embedded locally;", len(miss) - len(got), "failed to render")


def cmd_merge(a):
    """WORK/out -> emb_out: rows of documents embedded again replace their old rows (also stale ones beyond the file's end); other
    new rows are added unless (doc, page) is there. Keys doc * KEY + page. The old files are kept once as *.before_more_pages."""
    import shutil
    outs = [d for d in (WORK / "out", WORK / "out_local") if (d / "ids.npy").exists()]
    vec = np.concatenate([np.load(d / "embeddings.npy") for d in outs])
    ids = np.concatenate([np.load(d / "ids.npy") for d in outs])
    old_v, old_i = np.load(original("embeddings.npy")), np.load(original("ids.npy"))
    S = pd.read_parquet(WORK / "select.parquet")
    P = load_docs()
    key = lambda x: x[:, 0].astype(np.int64) * KEY + x[:, 1]
    sel = set((S.doc.astype(np.int64) * KEY + S.page).tolist())
    ok = np.isfinite(vec.astype(np.float32)).all(1) & np.isin(key(ids), list(sel))
    _, first = np.unique(key(ids), return_index=True)
    ok &= np.isin(np.arange(len(ids)), first)
    vec, ids = vec[ok], ids[ok]
    redo = set(P.doc[P.redo].tolist())
    drop = np.isin(old_i[:, 0], list(redo)) | np.isin(key(old_i), key(ids))
    all_v, all_i = np.concatenate([old_v[~drop], vec]), np.concatenate([old_i[~drop], ids])
    order = np.lexsort((all_i[:, 1], all_i[:, 0]))
    for name in ("embeddings.npy", "ids.npy"):
        bak = EMB / (name + ".before_more_pages")
        if not bak.exists():
            shutil.copy(EMB / name, bak)
    for name, arr in (("embeddings.npy", all_v[order]), ("ids.npy", all_i[order])):
        tmp = EMB / (name + ".tmp.npy")
        np.save(tmp, arr)
        tmp.replace(EMB / name)
    print(f"selected {len(S)}, embedded {len(vec)} new rows ({(~ok).sum()} of the output dropped); old rows replaced {int(drop.sum())}; "
          f"total {len(all_v)} pages of {len(np.unique(all_i[:, 0]))} documents")


COMMANDS = {"check": cmd_check, "plan": cmd_plan, "shards": cmd_shards, "upload": cmd_upload, "run": cmd_run, "test": cmd_test, "local": cmd_local, "merge": cmd_merge}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=COMMANDS)
    ap.add_argument("--per", type=int, default=8, help="shards per Kaggle dataset (~5 GB)")
    a = ap.parse_args()
    COMMANDS[a.step](a)
