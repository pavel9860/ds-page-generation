"""768 x 1024 JPEG shards of the form pages (forms_blank / forms_filled probability > P_MIN by the pagetype head on the embeddings) and
their upload to Kaggle, each dataset as soon as its shards are built. Every page is resized to 768 x 1024 (the A4 ratio, no padding, no margin
crop: a page of another shape is stretched), JPEG quality 90. Pages the embedding file lists but the file does not have (page 1 of a
one-image file) are skipped; files MuPDF cannot open (webp, a broken pdf) are read by PIL / pdftoppm. Data goes to Layouts/archive/info/form_pages (WORK), never to the repo.

  python tools/form_pages.py select    # embeddings -> WORK/select.parquet (doc, page, path, pb, pf)
  python tools/form_pages.py build     # shards -> WORK/npz/forms_NNNN.npz, datasets pavel9860/form-pages-724x1024-<k> uploaded as they complete

Shard = ids int32 (n, 2) [doc, page] (emb_out ids), offsets int64 (n + 1), jpeg uint8 (page i = jpeg[offsets[i]:offsets[i + 1]]), the
layout of pagetype.pack_npz. A dataset holds SHARDS_PER_DATASET shards (~5 GB). 6 workers at most; build / upload skip what exists."""

import argparse
import io
import json
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pymupdf
from PIL import Image

from dspages import paths as _P  # noqa: E402
from dspages.hand.config import PATHS

sys.path.insert(0, f"{_P.CORPUS}")

LAYOUTS = Path(PATHS.layouts)
EMB = Path(PATHS.uvtm) / "emb_out"
WORK = LAYOUTS / "archive" / "info" / "form_pages"
NPZ = WORK / "npz"
P_MIN = 0.2
W, H, QUALITY = 768, 1024, 90
SHARD_PAGES, SHARDS_PER_DATASET = 12_000, 10  # ~2 GB a shard, ~20 GB a dataset
WORKERS = 6
DOC_SECONDS = 120
DATASET = "pavel9860/form-pages-724x1024"

pymupdf.TOOLS.mupdf_display_errors(False)
cv2.setNumThreads(1)


class Timeout(BaseException):
    pass


def _alarm(*_):
    raise Timeout


def cmd_select(_):
    from pagetype import config as C
    from pagetype.head import Head
    from pagetype.layouts import Layouts, load_manifest

    h = Head.load(C.MODEL_PATH)
    fb, ff = h.classes.index("forms_blank"), h.classes.index("forms_filled")
    E, ids = np.load(EMB / "embeddings.npy", mmap_mode="r"), np.load(EMB / "ids.npy")
    pb, pf = np.zeros(len(ids), np.float32), np.zeros(len(ids), np.float32)
    for a in range(0, len(ids), 100_000):
        p = h.predict_proba(np.asarray(E[a : a + 100_000], np.float32))
        pb[a : a + 100_000], pf[a : a + 100_000] = p[:, fb], p[:, ff]
    S = pd.DataFrame({"doc": ids[:, 0], "page": ids[:, 1], "pb": pb, "pf": pf})
    S = S[(S.pb > P_MIN) | (S.pf > P_MIN)].sort_values(["doc", "page"]).reset_index(drop=True)
    lay = Layouts(LAYOUTS, load_manifest(EMB / "manifest.jsonl"))
    path = {d: lay.path(int(d)) for d in S.doc.unique()}
    S["path"] = S.doc.map(path)
    print(len(S), "pages of", S.doc.nunique(), "documents;", int(S.path.isna().sum()), "pages without a file (dropped)")
    WORK.mkdir(parents=True, exist_ok=True)
    S[S.path.notna()].reset_index(drop=True).to_parquet(WORK / "select.parquet")


def resize(rgb):
    """Any page to W x H, no padding (a page rendered at that size is left as it is)."""
    if rgb.shape[:2] == (H, W):
        return rgb
    return cv2.resize(
        rgb, (W, H), interpolation=cv2.INTER_AREA if rgb.shape[0] * rgb.shape[1] > W * H else cv2.INTER_CUBIC
    )


def encode(rgb):
    """JPEG bytes, quality QUALITY, 4:2:0: PIL's libjpeg-turbo takes half the time of cv2.imencode for the same bytes."""
    buf = io.BytesIO()
    Image.fromarray(resize(rgb)).save(buf, "JPEG", quality=QUALITY)
    return buf.getvalue()


def fallback(path, d, p):
    """(doc, page, jpeg) by another reader: PIL for an image MuPDF refuses, pdftoppm for a pdf; None when that fails too."""
    import subprocess
    import tempfile

    try:
        if path.lower().endswith(".pdf"):
            with tempfile.TemporaryDirectory() as tmp:
                subprocess.run(
                    [
                        "pdftoppm",
                        "-jpeg",
                        "-f",
                        str(p + 1),
                        "-l",
                        str(p + 1),
                        "-scale-to-x",
                        str(W),
                        "-scale-to-y",
                        str(H),
                        path,
                        f"{tmp}/o",
                    ],
                    check=True,
                    timeout=120,
                    capture_output=True,
                )
                got = sorted(Path(tmp).glob("o*.jpg"))
                rgb = np.asarray(Image.open(got[0]).convert("RGB"))
        elif p == 0:
            rgb = np.asarray(Image.open(path).convert("RGB"))
        else:
            return None
        return d, p, encode(rgb)
    except Exception:
        return None


def render_job(args):
    """(path, [(doc, page)]) -> [(doc, page, jpeg)]: scan pictures decoded at reduced scale, other pages rendered."""
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
                    rgb = T.scan_image(doc, pg, H)
                    if rgb is None:
                        pm = pg.get_pixmap(
                            matrix=pymupdf.Matrix(W / pg.rect.width, H / pg.rect.height),
                            colorspace=pymupdf.csRGB,
                            alpha=False,
                        )
                        rgb = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, 3)
                    out.append((d, p, encode(rgb)))
                except Timeout:
                    raise
                except Exception:
                    pass
    except BaseException:
        pass
    finally:
        signal.alarm(0)
    done = {(d, p) for d, p, _ in out}
    out += [r for d, p in items if (d, p) not in done and (r := fallback(path, d, p))]
    return out


def run_jobs(jobs):
    """render_job over jobs on WORKERS processes; a crash of the pool (a file that kills MuPDF) splits the jobs until the file is found."""
    try:
        with ProcessPoolExecutor(WORKERS if len(jobs) > 1 else 1) as ex:
            return [x for r in ex.map(render_job, jobs, chunksize=8) for x in r]
    except BrokenProcessPool:
        if len(jobs) == 1:
            print("skipped (crash):", jobs[0][0], flush=True)
            return []
        h = len(jobs) // 2
        return run_jobs(jobs[:h]) + run_jobs(jobs[h:])


def upload(k, i, files, dataset=DATASET, work=None):
    """Dataset <dataset>-<i> with `files`; waits until they are listed (the status endpoint answers 403 for a while)."""
    import shutil

    ref = f"{dataset}-{i}"
    work = work or WORK
    names = {f.name for f in files}
    try:
        if names <= {x.name for x in k.dataset_list_files(ref).files}:
            return f"{ref} already there"
    except Exception:
        pass
    folder = work / f"upload_{i}"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir()
    for f in files:
        (folder / f.name).hardlink_to(f)
    (folder / "dataset-metadata.json").write_text(
        json.dumps(
            {"title": ref.split("/")[1], "id": ref, "licenses": [{"name": "CC-BY-NC-SA-4.0"}], "isPrivate": False}
        )
    )
    t0 = time.time()
    try:
        k.dataset_status(ref)
        k.dataset_create_version(str(folder), version_notes=time.strftime("%Y-%m-%d"), dir_mode="skip", quiet=False)
    except Exception:
        k.dataset_create_new(str(folder), dir_mode="skip", quiet=False)
    for _ in range(90):
        try:
            if names <= {x.name for x in k.dataset_list_files(ref).files}:
                try:  # the CLI creates every dataset private whatever isPrivate says: make it public
                    k.dataset_metadata_update(ref, str(folder))
                except Exception as e:
                    print(f"{ref}: could not make it public: {e!r}"[:200], flush=True)
                shutil.rmtree(folder)
                return f"{ref}: {len(files)} shards, {round(sum(f.stat().st_size for f in files) / 1e9, 2)} GB uploaded in {int(time.time() - t0)} s"
        except Exception:
            pass
        time.sleep(20)
    return f"{ref}: FILES NOT LISTED after the upload"


def cmd_build(_):
    from pagetype.pack_npz import write_shard

    from dspages.kaggle_api import api

    S = pd.read_parquet(WORK / "select.parquet").reset_index(drop=True)
    NPZ.mkdir(parents=True, exist_ok=True)
    k = api()
    n_sh = -(-len(S) // SHARD_PAGES)
    pool, pending, t0 = ThreadPoolExecutor(1), [], time.time()  # one upload at a time, while the next shards are built
    for s in range(n_sh):
        f = NPZ / f"forms_{s:04d}.npz"
        if not f.exists():
            part = S.iloc[s * SHARD_PAGES : (s + 1) * SHARD_PAGES]
            got = run_jobs(
                [(str(LAYOUTS / g.path.iloc[0]), list(zip(g.doc, g.page))) for _, g in part.groupby("doc", sort=True)]
            )
            got.sort(key=lambda x: (x[0], x[1]))
            write_shard(f, [(d, p) for d, p, _ in got], [j for *_, j in got])
            print(
                f"{f.name}: {len(got)}/{len(part)} pages, {round(f.stat().st_size / 1e9, 2)} GB, {int(time.time() - t0)} s",
                flush=True,
            )
        if (s + 1) % SHARDS_PER_DATASET == 0 or s == n_sh - 1:
            i = s // SHARDS_PER_DATASET
            files = [NPZ / f"forms_{j:04d}.npz" for j in range(i * SHARDS_PER_DATASET, s + 1)]
            pending.append(pool.submit(upload, k, i, files))
    for fut in pending:
        print(fut.result(), flush=True)
    print("done", int(time.time() - t0), "s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select").set_defaults(f=cmd_select)
    sub.add_parser("build").set_defaults(f=cmd_build)
    a = ap.parse_args()
    a.f(a)
