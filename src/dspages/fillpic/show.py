"""Overlay images (window red, ink tiles green, picture tiles blue) and values.csv of the detector on page sources; label images of the training pages."""
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pymupdf

from ..layout import refont
from .config import CONFIG
from .detector import Detector, fit
from .labels import picture_rects, picture_tiles, text_tiles, word_mask
from .maps import ink
from .paths import LAYOUTS, MANIFEST_DB, MODEL, TESTS, doc_hash, doc_paths
from .render import render, render_page
from .tiles import tile_mean, window
from .train import train_pages

COLUMNS = ["set", "name", "fill", "pic", "sec"]


def paste(m, win, shape):
    """The grid m stretched over the window (px of an image of `shape`), as a full bool mask of that shape."""
    x, y, w, h = win
    big = cv2.resize(m.astype(np.uint8), (max(int(round(w)), 1), max(int(round(h)), 1)), interpolation=cv2.INTER_NEAREST)
    out = np.zeros(shape, bool)
    x0, y0 = int(round(x)), int(round(y))
    a, b, c, d = max(x0, 0), min(x0 + big.shape[1], shape[1]), max(y0, 0), min(y0 + big.shape[0], shape[0])
    if b > a and d > c:
        out[c:d, a:b] = big[c - y0:d - y0, a - x0:b - x0] > 0
    return out


def title_strip(img, text):
    strip = np.full((28, img.shape[1], 3), 255, np.uint8)
    cv2.putText(strip, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return np.vstack([strip, img])


def preview(gray, cfg=CONFIG):
    side = cfg.geometry.mask_side
    s = side / max(gray.shape)
    return cv2.cvtColor(cv2.resize(gray, (round(gray.shape[1] * s), round(gray.shape[0] * s)), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)


def process(det, kind, name, gray, out):
    """Detector on one canvas page: its row of values.csv and its overlay image in out / kind."""
    t = time.time()
    gray = fit(gray, det.cfg)
    r = det.predict([gray])[0]
    sec = time.time() - t
    img = preview(gray, det.cfg)
    h, w = img.shape[:2]
    win = (r.window[0] * w, r.window[1] * h, r.window[2] * w, r.window[3] * h)
    lay = img.copy()
    lay[paste(r.ink, win, (h, w))] = (0, 200, 0)
    lay[paste(r.pic > 0.5, win, (h, w))] = (255, 0, 0)
    img = cv2.addWeighted(lay, 0.3, img, 0.7, 0)
    cv2.rectangle(img, (int(win[0]), int(win[1])), (int(win[0] + win[2]), int(win[1] + win[3])), (0, 0, 255), 2)
    text = f"{name} fill {r.fill:.2f} pic {r.pic_mean:.2f}"
    (Path(out) / kind).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(Path(out) / kind / f"{name}.jpg"), title_strip(img, text))
    return kind, name, r.fill, r.pic_mean, sec


def run(pages, kind, out, det=None):
    """pages: iterable of (name, gray) -> values.csv in out, a summary line."""
    det = det or Detector.load(MODEL)
    Path(out).mkdir(parents=True, exist_ok=True)
    R = pd.DataFrame([process(det, kind, name, gray, out) for name, gray in pages], columns=COLUMNS)
    R.to_csv(Path(out) / "values.csv", index=False)
    line = f"{len(R)} pages | mean fill {R.fill.mean():.3f} pic {R['pic'].mean():.3f}"
    print(line, "| images and values.csv in", out)


def folder_pages(folder):
    for f in sorted(Path(folder).glob("*.jpg")):
        yield f.stem, cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)


def hw_pages(skip=0, n=100, seed=7):
    """One random page of every handwritten PDF in the order of a seeded draw: the first `skip` left out, then n."""
    rng = np.random.default_rng(seed)
    files, k = sorted((LAYOUTS / "handwritten").glob("*.pdf")), 0
    for f in rng.permutation(files):
        if k == skip + n:
            return
        with pymupdf.open(str(f)) as doc:
            page = doc[int(rng.integers(doc.page_count))]
            if k >= skip:
                yield f"{f.stem}_{page.number}", render_page(page)
        k += 1


def done_names(tests, exclude=None):
    return {p.stem for p in tests.glob("*/*/*.jpg") if exclude is None or p.parent.parent != exclude}


def hw_new_pages(n, out, cfg=CONFIG):
    """Pages of handwritten PDFs that no earlier output holds: first all pages of Japanese, Chinese, Korean and rare-script documents (language = end of the file name; at
    random if more than n), then random ones."""
    done = done_names(TESTS, Path(out))
    pages = []
    for f in sorted((LAYOUTS / "handwritten").glob("*.pdf")):
        with pymupdf.open(str(f)) as doc:
            pages += [(f, i) for i in range(doc.page_count) if f"{f.stem}_{i}" not in done]
    first = np.array([f.stem.rsplit("_", 1)[1] in cfg.languages.cjk + cfg.languages.rare for f, _ in pages])
    order = np.random.default_rng(cfg.train.seed).permutation(len(pages))
    order = np.concatenate([order[first[order]], order[~first[order]]])[:n]
    print(int(first.sum()), "pages of Japanese / Chinese / Korean / rare-script documents,", len(pages), "not done", flush=True)
    for k in order:
        f, i = pages[k]
        yield f"{f.stem}_{i}", render(f, i)


def digital_hashes():
    """Hashes of the documents whose plan class is not scan-like."""
    with sqlite3.connect(f"file:{MANIFEST_DB}?mode=ro", uri=True) as db:
        return {d for (d,) in db.execute("select doc from plan where cls != 'scan-like'")}


def used_pages(out):
    """Names of earlier outputs and the training pages: (path, page) tuples and names."""
    return done_names(TESTS, Path(out)) | train_pages()


def random_pages(n, seed, digital, out):
    """n random embedded pages (digital ones too if `digital`) that are in no earlier output and not in the training set."""
    from .paths import EMB
    skip = used_pages(out)
    ids = np.load(EMB / "ids.npy")
    man = doc_paths()
    dig = set() if digital else digital_hashes()
    k = 0
    for i in np.random.default_rng(seed).permutation(len(ids)):
        d, p = int(ids[i][0]), int(ids[i][1])
        path = man[d]
        name = f"{Path(path).stem}_{p}"
        if doc_hash(path) in dig or (path, p) in skip or name in skip:
            continue
        try:
            g = render(path, p)
        except (AssertionError, ValueError, RuntimeError, IndexError):
            continue
        yield name, g
        k += 1
        if k == n:
            return


def label_image(path, number, title, cfg=CONFIG):
    """Labels of a PDF page on the clean render: text tiles (words of the PDF, also on a picture) orange, other ink tiles green, picture share of a tile blue (opacity =
    share, text tiles have none), tile grid grey; title in a strip above. None without ink."""
    g = cfg.geometry.grid
    with pymupdf.open(path) as doc:
        src = refont.Source(doc)
        page = src[number]
        clean = render_page(page, cfg)
        rects, _ = picture_rects(page, src, number, cfg)
        wm = word_mask(src, page, number, clean.shape)
    ink_ = ink(clean, cfg)
    mask = ink_.mask
    win = window(mask, cfg.geometry.infer_margin, cfg)
    if win is None:
        return None
    has_ink = tile_mean(mask, win, g) > cfg.ink.eps
    text = text_tiles(wm, win, mask.shape, cfg)
    pic = picture_tiles(rects, win, mask.shape, cfg) * ~text
    img = cv2.cvtColor(ink_.small, cv2.COLOR_GRAY2BGR).astype(np.float32)
    for tiles, col in ((has_ink & ~text, (0, 200, 0)), (text, (0, 140, 255))):
        p = paste(tiles, win, mask.shape)
        img[p] = 0.7 * img[p] + 0.3 * np.array(col, np.float32)
    share = cv2.resize(pic.astype(np.float32), (max(int(round(win[2])), 1), max(int(round(win[3])), 1)), interpolation=cv2.INTER_NEAREST)
    x0, y0 = int(round(win[0])), int(round(win[1]))
    a, b, c, d = max(x0, 0), min(x0 + share.shape[1], mask.shape[1]), max(y0, 0), min(y0 + share.shape[0], mask.shape[0])
    if b > a and d > c:
        al = 0.6 * share[c - y0:d - y0, a - x0:b - x0, None]
        img[c:d, a:b] = (1 - al) * img[c:d, a:b] + al * np.array([255, 0, 0], np.float32)
    img = img.astype(np.uint8)
    for k in range(g + 1):
        t = k / g
        cv2.line(img, (int(win[0] + t * win[2]), int(win[1])), (int(win[0] + t * win[2]), int(win[1] + win[3])), (150, 150, 150), 1)
        cv2.line(img, (int(win[0]), int(win[1] + t * win[3])), (int(win[0] + win[2]), int(win[1] + t * win[3])), (150, 150, 150), 1)
    return title_strip(img, f"{title} label fill {has_ink.mean():.2f} text tiles {text.mean():.2f} pic {pic.mean():.2f}")


def _label_job(job):
    path, number, out, name, title = job
    try:
        img = label_image(path, number, title)
        if img is not None:
            cv2.imwrite(str(Path(out) / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return int(img is not None)
    except BaseException:
        return 0


def labels(n=4000, cfg=CONFIG):
    """Label images of n training pages -> tests/scan_detector_samples."""
    out = TESTS / "scan_detector_samples"
    jobs = [(p, k, str(out), f"{Path(p).stem}_{k}", f"{p.replace(str(LAYOUTS) + '/', '').split('/')[0]} {Path(p).stem} p{k}") for p, k in sorted(train_pages())[:n]]
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(cfg.train.workers) as ex:
        done = sum(ex.map(_label_job, jobs, chunksize=4))
    print(done, "of", len(jobs), "images in", out)
