"""A bank of real handwriting lines in the format of `bank.py` (items {ink: alpha uint8, rgb: ink colour relative to the paper, key: language, wid: document, boxes: [one line box]}), cut from real handwritten pages: the pages the agents accepted (`ga/real_hw_review/good`) plus the pages of the real `handwritten` set that a classifier on their page embeddings, trained on the agents' accepted / rejected pages, scores above `min_p`.
python -m dspages.multilang.realbank [--min-p 0.95]"""
import argparse
import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from . import bank, paths
from .config import Config

U = paths.UVTM
GA = U / "handwriting_detection/ga"
CHUNKS = 100
DPI = 150
MIN_H, MAX_H, MIN_ASPECT = 15, 120, 8.0


def lang_of(path):
    m = re.search(r"_([a-z]{2,3})\.pdf$", path)
    return m.group(1) if m else "und"


def reviewed(items):
    """Indices into items of the pages the agents accepted and rejected (matched by path and page: the embeddings are rebuilt now and then)."""
    ix = {it: i for i, it in enumerate(items)}
    pos, neg = [], []
    for d in ("real260", "real_b"):
        m = json.load(open(GA / d / "meta.json"))
        bad = json.load(open(GA / d / "review.json"))["bad"]
        for k, v in m.items():
            i = ix.get((v["path"], v["page"]))
            if i is not None:
                (neg if str(int(k)) in bad else pos).append(i)
    return pos, neg


def pages_to_use(min_p):
    """[(path, page)] of the agent-accepted pages and of the pages the embedding classifier accepts."""
    from sklearn.linear_model import LogisticRegression
    R, items = load_real()
    pos, neg = reviewed(items)
    X = R[pos + neg]
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    p = LogisticRegression(max_iter=3000, C=3.0).fit(X, y).predict_proba(R)[:, 1]
    use = set(pos) | {i for i in range(len(R)) if p[i] >= min_p and i not in set(neg)}
    return [items[i] for i in sorted(use)], len(pos), len(use)


def lines_of(job):
    """Items of one page: lines with alpha, ink colour and box."""
    path, page = job
    import pymupdf
    cv2.setNumThreads(1)
    try:
        pg = pymupdf.open(U / "Layouts" / path)[page]
        pix = pg.get_pixmap(dpi=DPI, colorspace=pymupdf.csRGB)
    except Exception:
        return []
    rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, 3)
    return lines_of_rgb(rgb, lang_of(path), Path(path).stem)


def lines_of_rgb(rgb, key, wid):
    """Line items of a page image (rgb uint8 at DPI): the cut of lines_of, for real and generated pages alike."""
    from hwsynth.realism import clean
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    b = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    rules = np.zeros_like(b)
    for k in ((g.shape[1] // 8, 1), (1, g.shape[0] // 8)):
        rules |= cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k))
    b = cv2.subtract(b, rules)
    prof = cv2.GaussianBlur((b > 0).sum(1).astype(np.float32)[:, None], (1, 0), 3)[:, 0]
    on = prof > max(prof.max() * 0.08, 4)
    paper = cv2.GaussianBlur(cv2.dilate(g, np.ones((15, 15), np.uint8)), (0, 0), 7).astype(np.float32)
    rule_mask = cv2.dilate(rules, np.ones((5, 5), np.uint8)) > 0
    out, y = [], 0
    while y < len(on):
        if not on[y]:
            y += 1
            continue
        y1 = y
        while y1 < len(on) and on[y1]:
            y1 += 1
        if MIN_H <= y1 - y <= MAX_H:
            cols = np.nonzero((b[y:y1] > 0).sum(0) > 0)[0]
            if len(cols) and cols[-1] - cols[0] >= MIN_ASPECT * (y1 - y):
                pad = (y1 - y) // 6
                ya, yb, xa, xb = max(y - pad, 0), y1 + pad, max(cols[0] - pad, 0), cols[-1] + pad
                gc = g[ya:yb, xa:xb]
                if clean(gc):
                    pp = np.maximum(paper[ya:yb, xa:xb], 1)[..., None]
                    ratio = np.clip(rgb[ya:yb, xa:xb].astype(np.float32) / pp, 0, 1)
                    a = np.clip(1 - ratio.min(2), 0, 1)
                    a[rule_mask[ya:yb, xa:xb] & (b[ya:yb, xa:xb] == 0)] = 0
                    live = a > 0.05
                    if live.sum() > 50 and np.percentile(a[live], 95) > 0.12:
                        ys, xs = np.nonzero(a > 0.12)
                        y0_, y1_, x0_, x1_ = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                        a, ratio = a[y0_:y1_, x0_:x1_], ratio[y0_:y1_, x0_:x1_]
                        strong = a > np.percentile(a[a > 0.05], 80)
                        if not strong.any():
                            strong = a >= a.max() * 0.5
                        am = float(np.median(a[strong]))
                        col = np.clip(1 - (1 - np.median(ratio[strong], 0)) / max(am, 1e-3), 0, 1)
                        out.append(dict(ink=(a * 255).astype(np.uint8), rgb=col.astype(np.float32), key=key, wid=wid,
                                        boxes=[(0, 0, a.shape[1], a.shape[0])]))
        y = y1
    return out


def chunk_of(wid):
    return int(hashlib.md5(wid.encode()).hexdigest(), 16) % CHUNKS


def build(cfg: Config, min_p=0.95, max_lines_per_page=12, workers=11):
    use, n_agent, n_all = pages_to_use(min_p)
    print(f"{n_agent} agent-accepted pages, {n_all} pages in all", flush=True)
    out = cfg.data / "bank_real"
    out.mkdir(parents=True, exist_ok=True)
    by = {}
    with ProcessPoolExecutor(workers) as ex:
        for page_items in ex.map(lines_of, use, chunksize=4):
            rng = np.random.default_rng(len(by))
            if len(page_items) > max_lines_per_page:
                page_items = [page_items[i] for i in rng.permutation(len(page_items))[:max_lines_per_page]]
            for it in page_items:
                by.setdefault(chunk_of(it["wid"]), []).append(it)
    for c, items in by.items():
        bank.write_chunk(out / f"{c:05d}.pkl", items)
    keys = Counter(it["key"] for items in by.values() for it in items)
    print(sum(keys.values()), "lines;", len(keys), "languages;", keys.most_common(40), flush=True)
    return keys


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-p", type=float, default=0.95)
    a = ap.parse_args()
    build(Config(data=paths.DATA), a.min_p)


def load_real():
    E = np.load(paths.UVTM / "emb_out/embeddings.npy").astype(np.float32)
    ids = np.load(paths.UVTM / "emb_out/ids.npy")
    dt = {}
    for line in open(paths.UVTM / "emb_out/doc_types.jsonl"):
        r = json.loads(line)
        dt[r["doc"]] = r
    ok = np.array([d in dt and dt[d]["label"] == "handwritten" and (U / "Layouts" / dt[d]["path"]).suffix == ".pdf"
                   and (U / "Layouts" / dt[d]["path"]).exists() for d in ids[:, 0]])
    ix = np.flatnonzero(ok)
    E = E[ix]
    return E / np.linalg.norm(E, axis=1, keepdims=True), [(dt[ids[i, 0]]["path"], int(ids[i, 1])) for i in ix]


def select(synth, n=200, per_doc=1):
    """Indices into the real handwritten set"""
    R, items = load_real()
    S = synth / np.linalg.norm(synth, axis=1, keepdims=True)
    order = np.argsort(-(S @ R.T), axis=1)
    chosen, docs, rank = [], {}, 0
    while len(chosen) < n and rank < order.shape[1]:
        for k in range(len(S)):
            j = int(order[k, rank])
            if j not in chosen and docs.get(items[j][0], 0) < per_doc:
                chosen.append(j)
                docs[items[j][0]] = docs.get(items[j][0], 0) + 1
                if len(chosen) == n:
                    break
        rank += 1
    return chosen, R, items


def render(path, page, side=1024):
    d = pymupdf.open(U / "Layouts" / path)
    pg = d[min(page, len(d) - 1)]
    k = side / max(pg.rect.width, pg.rect.height)
    px = pg.get_pixmap(matrix=pymupdf.Matrix(k, k), alpha=False)
    return np.frombuffer(px.samples, np.uint8).reshape(px.h, px.w, 3).copy()


def write(chosen, items, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    for n, j in enumerate(chosen):
        img = render(*items[j])
        cv2.imwrite(str(out / f"{n:03d}.jpg"), img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
        meta[f"{n:03d}"] = dict(path=items[j][0], page=items[j][1], real_index=j)
    (out / "meta.json").write_text(json.dumps(meta))
    for s in range(0, len(chosen), 20):
        tiles = []
        for n in range(s, min(s + 20, len(chosen))):
            im = cv2.imread(str(out / f"{n:03d}.jpg"))
            t = cv2.resize(im, (256, int(256 * im.shape[0] / im.shape[1])))[:360]
            t = cv2.copyMakeBorder(t, 0, 360 - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(t, str(n), (4, 24), 0, 0.9, (0, 0, 255), 2)
            tiles.append(t)
        while len(tiles) < 20:
            tiles.append(np.full((360, 256, 3), 255, np.uint8))
        cv2.imwrite(str(out / f"sheet_{s // 20:02d}.jpg"), np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, 20, 5)]))
