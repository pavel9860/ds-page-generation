"""Layouts of the detection set
python -m dspages.multilang.detlayouts [--workers 12]"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .config import GAP, PAGE, SAME_LINE  # noqa: F401
from .config import LANGDS as C
from .paths import DATA as D

LANGDS, OUT = D / "langds", D / "langds/det_layouts"
TYPES = {"text", "table", "form", "flyer", "magazine"}


def near_a4(w, h):
    return abs(w / h / PAGE.ratio - 1) <= PAGE.tolerance


@cache
def form_is_near_a4(path, page):
    import pymupdf

    with pymupdf.open(path) as doc:
        r = doc[int(page)].rect
    return near_a4(r.width, r.height)


@cache
def layout_is_near_a4(file):
    from PIL import Image

    with Image.open(D / "langds/det_layouts" / file) as im:
        return near_a4(*im.size)


def lines(words):
    """Line boxes [x0, y0, x1, y1] of text-layer words [(x0, y0, x1, y1, text), ...]."""
    rows = []
    for w in sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        h = w[3] - w[1]
        for r in rows:
            ov = min(r[3], w[3]) - max(r[1], w[1])
            if ov > SAME_LINE * min(h, r[3] - r[1]) and 0.5 < h / (r[3] - r[1]) < 2:
                r[4].append(w)
                r[1], r[3] = min(r[1], w[1]), max(r[3], w[3])
                break
        else:
            rows.append([w[0], w[1], w[2], w[3], [w]])
    out = []
    for r in rows:
        ws = sorted(r[4])
        h = float(np.median([w[3] - w[1] for w in ws]))
        seg = [ws[0]]
        for w in ws[1:]:
            if w[0] - max(s[2] for s in seg) > GAP * h:
                out.append(seg)
                seg = []
            seg.append(w)
        out.append(seg)
    return merge_nested(
        [[min(w[0] for w in s), min(w[1] for w in s), max(w[2] for w in s), max(w[3] for w in s)] for s in out]
    )


def merge_nested(boxes, share=0.5):
    """Boxes lying mostly (share of their area) inside another are joined to it, until none is."""
    bs = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(bs)):
            for j in range(len(bs)):
                if i == j:
                    continue
                a, b = bs[i], bs[j]
                iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
                if iw > 0 and ih > 0 and iw * ih > share * (a[2] - a[0]) * (a[3] - a[1]):
                    bs[j] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del bs[i]
                    changed = True
                    break
            if changed:
                break
    return bs


def pdf_index():
    """Content id (file name up to the first `_`) -> current path of the layout PDFs (files are renamed and moved between folders; the last destination in Layouts/archive/info/moves.jsonl is used when the id is not found by name)."""
    lay = D.parent / "Layouts"
    idx = {}
    for f in lay.rglob("*.pdf"):
        idx.setdefault(f.name.split("_")[0], f)
    for line in open(lay / "archive/info/moves.jsonl"):
        m = json.loads(line)
        h = Path(m.get("dst", "")).name.split("_")[0]
        if h not in idx and (lay / m["dst"]).exists():
            idx[h] = lay / m["dst"]
    return idx


def render(r):
    import cv2
    import pymupdf

    cv2.setNumThreads(1)
    doc = pymupdf.open(r["path"])
    pg = doc[r["page"]]
    clip = pg.rect
    k = C.side / max(clip.width, clip.height)
    px = pg.get_pixmap(matrix=pymupdf.Matrix(k, k), colorspace=pymupdf.csGRAY)
    canvas = np.frombuffer(px.samples, np.uint8).reshape(px.h, px.w).copy()
    words = [
        (w[0], w[1], w[2], w[3], w[4]) for w in pg.get_text("words", clip=clip) if w[4].strip() and w[3] - w[1] > 2
    ]
    boxes = [
        [
            round((b[0] - clip.x0) * k),
            round((b[1] - clip.y0) * k),
            round((b[2] - clip.x0) * k),
            round((b[3] - clip.y0) * k),
        ]
        for b in lines(words)
    ]
    H, W = canvas.shape
    boxes = [
        b for b in boxes if b[2] > 0 and b[3] > 0 and b[0] < W and b[1] < H and b[2] - b[0] > 2 and b[3] - b[1] > 2
    ]
    if len(boxes) < 3:
        return None
    cv2.imwrite(str(OUT / "img" / f"{r['id']}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    imgs = [pymupdf.Rect(i["bbox"]) & clip for i in pg.get_image_info()]
    imgs = [r_ for r_ in imgs if r_.width > 20 and r_.height > 10]
    on_image = sum(any(r_.contains(pymupdf.Point((w[0] + w[2]) / 2, (w[1] + w[3]) / 2)) for r_ in imgs) for w in words)
    if on_image > 0.5 * len(words):
        return None
    px_words = [
        [
            round((w[0] - clip.x0) * k),
            round((w[1] - clip.y0) * k),
            round((w[2] - clip.x0) * k),
            round((w[3] - clip.y0) * k),
            w[4],
        ]
        for w in words
    ]
    from hwsynth.scripts import text_script

    from .realpages import text_layer_ok

    script = text_script(" ".join(w[4] for w in words[:60]) or "a")
    return dict(
        id=r["id"],
        lang=r["lang"],
        type=r["type"],
        script=script,
        src=r["path"],
        page=r["page"],
        text_ok=text_layer_ok(pg),
        doc=Path(r["src"]).name.split("_")[0],
        file=f"img/{r['id']}.jpg",
        prints=boxes,
        words=[w for w in px_words if 0 <= w[0] and w[2] <= W and 0 <= w[1] and w[3] <= H],
    )


def make(r):
    return render(r) if r.get("path") else None


def candidates():
    out = []
    extra = [l for f in sorted(LANGDS.glob("det_extra_*.jsonl")) for l in open(f)]
    for line in list(open(LANGDS / "real.jsonl")) + extra:
        r = json.loads(line)
        src = Path(r["src"])
        if not r["scan"] and r["type"] in TYPES and r["lang_status"] == "ok" and src.suffix.lower() == ".pdf":
            out.append(r)
    return out


def load():
    return [json.loads(l) for l in open(OUT / "layouts.jsonl")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    log = OUT / "layouts.jsonl"
    done = {json.loads(l)["id"] for l in open(log)} if log.exists() else set()
    idx = pdf_index()
    todo = [
        dict(r, path=r["src"] if Path(r["src"]).exists() else str(idx.get(Path(r["src"]).name.split("_")[0], "")))
        for r in candidates()
        if r["id"] not in done
    ]
    print(len(done), "done,", len(todo), "to render", flush=True)
    n = 0
    with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn")) as ex, open(log, "a") as f:
        for row in ex.map(make, todo, chunksize=8):
            if row:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
    print(n, "rendered")


if __name__ == "__main__":
    main()
