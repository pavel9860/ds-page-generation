"""Handwriting vs print detector: synthetic pages (hwsynth lines on real PDF pages), YOLO labels, evaluation.

python tools/hwdet.py text                      text lines per language from the PDFs, random words of rare scripts
python tools/hwdet.py bank [--n 16000]          handwritten lines and marks by hwsynth (chunks, resumable)
python tools/hwdet.py pages [--n 6000 --val 600] composed pages + YOLO labels (0 handwriting, 1 print), resumable
python tools/hwdet.py eval MODEL [--real 150]   FP/FN of handwriting per page on val (by script) and on real pages
"""
import argparse
import hashlib
import json
import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from dspages.config import Paths

LAYOUTS = Path(Paths().layouts)
OUT = Path("/run/media/me/D/ML_DS/UVTM/hwdet")
TEXT = OUT / "text.pkl"
BANK = OUT / "bank"
PARA = OUT / "para"
YOLO = OUT / "ds"
BG_FOLDERS = ("forms", "books_pdf", "other_docs", "bureaucracy", "certificates", "exams", "tables", "receipts",
              "scientific_paper", "magazines", "flyers")
DPI = 150
LONG = 1024
CHUNK = 100
PCHUNK = 10
REUSE = 3
STACK_FROM = 7500
LEADER = set(".…_·‥")


def lang_of(p):
    s = p.stem.rsplit("_", 1)[-1]
    return s if 2 <= len(s) <= 3 and s.isalpha() else None


def page_lines(page):
    out = []
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            t = "".join(s["text"] for s in ln["spans"]).strip()
            if t and any(s["text"].strip() and s["size"] > 3 for s in ln["spans"]):
                out.append((t, ln["bbox"]))
    return out


def cmd_text(a):
    from hwsynth.cli import coverage
    from hwsynth.scripts import letters
    by = {}
    for d in BG_FOLDERS:
        for p in sorted((LAYOUTS / d).glob("*.pdf")):
            if (l := lang_of(p)) and l not in ("mul", "xx"):
                by.setdefault(l, []).append(p)
    pool = {}
    for l, fs in sorted(by.items()):
        random.Random(l).shuffle(fs)
        lines = []
        for p in fs[:25]:
            try:
                with pymupdf.open(p) as doc:
                    for pg in list(doc)[:4]:
                        lines += [t for t, _ in page_lines(pg) if 3 <= len(t) <= 80]
            except Exception:
                continue
        if len(lines) >= 30:
            pool[l] = lines[:3000]
    rng = random.Random(0)
    for s in coverage():
        ch = letters(s)
        if len(ch) >= 5:
            pool[f"script:{s}"] = [" ".join("".join(rng.choices(ch, k=rng.randint(2, 7))) for _ in range(rng.randint(1, 5)))
                                   for _ in range(200)]
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(pool, open(TEXT, "wb"))
    print(len(pool), "keys,", sum(k.startswith("script:") for k in pool), "rare scripts")


def fill_text(rng):
    k = rng.random()
    if k < 0.35:
        return rng.choice([f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.{rng.randint(1950, 2026)}",
                           f"{rng.randint(1, 12)}/{rng.randint(1, 28)}/{rng.randint(50, 99)}",
                           f"+{rng.randint(1, 99)} {rng.randint(100, 999)} {rng.randint(100, 999)} {rng.randint(1000, 9999)}",
                           f"{rng.randint(1, 99999)}.{rng.randint(0, 99):02d}", str(rng.randint(1, 999)),
                           f"№ {rng.randint(1, 9999)}", f"{rng.randint(1, 23)}:{rng.randint(0, 59):02d}"])
    return None


def line_boxes(words, ink):
    rows = []
    for _, (x0, y0, x1, y1), _ in sorted(words, key=lambda w: (w[1][1] + w[1][3]) / 2):
        c = (y0 + y1) / 2
        if rows and abs(c - rows[-1][4]) < 0.5 * (rows[-1][3] - rows[-1][1]):
            r = rows[-1]
            rows[-1] = [min(r[0], x0), min(r[1], y0), max(r[2], x1), max(r[3], y1), r[4]]
        else:
            rows.append([x0, y0, x1, y1, c])
    out = []
    h, w = ink.shape
    for x0, y0, x1, y1, _ in rows:
        pad = int(0.3 * (y1 - y0))
        X0, Y0, X1, Y1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, w), min(y1 + pad, h)
        ys, xs = np.nonzero(ink[Y0:Y1, X0:X1] > 0.15)
        if len(ys):
            out.append((X0 + xs.min(), Y0 + ys.min(), X0 + xs.max() + 1, Y0 + ys.max() + 1))
    return out


def bank_item(rng, pool, keys):
    import hwsynth as H
    w = H.sample_writer(rng.choice(list(H.PERSONS)), rng.choice(list(H.STYLES)), seed=rng.randrange(2**31))
    if rng.random() < 0.08:
        s = H.figure(rng.choice(["check", "cross", "circle", "underline", "check"]), w, size=rng.uniform(4, 12))
        ys, xs = np.nonzero(s.ink > 0.15)
        return s.ink, w.pen.ink, [(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)], "mark"
    key = rng.choice(keys)
    t = fill_text(rng)
    if t is None:
        t = rng.choice(pool[key])
        if rng.random() < 0.3:
            t = t + " " + rng.choice(pool[key])
        wds = t.split()
        if len(wds) > 3 and rng.random() < 0.5:
            i = rng.randrange(len(wds))
            t = " ".join(wds[i:i + rng.randint(1, 4)])
    s = H.write(t, w, width_mm=rng.uniform(40, 170), dpi=DPI, ruled=False, scan=False)
    return s.ink, w.pen.ink, line_boxes(s.words, s.ink), key


def bank_chunk(i):
    p = BANK / f"{i:05d}.pkl"
    if p.exists():
        return 0
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    pool = pickle.load(open(TEXT, "rb"))
    langs = [k for k in pool if not k.startswith("script:")]
    rare = [k for k in pool if k.startswith("script:")]
    rng = random.Random(i)
    items = []
    for _ in range(CHUNK):
        keys = rare if rng.random() < 0.2 else langs
        try:
            ink, rgb, boxes, key = bank_item(rng, pool, keys)
        except Exception as e:
            items.append(dict(err=repr(e)))
            continue
        ys, xs = np.nonzero(ink > 0.05)
        if not len(ys) or not boxes:
            continue
        y0, x0 = ys.min(), xs.min()
        items.append(dict(ink=(ink[y0:ys.max() + 1, x0:xs.max() + 1] * 255).astype(np.uint8), rgb=np.asarray(rgb, np.float32),
                          boxes=[(a - x0, b - y0, c - x0, d - y0) for a, b, c, d in boxes], key=key))
    tmp = p.with_suffix(".tmp")
    pickle.dump(items, open(tmp, "wb"))
    tmp.replace(p)
    return len(items)


def para_chunk(i):
    p = PARA / f"{i:05d}.pkl"
    if p.exists():
        return 0
    import hwsynth as H
    pool = pickle.load(open(TEXT, "rb"))
    langs = [k for k in pool if not k.startswith("script:")]
    rare = [k for k in pool if k.startswith("script:")]
    rng = random.Random(10**6 + i)
    items = []
    for _ in range(PCHUNK):
        key = rng.choice(rare if rng.random() < 0.15 else langs)
        t = " ".join(rng.choice(pool[key]) for _ in range(rng.randint(4, 25)))
        try:
            w = H.sample_writer(rng.choice(list(H.PERSONS)), rng.choice(list(H.STYLES)), seed=rng.randrange(2**31))
            s = H.write(t, w, width_mm=rng.uniform(110, 190), dpi=DPI, ruled=None, scan=None)
        except Exception as e:
            items.append(dict(err=repr(e)))
            continue
        boxes = line_boxes(s.words, s.ink)
        if boxes:
            items.append(dict(jpg=cv2.imencode(".jpg", cv2.cvtColor(s.image, cv2.COLOR_RGB2BGR))[1].tobytes(), boxes=boxes, key=key))
    tmp = p.with_suffix(".tmp")
    pickle.dump(items, open(tmp, "wb"))
    tmp.replace(p)
    return len(items)


def cmd_para(a):
    PARA.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        done = sum(ex.map(para_chunk, range(-(-a.n // PCHUNK))))
    print("para:", done, "new")


def para_page(args):
    i, it = args
    d = YOLO / "train"
    if (d / "labels" / f"{i:05d}.txt").exists():
        return None
    rng = random.Random(i)
    img = cv2.cvtColor(cv2.imdecode(np.frombuffer(it["jpg"], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    H = max(h, int(w * rng.uniform(1.3, 1.45)))
    if H > h:
        top = rng.randint(0, H - h)
        fill = np.median(img.reshape(-1, 3)[:: 97], 0).astype(np.uint8)
        img = np.vstack([np.full((top, w, 3), fill), img, np.full((H - h - top, w, 3), fill)])
    else:
        top = 0
    k = LONG / max(img.shape[:2])
    img = degrade(cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA), rng)
    hw = [(x0 * k, (y0 + top) * k, x1 * k, (y1 + top) * k) for x0, y0, x1, y1 in it["boxes"]]
    write_page(d, i, img, hw, [], rng)
    return dict(i=i, src="para", hw=[it["key"]] * len(hw))


def cmd_paras(a):
    from hwsynth.cli import coverage
    hand = {f"script:{k}" for k, v in coverage().items() if v == "hand"}
    items = [it for f in sorted(PARA.glob("*.pkl")) for it in pickle.load(open(f, "rb"))
             if "jpg" in it and (not it["key"].startswith("script:") or it["key"] in hand)]
    tasks = [(a.start + j, it) for j, it in enumerate(items)]
    with ProcessPoolExecutor(a.workers) as ex, open(YOLO / "train_meta.jsonl", "a") as f:
        for m in ex.map(para_page, tasks, chunksize=8):
            if m:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print("para pages:", len(tasks))


def cmd_bank(a):
    BANK.mkdir(parents=True, exist_ok=True)
    n = -(-a.n // CHUNK)
    with ProcessPoolExecutor(a.workers) as ex:
        done = sum(ex.map(bank_chunk, range(n)))
    errs = sum("err" in it for f in BANK.glob("*.pkl") for it in pickle.load(open(f, "rb")))
    print(f"bank: {n} chunks, {done} new items, {errs} errors")


def backgrounds():
    out = []
    for d in BG_FOLDERS:
        out += sorted((LAYOUTS / d).glob("*.pdf"))
    return out


def is_val(p):
    return int(hashlib.md5(p.name.encode()).hexdigest(), 16) % 10 == 0


def page_segments(pg):
    """Print boxes split at dot/underscore leaders and wide gaps; the leaders are returned as fields to fill."""
    prints, fields = [], []
    for b in pg.get_text("rawdict")["blocks"]:
        for ln in b.get("lines", []):
            chars = [c for sp in ln["spans"] if sp["size"] > 3 for c in sp["chars"]]
            if not chars:
                continue
            h = max(c["bbox"][3] - c["bbox"][1] for c in chars)
            seg, kind = [], None
            def flush():
                if seg and kind == "p" and any(c["c"].strip() for c in seg):
                    r = [c["bbox"] for c in seg if c["c"].strip()]
                    prints.append((min(q[0] for q in r), min(q[1] for q in r), max(q[2] for q in r), max(q[3] for q in r)))
                elif seg and kind == "f" and len(seg) >= 4:
                    fields.append((seg[0]["bbox"][0], seg[0]["bbox"][1], seg[-1]["bbox"][2], seg[-1]["bbox"][3]))
                elif seg and kind == "f":
                    r = [c["bbox"] for c in seg]
                    prints.append((min(q[0] for q in r), min(q[1] for q in r), max(q[2] for q in r), max(q[3] for q in r)))
            prev = None
            for c in chars:
                k = "f" if c["c"] in LEADER else ("s" if not c["c"].strip() else "p")
                if k == "s":
                    if seg:
                        seg.append(c)
                    continue
                gap = prev is not None and c["bbox"][0] - prev["bbox"][2] > 2 * h
                if k != kind or gap:
                    flush()
                    seg, kind = [], k
                seg.append(c)
                prev = c
            flush()
    return prints, fields


def render_bg(p, rng):
    with pymupdf.open(p) as doc:
        pg = doc[rng.randrange(min(len(doc), 4))]
        pa = abs(pg.rect)
        if any(abs(pymupdf.Rect(i["bbox"]) & pg.rect) > 0.5 * pa for i in pg.get_image_info()):
            return None
        prints, fields = page_segments(pg)
        if len(prints) < 3:
            return None
        k = LONG / max(pg.rect.width, pg.rect.height)
        px = pg.get_pixmap(matrix=pymupdf.Matrix(k, k), alpha=False)
    img = np.frombuffer(px.samples, np.uint8).reshape(px.h, px.w, px.n)[:, :, :3].copy()
    sc = lambda bs: [tuple(v * k for v in b) for b in bs]
    return img, sc(prints), sc(fields)


def free_mask(img, pboxes):
    """occ: everything that is not plain paper (ink, pictures, tints), form rules excepted; rules: long horizontal
    rules (fields to write on)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    ink = ((hsv[..., 2] < 225) | (hsv[..., 1] > 40)).astype(np.uint8)
    hor = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1)))
    ver = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)))
    occ = ink & ~(hor | ver)
    for x0, y0, x1, y1 in pboxes:
        occ[int(y0):int(y1) + 1, int(x0):int(x1) + 1] = 1
    n, _, st, _ = cv2.connectedComponentsWithStats(hor)
    rules = [(x, y, x + w, y + h) for x, y, w, h, _ in st[1:] if w >= 60 and h <= 5]
    return cv2.dilate(occ, np.ones((5, 5), np.uint8)), rules


def blend(img, a, rgb, x, y, d=1.0):
    h, w = a.shape
    img[y:y + h, x:x + w] = (img[y:y + h, x:x + w] * (1 - d * a[..., None] * (1 - rgb[None, None]))).astype(np.uint8)


def paste(img, occ, it, scale, rng, tries=30):
    ink = it["ink"]
    h, w = int(ink.shape[0] * scale), int(ink.shape[1] * scale)
    H, W = img.shape[:2]
    if h < 4 or w < 4 or h >= H or w >= W:
        return None
    a = cv2.resize(ink, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    for _ in range(tries):
        y, x = rng.randrange(H - h), rng.randrange(W - w)
        if occ[y:y + h, x:x + w].mean() < 0.02:
            blend(img, a, it["rgb"], x, y, rng.uniform(0.75, 1.0))
            occ[y:y + h, x:x + w] = 1
            return [(x + b0 * scale, y + b1 * scale, x + b2 * scale, y + b3 * scale) for b0, b1, b2, b3 in it["boxes"]]
    return None


def fill_field(img, pmask, it, field, lh, rng):
    """One-line item written on a field (leader or rule): baseline near the field bottom, may cross the dots."""
    fx0, fy0, fx1, fy1 = field
    ink = it["ink"]
    k = rng.uniform(1.0, 1.9) * lh / ink.shape[0]
    w = int(ink.shape[1] * k)
    if w > fx1 - fx0:
        k *= (fx1 - fx0) / w
    h, w = int(ink.shape[0] * k), int(ink.shape[1] * k)
    H, W = img.shape[:2]
    if h < 5 or w < 5:
        return None
    x = int(fx0 + rng.uniform(0, max(fx1 - fx0 - w, 0)))
    y = int(fy1 - h + rng.uniform(-0.1, 0.25) * h)
    if x < 0 or y < 0 or x + w >= W or y + h >= H or pmask[y:y + h, x:x + w].mean() > 0.05:
        return None
    blend(img, cv2.resize(ink, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255, it["rgb"], x, y, rng.uniform(0.8, 1.0))
    pmask[y:y + h, x:x + w] = 1
    return [(x + b0 * k, y + b1 * k, x + b2 * k, y + b3 * k) for b0, b1, b2, b3 in it["boxes"]]


def letter_page(rng, items):
    H, W = LONG, int(LONG * rng.uniform(0.68, 0.78))
    img = np.full((H, W, 3), rng.randint(225, 252), np.uint8)
    occ = np.zeros((H, W), np.uint8)
    occ[:30], occ[-30:], occ[:, :30], occ[:, -30:] = 1, 1, 1, 1
    out = []
    for it in items:
        b = paste(img, occ, it, rng.uniform(0.6, 1.2), rng, 60)
        if b:
            out += [(it["key"], bb) for bb in b]
    return img, [], out


def stack_page(rng, items):
    H, W = LONG, int(LONG * rng.uniform(0.68, 0.78))
    img = np.full((H, W, 3), [rng.randint(225, 252)] * 3, np.uint8)
    pitch = rng.randint(22, 48)
    if rng.random() < 0.5:
        c = (rng.randint(120, 200), rng.randint(150, 210), rng.randint(190, 240))
        for y in range(rng.randint(40, 120), H - 20, pitch):
            cv2.line(img, (0, y), (W, y), c, 1)
    occ = np.zeros((H, W), np.uint8)
    x0, y, out = rng.randint(20, 90), rng.randint(20, 120), []
    for it in items:
        lh = it["ink"].shape[0] / max(len(it["boxes"]), 1)
        k = pitch * rng.uniform(0.75, 0.95) / lh
        h, w = int(it["ink"].shape[0] * k), int(it["ink"].shape[1] * k)
        if w >= W - x0 - 40:
            k *= (W - x0 - 40) / w
            h, w = int(it["ink"].shape[0] * k), W - x0 - 40
        if y + h >= H - 20:
            break
        if h >= 4 and w >= 4:
            a = cv2.resize(it["ink"], (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            x = x0 + rng.randint(0, 25)
            img[y:y + h, x:x + w] = (img[y:y + h, x:x + w] * (1 - a[..., None] * (1 - it["rgb"][None, None]))).astype(np.uint8)
            out += [(it["key"], (x + b0 * k, y + b1 * k, x + b2 * k, y + b3 * k)) for b0, b1, b2, b3 in it["boxes"]]
        y += max(h, pitch) + rng.randint(0, pitch // 3)
    return img, [], out


def degrade(img, rng):
    H, W = img.shape[:2]
    out = img
    if rng.random() < 0.5:
        t = rng.uniform(0, 0.25)
        g = np.linspace(-1, 1, 64, dtype=np.float32)
        m = 1 - rng.uniform(0, 0.15) * (rng.uniform(-1, 1) * g[None] + rng.uniform(-1, 1) * g[:, None]) ** 2
        m = cv2.resize(m, (W, H))[..., None] * np.array((1, 1 - 0.3 * t, 1 - 0.8 * t), np.float32)
        out = cv2.multiply(out, m, dtype=cv2.CV_8U)
    if rng.random() < 0.4:
        n = np.empty((H, W), np.int16)
        cv2.randn(n, 0, rng.uniform(2, 10))
        out = cv2.add(out.astype(np.int16), cv2.merge([n, n, n]), dtype=cv2.CV_8U)
    if rng.random() < 0.4:
        out = cv2.LUT(out, (255 * (np.arange(256) / 255) ** rng.uniform(0.6, 1.6)).astype(np.uint8))
    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.4, 1.0))
    if rng.random() < 0.3:
        out = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    if rng.random() < 0.2:
        f = rng.uniform(0.4, 0.75)
        out = cv2.resize(cv2.resize(out, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), (W, H), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.15:
        g = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        out = cv2.cvtColor(np.where(g < rng.randint(140, 200), 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    return out


_W = {}


def init_worker(bank_files):
    from hwsynth.cli import coverage
    hand = {f"script:{k}" for k, v in coverage().items() if v == "hand"}
    _W["items"] = [it for f in bank_files for it in pickle.load(open(f, "rb"))
                   if "ink" in it and it["key"] != "mark" and (not it["key"].startswith("script:") or it["key"] in hand)]
    _W["short"] = [it for it in _W["items"] if len(it["boxes"]) == 1]


def background(key, bgs):
    if _W.get("bg_key") != key:
        rng = random.Random(key)
        r = None
        while r is None:
            p = rng.choice(bgs)
            try:
                r = render_bg(p, rng)
            except Exception:
                r = None
        pm = np.zeros(r[0].shape[:2], np.uint8)
        for x0, y0, x1, y1 in r[1]:
            pm[int(y0):int(y1) + 1, int(x0):int(x1) + 1] = 1
        _W["bg_key"], _W["bg"] = key, (p.name, r[0], r[1], r[2], *free_mask(r[0], r[1]), pm)
    return _W["bg"]


def write_page(d, i, img, hw, pboxes, rng):
    H, W = img.shape[:2]
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "images" / f"{i:05d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, rng.randint(70, 95)])
    rows = []
    for c, bs in ((0, hw), (1, pboxes)):
        for x0, y0, x1, y1 in bs:
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
            if x1 - x0 > 2 and y1 - y0 > 2:
                rows.append(f"{c} {(x0 + x1) / 2 / W:.5f} {(y0 + y1) / 2 / H:.5f} {(x1 - x0) / W:.5f} {(y1 - y0) / H:.5f}")
    tmp = d / "labels" / f"{i:05d}.tmp"
    tmp.write_text("\n".join(rows))
    tmp.replace(d / "labels" / f"{i:05d}.txt")


def make_page(args):
    i, split, bgs = args
    d = YOLO / split
    if (d / "labels" / f"{i:05d}.txt").exists():
        return None
    rng = random.Random(i * 7 + (split == "val"))
    items = rng.sample(_W["items"], 40)
    if split == "train" and i >= STACK_FROM:
        img, pboxes, hw = stack_page(rng, [it for it in _W["items"][rng.randrange(len(_W["items"]) - 400):][:400]
                                         if it["key"] != "mark" and it["ink"].shape[1] * len(it["boxes"]) > 8 * it["ink"].shape[0]])
        src = "stack"
    elif rng.random() < 0.08:
        img, pboxes, hw = letter_page(rng, items[:rng.randint(6, 30)])
        src = "letter"
    else:
        src, img, pboxes, fields, occ, rules, pm = background(f"{split}{i // REUSE}", bgs)
        img, occ, pm = img.copy(), occ.copy(), pm.copy()
        hts = sorted(b[3] - b[1] for b in pboxes)
        lh = hts[len(hts) // 2]
        hw = []
        spots = fields + [(x0, y0 - 2 * lh, x1, y0 + 1) for x0, y0, x1, _ in rules]
        if spots and rng.random() < 0.75:
            rng.shuffle(spots)
            for f in spots[:max(1, int(len(spots) * rng.uniform(0.3, 1.0)))]:
                for it in rng.sample(_W["short"], 3):
                    if b := fill_field(img, pm, it, f, lh, rng):
                        hw += [(it["key"], bb) for bb in b]
                        x0, y0, x1, y1 = map(int, b[0])
                        occ[max(y0, 0):y1, max(x0, 0):x1] = 1
                        break
        n = 0 if rng.random() < 0.2 else rng.choice([rng.randint(1, 4), rng.randint(3, 20)])
        for it in items[:n]:
            lines = max(len(it["boxes"]), 1)
            line_h = it["ink"].shape[0] / lines
            b = paste(img, occ, it, max(rng.uniform(0.8, 2.4) * lh / line_h, 0.15), rng)
            if b:
                hw += [(it["key"], bb) for bb in b]
    img = degrade(img, rng)
    write_page(d, i, img, [b for _, b in hw], pboxes, rng)
    return dict(i=i, src=src, hw=[k for k, _ in hw])


def cmd_pages(a):
    bank_files = sorted(BANK.glob("*.pkl"))
    bgs = backgrounds()
    tr, va = [p for p in bgs if not is_val(p)], [p for p in bgs if is_val(p)]
    for split, n, b in (("val", a.val, va), ("train", a.n, tr)):
        tasks = [(i, split, b) for i in range(n)]
        with ProcessPoolExecutor(a.workers, initializer=init_worker, initargs=(bank_files,)) as ex, open(YOLO / f"{split}_meta.jsonl", "a") as f:
            for m in ex.map(make_page, tasks, chunksize=4 * REUSE):
                if m:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
                    f.flush()
    (YOLO / "data.yaml").write_text("path: .\ntrain: train/images\nval: val/images\nnames:\n  0: handwriting\n  1: print\n")
    print("pages:", len(list((YOLO / "train/labels").glob("*.txt"))), "train,", len(list((YOLO / "val/labels").glob("*.txt"))), "val")


def iou(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    i = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    ar = lambda q: (q[:, 2] - q[:, 0]) * (q[:, 3] - q[:, 1])
    return i / np.minimum(ar(a)[:, None], ar(b)[None])


def group(key):
    if key.startswith("script:"):
        return "rare"
    return {"zh": "cjk", "ja": "cjk", "ko": "cjk", "ar": "arabic", "fa": "arabic", "ur": "arabic"}.get(key, key if key == "mark" else "other")


def cmd_eval(a):
    from ultralytics import YOLO as Y
    m = Y(a.model, task="detect")
    pred = lambda img: m.predict(img, imgsz=LONG, conf=a.conf, verbose=False)[0].boxes
    stats = {}
    for line in open(YOLO / "val_meta.jsonl"):
        r = json.loads(line)
        img = cv2.imread(str(YOLO / "val/images" / f"{r['i']:05d}.jpg"))
        H, W = img.shape[:2]
        gt = [list(map(float, l.split())) for l in (YOLO / "val/labels" / f"{r['i']:05d}.txt").read_text().splitlines()]
        g = np.array([[(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H] for c, x, y, w, h in gt if c == 0]).reshape(-1, 4)
        b = pred(img)
        p = b.xyxy.cpu().numpy()[b.cls.cpu().numpy() == 0]
        ov = iou(g, p)
        fn = (ov.max(1) < 0.5) if ov.size else np.ones(len(g), bool)
        fp = int(((ov.max(0) < 0.5) if ov.size else np.ones(len(p), bool)).sum())
        keys = r["hw"] if len(r["hw"]) == len(g) else ["other"] * len(g)
        gs = {group(k) for k in keys} or {"none"}
        for gname in gs:
            s = stats.setdefault(gname, dict(pages=0, fp=[], fn=[]))
            s["pages"] += 1
            s["fp"].append(fp)
            s["fn"].append(int(sum(f for f, k in zip(fn, keys) if group(k) == gname)))
    print(f"{'group':8s} pages  FP/page mean max >2   FN/page mean max >2")
    for k, s in sorted(stats.items()):
        fp, fn = np.array(s["fp"]), np.array(s["fn"])
        print(f"{k:8s} {s['pages']:5d}  {fp.mean():.2f} {fp.max():3d} {(fp > 2).sum():3d}       {fn.mean():.2f} {fn.max():3d} {(fn > 2).sum():3d}")
    rng = random.Random(5)
    for d in ("books_pdf", "forms", "handwritten"):
        fs = rng.sample(sorted((LAYOUTS / d).glob("*.*")), a.real)
        n = []
        for f in fs:
            try:
                if f.suffix == ".pdf":
                    with pymupdf.open(f) as doc:
                        px = doc[0].get_pixmap(dpi=DPI, alpha=False)
                    img = cv2.cvtColor(np.frombuffer(px.samples, np.uint8).reshape(px.h, px.w, px.n)[:, :, :3], cv2.COLOR_RGB2BGR)
                else:
                    img = cv2.imread(str(f))
                b = pred(img)
                n.append(int((b.cls.cpu().numpy() == 0).sum()))
            except Exception:
                continue
        n = np.array(n)
        print(f"real {d:12s} {len(n)} pages: hw boxes/page mean {n.mean():.2f}, pages with >2: {(n > 2).sum()}, with 0: {(n == 0).sum()}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("text")
    p = sp.add_parser("bank")
    p.add_argument("--n", type=int, default=16000)
    p.add_argument("--workers", type=int, default=10)
    p = sp.add_parser("pages")
    p.add_argument("--n", type=int, default=6000)
    p.add_argument("--val", type=int, default=600)
    p.add_argument("--workers", type=int, default=2)
    p = sp.add_parser("para")
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--workers", type=int, default=2)
    p = sp.add_parser("paras")
    p.add_argument("--start", type=int, default=9000)
    p.add_argument("--workers", type=int, default=2)
    p = sp.add_parser("eval")
    p.add_argument("model")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--real", type=int, default=150)
    a = ap.parse_args()
    {"text": cmd_text, "bank": cmd_bank, "para": cmd_para, "paras": cmd_paras, "pages": cmd_pages, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
