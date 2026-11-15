"""Handwriting bank: hwsynth lines (one writer per group, one language per writer) and paragraph pages."""

import os
import pickle
import random

import cv2
import numpy as np

from dspages.pagegen.canvas import line_boxes, new_writer

from . import textpool
from .config import Config


def fill_text(rng, cfg: Config):
    """A date, phone, amount or number as in forms, or None."""
    if rng.random() >= cfg.bank_p_date:
        return None
    return rng.choice(
        [
            f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.{rng.randint(1950, 2026)}",
            f"{rng.randint(1, 12)}/{rng.randint(1, 28)}/{rng.randint(50, 99)}",
            f"+{rng.randint(1, 99)} {rng.randint(100, 999)} {rng.randint(100, 999)} {rng.randint(1000, 9999)}",
            f"{rng.randint(1, 99999)}.{rng.randint(0, 99):02d}",
            str(rng.randint(1, 999)),
            f"№ {rng.randint(1, 9999)}",
            f"{rng.randint(1, 23)}:{rng.randint(0, 59):02d}",
        ]
    )


def line_text(rng, lines, cfg: Config):
    t = rng.choice(lines)
    if rng.random() < cfg.bank_p_join:
        t += " " + rng.choice(lines)
    words = t.split()
    if len(words) > cfg.bank_crop_words[1] - 1 and rng.random() < cfg.bank_p_crop:
        i = rng.randrange(len(words))
        t = " ".join(words[i : i + rng.randint(*cfg.bank_crop_words)])
    return t


def bank_chunk(i, cfg: Config):
    """bank_chunk lines = bank_chunk // bank_group writers"""
    import hwsynth as H

    path = cfg.bank / f"{i:05d}.pkl"
    if path.exists():
        return 0
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    pool = textpool.load(cfg)
    langs = [k for k in pool if not k.startswith("script:")]
    rare = [k for k in pool if k.startswith("script:")]
    rng = random.Random(cfg.seed * 10**6 + i)
    items = []
    for g in range(cfg.bank_chunk // cfg.bank_group):
        key = rng.choice(rare if rng.random() < cfg.bank_p_rare else langs)
        w = new_writer(rng)
        for _ in range(cfg.marks_per_group if cfg.ticks else 0):
            s = H.figure(rng.choice(cfg.mark_kinds), w, size=rng.uniform(4, 12))
            ys, xs = np.nonzero(s.ink > cfg.box_ink)
            if len(ys):
                items.append(
                    dict(
                        ink=(s.ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] * 255).astype(np.uint8),
                        rgb=np.asarray(w.pen.ink, np.float32),
                        key="mark",
                        wid=f"{i}.{g}",
                        boxes=[(0, 0, xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min())],
                    )
                )
        for _ in range(cfg.bank_group):
            try:
                s = H.write(
                    fill_text(rng, cfg) or line_text(rng, pool[key], cfg),
                    w,
                    width_mm=rng.uniform(*cfg.bank_width_mm),
                    dpi=cfg.dpi,
                    ruled=False,
                    scan=False,
                    image=False,
                )
                boxes = line_boxes(s.words, s.ink, cfg)
            except Exception:
                continue
            ys, xs = np.nonzero(s.ink > cfg.crop_ink)
            if not len(ys) or not boxes:
                continue
            y0, x0 = ys.min(), xs.min()
            items.append(
                dict(
                    ink=(s.ink[y0 : ys.max() + 1, x0 : xs.max() + 1] * 255).astype(np.uint8),
                    rgb=np.asarray(w.pen.ink, np.float32),
                    key=key,
                    wid=f"{i}.{g}",
                    boxes=[(a - x0, b - y0, c - x0, d - y0) for a, b, c, d in boxes],
                )
            )
    write_chunk(path, items)
    return len(items)


def para_chunk(i, cfg: Config):
    import hwsynth as H

    path = cfg.para / f"{i:05d}.pkl"
    if path.exists():
        return 0
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    pool = textpool.load(cfg)
    langs = [k for k in pool if not k.startswith("script:")]
    rare = [k for k in pool if k.startswith("script:")]
    rng = random.Random(cfg.seed * 10**6 + 10**5 + i)
    items = []
    for _ in range(cfg.para_chunk):
        key = rng.choice(rare if rng.random() < cfg.para_p_rare else langs)
        text = " ".join(rng.choice(pool[key]) for _ in range(rng.randint(*cfg.para_sentences)))
        try:
            s = H.write(
                text, new_writer(rng), width_mm=rng.uniform(*cfg.para_width_mm), dpi=cfg.para_dpi, ruled=None, scan=None
            )
            boxes = line_boxes(s.words, s.ink, cfg)
        except Exception:
            continue
        if boxes:
            items.append(
                dict(
                    jpg=cv2.imencode(".jpg", cv2.cvtColor(s.image, cv2.COLOR_RGB2BGR))[1].tobytes(),
                    boxes=boxes,
                    key=key,
                )
            )
    write_chunk(path, items)
    return len(items)


def load_bank(cfg: Config, val: bool):
    """({key: {writer id: [items]}}, {writer id: [marks]}); chunk files with index % val_chunk_every == 0 are val."""
    by, marks = {}, {}
    for f in sorted(cfg.bank.glob("*.pkl")):
        if (int(f.stem) % cfg.val_chunk_every == 0) != val:
            continue
        for it in pickle.load(open(f, "rb")):
            if it["key"] == "mark":
                marks.setdefault(it["wid"], []).append(it)
            else:
                by.setdefault(it["key"], {}).setdefault(it["wid"], []).append(it)
    return by, marks


def write_chunk(path, obj):
    tmp = path.with_suffix(".tmp")
    pickle.dump(obj, open(tmp, "wb"))
    tmp.replace(path)
