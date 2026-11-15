"""Handwritten pages of dspages.hand (real structured text, 24 writer types, many papers) as detection data"""

import json
import random
from pathlib import Path

import cv2
import numpy as np

from . import pages
from .config import (
    COLLIDE,
    COLLIDE_SHARE,
    HAND_GAP,
    HW,
    MAX_COLLIDE,
    MIN_CONTRAST,
    MIN_WORD_PX,
    Config,
)  # noqa: F401
from .config import (
    HANDPAGES_SAME_LINE as SAME_LINE,
)


def line_boxes(words, gap_factor=HAND_GAP):
    """[(x0, y0, x1, y1)] line segments from word boxes"""
    med = float(np.median([w[3] - w[1] for w in words]))
    lines = []
    for w in sorted(words, key=lambda b: b[0]):
        yc, h = (w[1] + w[3]) / 2, max(w[3] - w[1], med)
        best = None
        for ln in lines:
            last = ln[-1]
            gap = w[0] - last[2]
            dy = abs(yc - (last[1] + last[3]) / 2)
            if (
                gap <= gap_factor * max(h, last[3] - last[1], med)
                and dy < SAME_LINE * max(h, last[3] - last[1])
                and (best is None or dy < best[0])
            ):
                best = (dy, ln)
        if best:
            best[1].append(w)
        else:
            lines.append([w])
    return [
        (min(b[0] for b in ln), min(b[1] for b in ln), max(b[2] for b in ln), max(b[3] for b in ln)) for ln in lines
    ]


def collisions(words):
    """Pairs of words of different lines overlapping by more than COLLIDE of the smaller box (neighbours on one line touch in scripts that split words at syllable marks, e.g."""
    a = np.array(words, float)
    cy, hh = (a[:, 1] + a[:, 3]) / 2, a[:, 3] - a[:, 1]
    n = 0
    for i in range(len(a)):
        w = np.minimum(a[i, 2], a[i + 1 :, 2]) - np.maximum(a[i, 0], a[i + 1 :, 0])
        h = np.minimum(a[i, 3], a[i + 1 :, 3]) - np.maximum(a[i, 1], a[i + 1 :, 1])
        small = np.minimum(
            (a[i, 2] - a[i, 0]) * (a[i, 3] - a[i, 1]), (a[i + 1 :, 2] - a[i + 1 :, 0]) * (a[i + 1 :, 3] - a[i + 1 :, 1])
        )
        other_line = np.abs(cy[i] - cy[i + 1 :]) > SAME_LINE * np.maximum(hh[i], hh[i + 1 :])
        n += int(((np.clip(w, 0, None) * np.clip(h, 0, None) > COLLIDE * np.maximum(small, 1)) & other_line).sum())
    return n


def reject_reason(img, meta, cfg: Config):
    """Why a page is unusable (None when it is fine): the first failed check of `finish`."""
    words = [
        tuple(w[1]) for w in meta["words"] if w[1][2] - w[1][0] > cfg.min_box_px and w[1][3] - w[1][1] > cfg.min_box_px
    ]
    if collisions(words) > COLLIDE_SHARE * len(words):
        return "colliding boxes"
    if np.median([w[3] - w[1] for w in words]) < MIN_WORD_PX:
        return "writing too small"
    gray = img.mean(axis=2)
    ink = np.concatenate([gray[int(w[1]) : int(w[3]), int(w[0]) : int(w[2])].ravel() for w in words[:400]])
    if np.median(gray) - np.percentile(ink, 3) < MIN_CONTRAST:
        return "faint ink"
    return None


def finish(img, meta, rng: random.Random, cfg: Config, scan=True):
    """(image, YOLO rows) of one generated page, or None when its boxes are unusable. img is RGB uint8."""
    words = [
        tuple(w[1]) for w in meta["words"] if w[1][2] - w[1][0] > cfg.min_box_px and w[1][3] - w[1][1] > cfg.min_box_px
    ]
    if collisions(words) > MAX_COLLIDE:
        return None
    H, W = img.shape[:2]
    gray = img.mean(axis=2)
    ink = np.concatenate([gray[int(w[1]) : int(w[3]), int(w[0]) : int(w[2])].ravel() for w in words[:400]])
    if np.median(gray) - np.percentile(ink, 3) < MIN_CONTRAST:
        return None
    if np.median([w[3] - w[1] for w in words]) < MIN_WORD_PX:
        return None
    boxes = line_boxes(words)
    if scan:
        img = pages.degrade(img, rng, cfg, cfg.p_scan_hw)
    return img, pages.yolo_rows([(HW, boxes)], W, H, cfg)


def convert(src, out, cfg: Config, val_every=10, seed=0):
    """Pages of a dspages.hand run (src/pages/*.jpg, src/meta_*.jsonl) -> out/{train,val}/{images,labels,meta}"""
    src, out = Path(src), Path(out)
    rng = random.Random(seed)
    kept = dropped = 0
    for f in sorted(src.glob("meta_*.jsonl")):
        for line in open(f):
            m = json.loads(line)
            img = cv2.imread(str(src / "pages" / f"{m['i']:05d}.jpg"))
            if img is None:
                continue
            res = finish(img[..., ::-1], m, rng, cfg)
            if res is None:
                dropped += 1
                continue
            split = "val" if pages.is_val(f"{m['doc']}_x") else "train"
            name = f"hand_{m['i']:06d}"
            for d in ("images", "labels", "meta"):
                (out / split / d).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(out / split / "images" / f"{name}.jpg"),
                np.ascontiguousarray(res[0][..., ::-1]),
                [cv2.IMWRITE_JPEG_QUALITY, rng.randint(*cfg.jpeg_quality)],
            )
            (out / split / "labels" / f"{name}.txt").write_text("\n".join(res[1]) + "\n")
            (out / split / "meta" / f"{name}.json").write_text(
                json.dumps(
                    dict(
                        src="hand:" + m["kind"],
                        lang=m["lang"],
                        persona=m["persona"],
                        paper=m["paper"],
                        doc=m["doc"],
                        hw={m["lang"]: len(res[1])},
                        print=0,
                    ),
                    ensure_ascii=False,
                )
            )
            kept += 1
    return kept, dropped
