"""Page composition: hwsynth lines on real PDF pages (forms filled, text gaps) and on handwritten pages"""
import hashlib
import json
import random
from dataclasses import dataclass, replace

import cv2
import numpy as np
import pymupdf
from hwsynth.pagegen.paper import blend, degrade, free_mask, page_segments

from . import bank
from .config import HW, PRINT, Config


@dataclass
class Bg:
    img: np.ndarray
    prints: list
    fields: list
    labels: list
    occ: np.ndarray
    rules: list
    pmask: np.ndarray
    checks: list


def is_val(name):
    """Split by the content id of a PDF (file name up to the first `_`), so that its pages are in one split."""
    return int(hashlib.md5(name.split("_")[0].encode()).hexdigest(), 16) % 10 == 0


def render_bg(path, rng, cfg: Config):
    """A page of the PDF with its print boxes in pixels, or None if it is mostly picture or has too little text."""
    with pymupdf.open(path) as doc:
        pg = doc[rng.randrange(min(len(doc), cfg.text_pages_per_file))]
        area = abs(pg.rect)
        if any(abs(pymupdf.Rect(i["bbox"]) & pg.rect) > cfg.max_picture_cover * area for i in pg.get_image_info()):
            return None
        prints, fields, labels = page_segments(pg, cfg)
        lo, hi = cfg.mark_box_pt
        checks = [tuple(d["rect"]) for d in pg.get_drawings()
                  if lo <= d["rect"].width <= hi and lo <= d["rect"].height <= hi
                  and cfg.mark_aspect[0] <= d["rect"].width / d["rect"].height <= cfg.mark_aspect[1]] if cfg.ticks else []
        k = cfg.long_side / max(pg.rect.width, pg.rect.height)
        px = pg.get_pixmap(matrix=pymupdf.Matrix(k, k), alpha=False)
    img = np.frombuffer(px.samples, np.uint8).reshape(px.h, px.w, px.n)[:, :, :3].copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def seen(b):
        r = gray[max(int(b[1] * k), 0):int(b[3] * k) + 1, max(int(b[0] * k), 0):int(b[2] * k) + 1]
        return r.size and np.ptp(r) >= cfg.seen_contrast

    prints = [tuple(v * k for v in b) for b in prints if seen(b)]
    if not cfg.min_prints <= len(prints) <= cfg.max_prints:
        return None
    occ, rules = free_mask(img, prints, cfg)
    pmask = np.zeros(img.shape[:2], np.uint8)
    for x0, y0, x1, y1 in prints:
        pmask[int(y0):int(y1) + 1, int(x0):int(x1) + 1] = 1
    sc = lambda bs: [tuple(v * k for v in b) for b in bs]
    overlaps = lambda c: any(c[0] < q[2] and c[2] > q[0] and c[1] < q[3] and c[3] > q[1] for q in prints)
    return Bg(img, prints, sc(fields), sc(labels), occ, rules, pmask, [c for c in sc(checks) if not overlaps(c)])


def scaled_alpha(ink, k):
    """Ink alpha resized by k (<= 1: handwriting is never enlarged, so it is as sharp as the print around it)."""
    assert k <= 1.0 + 1e-9, k
    h, w = int(ink.shape[0] * k), int(ink.shape[1] * k)
    if h < 4 or w < 4:
        return None
    return cv2.resize(ink, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


def paste(img, occ, it, scale, rng, tries, region, cfg: Config):
    """Item on free paper inside region (x0, y0, x1, y1) or the whole page; returns its boxes or None."""
    H, W = img.shape[:2]
    rx0, ry0, rx1, ry1 = map(int, region or (0, 0, W, H))
    if region:
        scale = min(scale, (rx1 - rx0) / it["ink"].shape[1], (ry1 - ry0) / it["ink"].shape[0])
    a = scaled_alpha(it["ink"], scale)
    if a is None or a.shape[0] >= H or a.shape[1] >= W:
        return None
    h, w = a.shape
    for _ in range(tries):
        y = rng.randint(max(ry0, 0), max(min(ry1, H) - h, ry0, 0))
        x = rng.randint(max(rx0, 0), max(min(rx1, W) - w, rx0, 0))
        if y + h <= H and x + w <= W and occ[y:y + h, x:x + w].mean() < cfg.free_cover:
            blend(img, a, it["rgb"], x, y, rng.uniform(*cfg.ink_density))
            occ[y:y + h, x:x + w] = 1
            return [(x + b0 * scale, y + b1 * scale, x + b2 * scale, y + b3 * scale) for b0, b1, b2, b3 in it["boxes"]]
    return None


def line_h(it):
    return it["ink"].shape[0] / len(it["boxes"])


def pick(rng, items, target, used):
    """An item not used yet whose line is at least `target` px tall"""
    free = [it for it in items if id(it) not in used] or items
    ok = [it for it in free if line_h(it) >= target]
    it = rng.choice(ok) if ok else max(free, key=line_h)
    used.add(id(it))
    return it


def fit(it, target):
    """Scale that brings the item's line to `target` px; never above 1 (no enlargement)."""
    return min(target / line_h(it), 1.0)


def fill_field(img, pmask, it, field, target, rng, cfg: Config):
    """One-line item on a field (leader or rule): baseline near the field bottom, may cross the dots."""
    fx0, fy0, fx1, fy1 = field
    ink = it["ink"]
    k = fit(it, target)
    if ink.shape[1] * k > fx1 - fx0:
        k = (fx1 - fx0) / ink.shape[1]
    a = scaled_alpha(ink, k)
    if a is None or a.shape[0] < 5 or a.shape[1] < 5:
        return None
    h, w = a.shape
    H, W = img.shape[:2]
    x = int(fx0 + rng.uniform(0, max(fx1 - fx0 - w, 0)))
    y = int(fy1 - h + rng.uniform(*cfg.field_dy) * h)
    if x < 0 or y < 0 or x + w >= W or y + h >= H or pmask[y:y + h, x:x + w].mean() > cfg.field_overlap:
        return None
    blend(img, a, it["rgb"], x, y, rng.uniform(*cfg.field_density))
    pmask[y:y + h, x:x + w] = 1
    return [(x + b0 * k, y + b1 * k, x + b2 * k, y + b3 * k) for b0, b1, b2, b3 in it["boxes"]]


def print_strip(bg: Bg, rng, cfg: Config):
    """Header or footer rows of a real page with their print boxes, or None."""
    H = bg.img.shape[0]
    top = rng.random() < cfg.p_strip_top
    sel = [b for b in bg.prints if (b[3] < cfg.strip_header * H if top else b[1] > cfg.strip_footer * H)]
    if not sel:
        return None
    if top:
        y0, y1 = 0, int(max(b[3] for b in sel)) + cfg.strip_pad
    else:
        y0, y1 = max(int(min(b[1] for b in sel)) - cfg.strip_pad, 0), H
    return dict(img=bg.img[y0:y1], boxes=[(a, b - y0, c, d - y0) for a, b, c, d in sel], top=top)


def fit_strip(strip, W, H, cfg: Config):
    """Strip scaled to width W, or None if it would take more than strip_max of a page of height H."""
    h, w = strip["img"].shape[:2]
    k = W / w
    if h * k > cfg.strip_max * H:
        return None
    return dict(img=cv2.resize(strip["img"], (W, max(int(h * k), 1)), interpolation=cv2.INTER_AREA),
                boxes=[tuple(v * k for v in b) for b in strip["boxes"]], top=strip["top"])


def put_strip(img, strip):
    """Strip onto the page edge; returns its boxes in page coordinates and the rows it takes."""
    h = strip["img"].shape[0]
    y = 0 if strip["top"] else img.shape[0] - h
    img[y:y + h] = strip["img"]
    return [(x0, y0 + y, x1, y1 + y) for x0, y0, x1, y1 in strip["boxes"]], (y, y + h)


def yolo_rows(boxes_by_class, W, H, cfg: Config):
    rows = []
    for c, boxes in boxes_by_class:
        for x0, y0, x1, y1 in boxes:
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
            if x1 - x0 > cfg.min_box_px and y1 - y0 > cfg.min_box_px:
                rows.append(f"{c} {(x0 + x1) / 2 / W:.5f} {(y0 + y1) / 2 / H:.5f} {(x1 - x0) / W:.5f} {(y1 - y0) / H:.5f}")
    return rows


def draw(rng, items, k):
    """k distinct items (fewer if the pool is small)."""
    return rng.sample(items, min(k, len(items)))


class Composer:
    """Builds the pages of one split in a worker: bank items of that split's writers, backgrounds of its PDFs."""

    def __init__(self, cfg: Config, split, pdfs):
        self.cfg, self.split, self.pdfs = cfg, split, pdfs
        self.by_key, self.marks = bank.load_bank(cfg, split == "val")
        self.paras = bank.load_para(cfg) if split == "train" else []
        self._bg = (None, None)
        self.pictures = sorted(p for p in cfg.pictures.glob("*.png") if is_val(p.name) == (split == "val"))

    def background(self, key):
        if self._bg[0] != key:
            rng = random.Random(key)
            bg = None
            while bg is None:
                path = rng.choice(self.pdfs)
                try:
                    bg = render_bg(path, rng, self.cfg)
                except Exception:
                    bg = None
            self._bg = (key, (path.name, bg))
        name, bg = self._bg[1]
        return name, replace(bg, img=bg.img.copy(), occ=bg.occ.copy(), pmask=bg.pmask.copy())

    def pool(self, rng):
        """Items of one page: a language/script drawn uniformly, then one writer (two with p 0.3)."""
        writers = self.by_key[rng.choice(list(self.by_key))]
        n = 2 if rng.random() < self.cfg.bank_p_join else 1
        ids = rng.sample(list(writers), min(len(writers), n))
        items = [it for w in ids for it in writers[w]]
        if rng.random() < self.cfg.p_second_lang:
            other = self.by_key[rng.choice(list(self.by_key))]
            items += other[rng.choice(list(other))]
        return items, [m for w in ids for m in self.marks.get(w, [])]

    def make(self, i):
        """Writes page i; returns its meta or None if it exists."""
        cfg = self.cfg
        d = cfg.ds / self.split
        if (d / "labels" / f"{i:05d}.txt").exists():
            return None
        rng = random.Random(cfg.seed * 10**7 + i * 2 + (self.split == "val"))
        if i >= cfg.n_train:
            img, prints, hw, src = self.para_page(i - cfg.n_train, rng)
        else:
            img, prints, hw, src = self.compose(i, rng)
        img = degrade(img, rng, cfg, cfg.p_scan_hw if hw else cfg.p_scan_text)
        H, W = img.shape[:2]
        for sub in ("images", "labels", "meta"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "images" / f"{i:05d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, rng.randint(*cfg.jpeg_quality)])
        meta = dict(i=i, src=src, hw=[k for k, _ in hw], print=len(prints))
        (d / "meta" / f"{i:05d}.json").write_text(json.dumps(meta, ensure_ascii=False))
        tmp = d / "labels" / f"{i:05d}.tmp"
        tmp.write_text("\n".join(yolo_rows(((HW, [b for _, b in hw]), (PRINT, prints)), W, H, cfg)))
        tmp.replace(d / "labels" / f"{i:05d}.txt")
        return meta

    def compose(self, i, rng):
        cfg = self.cfg
        u = rng.random()
        if u < cfg.p_picture and self.pictures:
            return self.picture_page(rng)
        kind = rng.random()
        items, marks = self.pool(rng)
        if u < cfg.p_picture + cfg.p_blank:
            src, bg = self.background(f"{self.split}{i // cfg.bg_reuse}")
            return bg.img, bg.prints, [], "blank:" + src
        if kind < cfg.p_full:
            return self.full_page(i, rng, items)
        for attempt in range(cfg.bg_attempts):
            src, bg = self.background(f"{self.split}{i // cfg.bg_reuse}" + (f"r{attempt}" if attempt else ""))
            img, prints, hw, src = self.on_background(src, bg, items, marks, kind, rng)
            if hw:
                break
        return img, prints, hw, src

    def picture_page(self, rng):
        """A picture without text or handwriting (multilang.pictures)."""
        img = cv2.cvtColor(cv2.imread(str(rng.choice(self.pictures))), cv2.COLOR_BGR2RGB)
        return img, [], [], "picture"

    def on_background(self, src, bg, items, marks, kind, rng):
        """A form filled in or text gaps filled; its boxes and source tag."""
        cfg = self.cfg
        lh = sorted(b[3] - b[1] for b in bg.prints)[len(bg.prints) // 2]
        H, W = bg.img.shape[:2]
        spots = bg.fields + [(x0, y0 - cfg.rule_above * lh, x1, y0 + 1) for x0, y0, x1, _ in bg.rules]
        for x0, y0, x1, y1 in bg.labels:
            right = [q[0] for q in bg.prints if q[0] > x1 and min(q[3], y1) - max(q[1], y0) > 0.5 * (y1 - y0)]
            r = min(right + [W - 20]) - 4
            if r - x1 > cfg.label_min_width * lh:
                spots.append((x1 + cfg.label_gap * lh, y0, r, y1))
        hw = []
        if kind < cfg.p_full + cfg.p_form and (spots or bg.checks):
            src = "form:" + src
            if cfg.ticks:
                hw += self.tick(bg, marks, rng)
            short = [it for it in items if len(it["boxes"]) == 1] or items
            used = set()
            rng.shuffle(spots)
            for f in spots[:max(1, int(len(spots) * rng.uniform(*cfg.field_fill)))]:
                target = rng.uniform(*cfg.field_scale) * lh
                for _ in range(cfg.field_tries):
                    it = pick(rng, short, target, used)
                    if b := fill_field(bg.img, bg.pmask, it, f, target, rng, cfg):
                        hw += [(it["key"], bb) for bb in b]
                        x0, y0, x1, y1 = map(int, b[0])
                        bg.occ[max(y0, 0):y1, max(x0, 0):x1] = 1
                        break
        else:
            src = "gap:" + src
            region = (min(b[0] for b in bg.prints), min(b[1] for b in bg.prints),
                      max(b[2] for b in bg.prints), max(b[3] for b in bg.prints))
            used = set()
            for _ in range(rng.randint(*cfg.gap_items)):
                target = rng.uniform(*cfg.gap_scale) * lh
                it = pick(rng, items, target, used)
                if b := paste(bg.img, bg.occ, it, fit(it, target), rng, cfg.gap_tries, region, cfg):
                    hw += [(it["key"], bb) for bb in b]
        return bg.img, bg.prints, hw, src

    def tick(self, bg, marks, rng):
        """Checks and crosses in a share of the page's checkboxes."""
        cfg = self.cfg
        out = []
        if not marks:
            return out
        p = rng.uniform(*cfg.mark_fill)
        for x0, y0, x1, y1 in bg.checks:
            if rng.random() >= p:
                continue
            m = rng.choice(marks)
            side = max(x1 - x0, y1 - y0)
            k = min(side * rng.uniform(*cfg.mark_scale) / max(m["ink"].shape), 1.0)
            a = scaled_alpha(m["ink"], k)
            if a is None:
                continue
            h, w = a.shape
            x = int((x0 + x1) / 2 - w / 2 + rng.uniform(-1, 1) * cfg.mark_shift * side)
            y = int((y0 + y1) / 2 - h / 2 + rng.uniform(-1, 1) * cfg.mark_shift * side)
            if x < 0 or y < 0 or x + w >= bg.img.shape[1] or y + h >= bg.img.shape[0]:
                continue
            blend(bg.img, a, m["rgb"], x, y, rng.uniform(*cfg.field_density))
            out.append(("mark", (x, y, x + w, y + h)))
        return out

    def full_page(self, i, rng, items):
        cfg = self.cfg
        from hwsynth.page import paper
        H, W = cfg.long_side, int(cfg.long_side * rng.uniform(*cfg.page_aspect))
        letter = rng.random() < cfg.p_letter
        pitch = max(min(rng.randint(*cfg.stack_pitch), int(max(map(line_h, items)) / cfg.stack_scale[1])), 10)
        px_mm = cfg.scan_dpi / 25.4
        start = rng.randint(20, 120)
        ruled = not letter and rng.random() < cfg.stack_p_ruled
        sheet = paper((H, W), np.random.default_rng(rng.getrandbits(32)), pitch / px_mm if ruled else None,
                      cfg.scan_dpi, (start + 0.8 * pitch) / px_mm)
        img = (np.clip(sheet, 0, 1) * 255).astype(np.uint8)
        strip = None
        if rng.random() < cfg.p_strip and (s := print_strip(self.background(f"{self.split}s{i}")[1], rng, cfg)):
            strip = fit_strip(s, W, H, cfg)
        prints, top, bottom = [], 0, H
        if strip:
            prints, (y0, y1) = put_strip(img, strip)
            top, bottom = (y1, H) if strip["top"] else (0, y0)
        if letter:
            return img, prints, self.scatter(img, rng, items, top, bottom), "letter"
        return img, prints, self.stack(img, rng, items, top, bottom, pitch, start), "stack"

    def scatter(self, img, rng, items, top, bottom):
        cfg = self.cfg
        H, W = img.shape[:2]
        occ = np.zeros((H, W), np.uint8)
        m = cfg.page_margin
        occ[:max(m, top + cfg.strip_pad)], occ[min(H - m, bottom - cfg.strip_pad):], occ[:, :m], occ[:, -m:] = 1, 1, 1, 1
        hw = []
        for it in draw(rng, items, rng.randint(*cfg.letter_items)):
            if b := paste(img, occ, it, rng.uniform(*cfg.letter_scale), rng, cfg.letter_tries, None, cfg):
                hw += [(it["key"], bb) for bb in b]
        return hw

    def stack(self, img, rng, items, top, bottom, pitch, start):
        """Lines one under another at a pitch, on ruled or blank paper, like a letter or notebook page."""
        cfg = self.cfg
        H, W = img.shape[:2]
        x0, y = rng.randint(20, 90), max(start, top + rng.randint(10, 50))
        multi = [it for it in items if it["ink"].shape[1] * len(it["boxes"]) > cfg.stack_min_aspect * it["ink"].shape[0]]
        hw = []
        used = set()
        for _ in range(cfg.stack_candidates):
            target = pitch * rng.uniform(*cfg.stack_scale)
            it = pick(rng, multi if len(multi) >= cfg.stack_min_items else items, target, used)
            k = min(fit(it, target), (W - x0 - 40) / it["ink"].shape[1])
            a = scaled_alpha(it["ink"], k)
            if a is None:
                continue
            h, w = a.shape
            if y + h >= bottom - 20:
                break
            x = x0 + rng.randint(0, 25)
            blend(img, a, it["rgb"], x, y, 1.0)
            hw += [(it["key"], (x + b0 * k, y + b1 * k, x + b2 * k, y + b3 * k)) for b0, b1, b2, b3 in it["boxes"]]
            y += max(h, pitch) + rng.randint(0, pitch // 3)
        return hw

    def para_page(self, j, rng):
        """A hwsynth paragraph page, padded to page aspect, with an optional printed header or footer."""
        cfg = self.cfg
        it = self.paras[j]
        img = cv2.cvtColor(cv2.imdecode(np.frombuffer(it["jpg"], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        H = max(h, int(w * rng.uniform(*cfg.para_aspect)))
        top = rng.randint(0, H - h)
        fill = np.median(img.reshape(-1, 3)[::97], 0).astype(np.uint8)
        img = np.vstack([np.full((top, w, 3), fill), img, np.full((H - h - top, w, 3), fill)])
        hw = [(it["key"], (x0, y0 + top, x1, y1 + top)) for x0, y0, x1, y1 in it["boxes"]]
        prints = []
        if rng.random() < cfg.p_strip:
            strip = print_strip(self.background(f"p{j}")[1], rng, cfg)
            strip = fit_strip(strip, w, img.shape[0], cfg) if strip else None
            if strip:
                sh = strip["img"].shape[0]
                if strip["top"]:
                    img = np.vstack([strip["img"], img])
                    hw = [(k, (x0, y0 + sh, x1, y1 + sh)) for k, (x0, y0, x1, y1) in hw]
                    prints = strip["boxes"]
                else:
                    prints = [(x0, y0 + img.shape[0], x1, y1 + img.shape[0]) for x0, y0, x1, y1 in strip["boxes"]]
                    img = np.vstack([img, strip["img"]])
        k = cfg.long_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        scale = lambda b: tuple(v * k for v in b)
        return img, [scale(b) for b in prints], [(key, scale(b)) for key, b in hw], "para"
