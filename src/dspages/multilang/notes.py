"""What a reader writes on a printed page, in one hand"""
import random
from dataclasses import replace

import cv2
import numpy as np
from hwsynth.pagegen.hand import FontHand, writing_height

from . import dettext as T
from . import linebank as LB
from . import pages
from .config import LINEBANK, Config

RTL = {"Arabic", "Hebrew", "Syriac", "Thaana"}
PAD = 1
EDGE_CHARS = ".,;:!?()[]\"'«»“”"
EDGE = 8
MIN_PRINTED_WORD = 4
SLOT_H = 1.5
MAX_NOTE = 44
MIN_NOTE = 5


def _free(occ, x, y, w, h):
    H, W = occ.shape
    return EDGE <= x and EDGE <= y and x + w <= W - EDGE and y + h <= H - EDGE and not occ[y - PAD:y + h + PAD, x - PAD:x + w + PAD].any()


def _tight(alpha, x, y, thr=0.2):
    ys, xs = np.nonzero(alpha > thr)
    return (x + int(xs.min()), y + int(ys.min()), x + int(xs.max()) + 1, y + int(ys.max()) + 1) if len(ys) else None


class Annotator:
    def __init__(self, img, prints, words, lang, script, cfg: Config, rng: random.Random):
        self.img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else img.copy()
        self.prints, self.words, self.lang, self.script, self.cfg, self.rng = prints, words, lang, script, cfg, rng
        self.occ, _ = pages.free_mask(self.img, prints, replace(cfg, occ_dilate=1))
        self.lh = float(np.median([min(b[3] - b[1], b[2] - b[0]) for b in prints]))
        self.H, self.W = self.img.shape[:2]
        self.height = writing_height(prints, self.H, rng)
        self.hand = FontHand(cfg, rng, self.H, self.W - 2 * EDGE, self.height)
        self.hand.script, self.hand.rtl = script, script in RTL
        self.pool = T.clean_pool(lang, script) or T.clean_pool(lang)
        bank = LB.load(lang)
        self.lines = bank.lines if bank is not None and len(bank) >= LINEBANK.min_lines else sorted((T.cut(t, 3, lang) for t in self.pool), key=len)
        self.char_px = self.hand.char_px(next((l for l in self.lines if self.hand.strip(l) is not None), "x" * 20))
        self.printed = {t for w in words for t in map(lambda x: x.strip(EDGE_CHARS), w[4].casefold().split()) if len(t) >= MIN_PRINTED_WORD}
        self._usable = {}
        self.used = set()
        self.boxes, self.ink = [], self.hand.rgb

    def usable(self, line):
        """A line that may be written"""
        if line not in self._usable:
            words = {t.strip(EDGE_CHARS) for t in line.casefold().split()}
            self._usable[line] = (len(line.strip()) > MIN_NOTE and not any(c.isdigit() for c in line) and not words & self.printed
                                  and self.hand.writer.fonts.writes(line))
        return self._usable[line]

    def slots(self, b):
        """(free width px, side, x, y) beside a printed line b"""
        g, h = max(4, int(0.4 * self.lh)), int(SLOT_H * self.height)
        cy, rtl = int((b[1] + b[3]) / 2), self.hand.rtl

        def right(x, y):
            return self.free_width(self.occ[y - PAD:y + h + PAD, x:self.W - EDGE], h, y)

        def left(x, y):
            return self.free_width(self.occ[y - PAD:y + h + PAD, EDGE:x][:, ::-1], h, y)

        top = cy - h // 2
        out = [(right(int(b[2]) + g, top), "after", int(b[2]) + g, top), (left(int(b[0]) - g, top), "before", int(b[0]) - g, top)]
        for side, y in (("under", int(b[3]) + 2), ("over", int(b[1]) - h - 2)):
            x = int(b[2]) if rtl else int(b[0])
            out.append((left(x, y) if rtl else right(x, y), side, x, y))
        return out

    def free_width(self, strip, h, y):
        """Free width of a strip of the occupancy mask cut for a row of height h at y (0 when the row leaves the page)."""
        if y < EDGE or y + h > self.H - EDGE or not strip.size:
            return 0
        busy = strip.any(axis=0)
        return int(busy.argmax()) if busy.any() else strip.shape[1]

    def note(self, b):
        """One note beside the printed line b: the widest free slot, a line of the language that fits it."""
        width, side, x, y = max(self.slots(b), key=lambda s: (s[0], self.rng.random()))
        cap = int(width / self.char_px * 0.9)
        fits = [l for l in self.lines if MIN_NOTE < len(l) <= min(cap, MAX_NOTE) and self.usable(l) and LB.key(l) not in self.used]
        if not fits:
            return None
        line = self.rng.choice(fits[-max(1, len(fits) // 3):])
        self.used.add(LB.key(line))
        s = self.hand.strip(line)
        if s is None or s.alpha.shape[1] > width:
            return None
        h, w = s.alpha.shape
        rtl = self.hand.rtl
        if side in ("under", "over"):
            x, y = (x - w if rtl else x), (y if side == "under" else int(b[1]) - h - 2)
        elif side == "after":
            y = int((b[1] + b[3]) / 2 - h / 2)
        else:
            x, y = x - w, int((b[1] + b[3]) / 2 - h / 2)
        if not _free(self.occ, x, y, w, h):
            return None
        pages.blend(self.img, s.alpha, self.ink, x, y, self.hand.density)
        self.occ[y - PAD:y + h + PAD, x - PAD:x + w + PAD] = 1
        box = _tight(s.alpha, x, y)
        if box:
            self.boxes.append(box)
        return box

    def words_of(self, b):
        return [w for w in self.words if b[0] - 2 <= (w[0] + w[2]) / 2 <= b[2] + 2 and b[1] - 2 <= (w[1] + w[3]) / 2 <= b[3] + 2]

    def stroke(self, pts):
        a = np.zeros((self.H, self.W), np.float32)
        p = np.array([[x, y] for x, y in pts], np.float32)
        cv2.polylines(a, [np.round(p * 4).astype(np.int32)], False, 1.0, max(1, round(self.hand.pen)), cv2.LINE_AA, shift=2)
        pages.blend(self.img, a, self.ink, 0, 0, self.hand.density)

    def mark(self):
        r = self.rng
        b = r.choice(self.prints)
        kind = r.choice(("underline", "underline", "circle", "bar", "tick"))
        x0, y0, x1, y1 = b
        if kind == "underline":
            a = x0 + r.uniform(0, 0.4) * (x1 - x0)
            e = a + r.uniform(0.3, 1.0) * (x1 - a)
            xs = np.linspace(a, e, 24)
            self.stroke([(x, y1 + 2 + r.uniform(-0.6, 0.6) + 0.02 * (x - a) * r.uniform(-0.05, 0.05)) for x in xs])
        elif kind == "circle":
            ws = self.words_of(b) or [b]
            w = r.choice(ws)
            cx, cy, rx, ry = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2, (w[2] - w[0]) / 2 + 5, (w[3] - w[1]) / 2 + 4
            t = np.linspace(0.2, 2 * np.pi + 0.5, 40)
            self.stroke([(cx + rx * np.cos(u) * r.uniform(0.97, 1.03), cy + ry * np.sin(u) * r.uniform(0.95, 1.05)) for u in t])
        elif kind == "bar":
            x = (x1 + 8) if self.hand.rtl else (x0 - 8)
            near = sorted(self.prints, key=lambda q: abs(q[1] - y0))[:r.randint(2, 5)]
            self.stroke([(x + r.uniform(-1, 1), min(q[1] for q in near)), (x + r.uniform(-2, 2), max(q[3] for q in near))])
        else:
            x, y, s = ((x1 + 6) if self.hand.rtl else (x0 - 6 - self.lh)), y0, self.lh
            self.stroke([(x, y + 0.5 * s), (x + 0.35 * s, y + s), (x + s, y)])

    def run(self, n_text, n_marks):
        order = list(self.prints)
        self.rng.shuffle(order)
        for b in order:
            if len(self.boxes) >= n_text:
                break
            self.note(b)
        for _ in range(n_marks):
            self.mark()
        return self.img, self.boxes


def annotate(img, prints, words, lang, script, cfg: Config, rng: random.Random, n_text=(5, 30), n_marks=(1, 6)):
    """(page RGB, handwriting boxes) of a printed page (grey or RGB uint8) with prints and text-layer words in px."""
    if len(prints) < 3:
        return img, []
    a = Annotator(img, prints, words, lang, script, cfg, rng)
    return a.run(rng.randint(*n_text), rng.randint(*n_marks))
