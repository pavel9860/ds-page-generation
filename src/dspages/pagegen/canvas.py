"""A page canvas (paper at the page long side) and ink strips: handwritten lines scaled to one size per page."""

import random
import unicodedata
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np
from hwsynth.scripts import script as char_script

from .config import INK_DPI, MIN_CONTRAST, PAGE_MM, GenCfg  # noqa: F401


@dataclass
class Strip:
    alpha: np.ndarray
    words: int
    glyphs: int

    def with_alpha(self, alpha, counted=True):
        """The same text drawn as `alpha`; `counted=False` for a copy that must not add to the page word count."""
        return Strip(alpha, self.words if counted else 0, self.glyphs if counted else 0)


def native_digits(script):
    """{'0': native zero, ...} of the decimal digits a script has in Unicode (Khmer, Thai, Devanagari...), or {}."""
    out = {}
    for cp in range(0x0300, 0x1F000):
        c = chr(cp)
        d = unicodedata.decimal(c, None)
        if d is not None and d > 0 or (d == 0 and unicodedata.name(c, "").endswith("DIGIT ZERO")):
            name = unicodedata.name(c, "")
            if name.startswith(script.upper() + " DIGIT") or (
                script == "Arabic" and name.startswith("ARABIC-INDIC DIGIT")
            ):
                out[str(d)] = c
    return out


def contrast(img):
    """Gray levels between the paper (median) and the darkest ink (0.3th percentile) of a page."""
    gray = img.mean(axis=2)
    return float(np.median(gray) - np.percentile(gray, 0.3))


def readable(degraded, clean):
    """The page with its scan look, or the clean one when the look left the writing too faint to read."""
    return degraded if contrast(degraded) >= MIN_CONTRAST else clean


def count_words(text):
    """Whitespace words; a text written without spaces counts about 2 characters per word."""
    tokens = text.split()
    chars = sum(map(len, tokens))
    return len(tokens) if tokens and chars / len(tokens) < 12 else chars // 2


class Canvas:
    """Paper of A4 proportions with a long side of cfg.long_side px; `put` blends ink in the writer's pen colour."""

    def __init__(self, cfg: GenCfg, rng: random.Random, ruled_mm=None, first_mm=0.0, dry=False):
        self.cfg, self.rng, self.dry = cfg, rng, dry
        self.H = cfg.long_side
        self.W = int(self.H * PAGE_MM[0] / PAGE_MM[1])
        self.px_mm = self.H / PAGE_MM[1]
        if dry:
            self.img = None
        else:
            from hwsynth.page import paper

            sheet = paper(
                (self.H, self.W), np.random.default_rng(rng.getrandbits(32)), ruled_mm, self.px_mm * 25.4, first_mm
            )
            self.img = (np.clip(sheet, 0, 1) * 255).astype(np.uint8)
        self.rules = (first_mm * self.px_mm, ruled_mm * self.px_mm) if ruled_mm else None
        self.y_max = self.H * 0.93
        self.ink_rows = []
        self.words = self.glyphs = 0

    def put(self, strip: Strip, x, y, rgb, density=1.0):
        h, w = strip.alpha.shape
        if self.rules:
            first, pitch = self.rules
            y = first + max(0, round((y + h - first) / pitch)) * pitch - 0.12 * h - h
        x, y = int(x), int(y)
        if x < 0 or y < 0 or x + w > self.W or y + h > self.H:
            return False
        if not self.dry:
            region = self.img[y : y + h, x : x + w].astype(np.float32)
            region *= 1 - density * strip.alpha[..., None] * (1 - np.asarray(rgb, np.float32))
            self.img[y : y + h, x : x + w] = region.astype(np.uint8)
        self.ink_rows.append((x, y, x + w, y + h))
        self.words += strip.words
        self.glyphs += strip.glyphs
        return True

    def stroke(self, x0, y0, x1, y1, rgb, width, density=1.0):
        """A hand-drawn line: it wanders a little around the straight one and its width follows two slow pressure waves and thins at the end."""
        if self.dry:
            return
        n = max(3, int(np.hypot(x1 - x0, y1 - y0) / 5))
        t = np.linspace(0, 1, n)
        wander = np.cumsum([self.rng.gauss(0, 0.3) for _ in range(n)])
        wander = (wander - np.linspace(wander[0], wander[-1], n)) * 0.6
        waves = sum(
            self.rng.uniform(0.06, 0.12) * np.sin(2 * np.pi * self.rng.uniform(0.7, 2.5) * t + self.rng.uniform(0, 6.3))
            for _ in range(2)
        )
        w = np.clip(width * (1 + waves) * (1 - 0.3 * np.clip((t - 0.9) / 0.1, 0, 1)), 0.6, None)
        xs, ys = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t + wander
        pad = int(width * 3 + 4)
        bx0, by0 = int(max(min(xs) - pad, 0)), int(max(min(ys) - pad, 0))
        bx1, by1 = int(min(max(xs) + pad, self.W)), int(min(max(ys) + pad, self.H))
        if bx1 <= bx0 or by1 <= by0:
            return
        S = 4
        mask = np.zeros(((by1 - by0) * S, (bx1 - bx0) * S), np.uint8)
        for i in range(n - 1):
            cv2.line(
                mask,
                (round((xs[i] - bx0) * S), round((ys[i] - by0) * S)),
                (round((xs[i + 1] - bx0) * S), round((ys[i + 1] - by0) * S)),
                255,
                max(1, round(w[i] * S)),
                cv2.LINE_AA,
            )
        alpha = cv2.resize(mask, (bx1 - bx0, by1 - by0), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        region = self.img[by0:by1, bx0:bx1].astype(np.float32)
        region *= 1 - density * alpha[..., None] * (1 - np.asarray(rgb, np.float32))
        self.img[by0:by1, bx0:bx1] = region.astype(np.uint8)

    def coverage(self):
        """Share of the page area inside the bounding box of all placed ink."""
        if not self.ink_rows:
            return 0.0
        x0, y0 = min(r[0] for r in self.ink_rows), min(r[1] for r in self.ink_rows)
        x1, y1 = max(r[2] for r in self.ink_rows), max(r[3] for r in self.ink_rows)
        return (x1 - x0) * (y1 - y0) / (self.W * self.H)


def pen_colour(rng: random.Random, cfg: GenCfg):
    """Transmittance of a pen's ink: a blue of some shade (share cfg.p_blue_ink) or a black of some depth."""
    if rng.random() < cfg.p_blue_ink:
        c = np.array([rng.uniform(0.02, 0.18), rng.uniform(0.04, 0.26), rng.uniform(0.22, 0.6)], np.float32)
    else:
        c = np.array([rng.uniform(0.03, 0.15)] * 3, np.float32) + np.array(
            [rng.uniform(0, 0.04) for _ in range(3)], np.float32
        )
    return c * min(1.0, 0.22 / max(float(c @ [0.2126, 0.7152, 0.0722]), 1e-3))


class Hand:
    """One writer and one handwriting size for a page."""

    def __init__(
        self, cfg: GenCfg, rng: random.Random, px_mm: float, max_width: int, row_mm: float, canvas: Canvas | None = None
    ):
        self.cfg, self.rng, self.px_mm, self.max_width, self.row_mm, self.canvas = (
            cfg,
            rng,
            px_mm,
            max_width,
            row_mm,
            canvas,
        )
        self.writer = new_writer(rng, cfg)
        self.writer.pen.width *= cfg.pen_width_scale * float(np.exp(np.clip(rng.gauss(0, cfg.pen_width_sd), -1.0, 1.0)))
        self.rgb = pen_colour(rng, cfg)
        self.ref_row = None
        self.cache = {}
        self.failures = Counter()
        self.align, self.rtl, self.script = "start", False, "Latin"
        self.tilt = rng.gauss(0, 0.008)
        self._digits = None
        self.dry, self.model = False, None
        self._fonts, own = {}, self.writer.font
        self.writer.font = lambda scr, _own=own: self._fonts.get(scr) or _own(scr)

    def _cover(self, text):
        """Fonts for the scripts of the text the writer's own font class lacks"""
        new = {k: v for k, v in pick_fonts(self.writer, text).items() if k not in self._fonts}
        self._fonts.update(new)
        return strip_missing(self.writer, text)

    def _raw(self, text):
        import hwsynth as H

        if text not in self.cache:
            written = self._cover(text)
            raw = None
            try:
                s = H.write(
                    written,
                    self.writer,
                    width_mm=min(1800.0, 14 + 14.0 * len(text)),
                    dpi=INK_DPI,
                    ruled=False,
                    scan=False,
                    image=False,
                )
            except Exception as e:
                self.failures[type(e).__name__] += 1
            else:
                ys, xs = np.nonzero(s.ink > self.cfg.crop_ink)
                if len(ys):
                    boxes = line_boxes(s.words, s.ink, self.cfg)
                    row = float(np.median([b[3] - b[1] for b in boxes])) if boxes else float(ys.max() - ys.min() + 1)
                    raw = (s.ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1], row)
                elif any(c.isdigit() for c in text) and self._localize(text) != text:
                    raw = self._raw(self._localize(text))
                else:
                    self.failures["no ink"] += 1
            self.cache[text] = raw
        return self.cache[text]

    def _localize(self, text):
        """The text with its ASCII digits in the digits of the writing script (a font of one script has no 0-9)."""
        if self._digits is None:
            self._digits = native_digits(self.script)
        return "".join(self._digits.get(c, c) for c in text)

    def calibrate(self, sample: str):
        """Fixes the size from a representative text: a number or a mark written first would give a wrong scale."""
        text = sample[:60].strip()
        raw = self._raw(text)
        if raw is not None:
            self.ref_row = raw[1]
            self.model = (raw[0].shape[1] / max(1, len(text)) / raw[1], raw[0].shape[0] / raw[1])

    def fit_height(self, height):
        """Sets the size once, before any line is written"""
        if self.model:
            self.row_mm = height / (self.px_mm * self.model[1])

    def char_px(self, sample: str):
        """Mean width in px of one character, from a sample line."""
        if self.dry:
            return self.model[0] * self.row_mm * self.px_mm
        if "char_px" not in self.cache:
            st = self.strip(sample[:40])
            self.cache["char_px"] = st.alpha.shape[1] / max(1, len(sample[:40])) if st is not None else 10.0
        return self.cache["char_px"]

    def wrap(self, text: str, width: float):
        """The text cut into pieces that fit `width` px"""
        per = max(4, int(width / self.char_px(text) * 0.97))
        words = text.split()
        spaced = len(words) > 1 and sum(map(len, words)) / len(words) < 14
        out, cur = [], ""
        if spaced:
            for w in words:
                while len(w) > per:
                    out.append((cur + " " + w[: per - len(cur)]).strip() if cur else w[:per])
                    w, cur = w[per - len(cur) :] if cur else w[per:], ""
                if len(cur) + 1 + len(w) > per and cur:
                    out.append(cur)
                    cur = w
                else:
                    cur = (cur + " " + w).strip()
            return out + ([cur] if cur else [])
        i = 0
        while i < len(text):
            j = min(len(text), i + per)
            while j < len(text) and (
                unicodedata.category(text[j]).startswith("M") or text[j - 1] in "\u17d2\u0e3a\u1039"
            ):
                j += 1
            out.append(text[i:j])
            i = j
        return out

    @property
    def pen(self):
        """Stroke width in px of the writing on the page: the pen width at INK_DPI times the strip scale."""
        if self.ref_row is None:
            return max(1.0, 0.06 * self.row_mm * self.px_mm)
        return max(1.0, self.writer.pen.width * INK_DPI / 25.4 * self.row_mm * self.px_mm / self.ref_row)

    @property
    def density(self):
        return float(np.clip(self.writer.pen.density, 0.95, 1.0))

    def _dry_strip(self, text) -> Strip | None:
        """A blank strip of the size the text would have"""
        if not text:
            return None
        k = self.row_mm * self.px_mm
        h, w = max(1, round(self.model[1] * k)), max(1, round(len(text) * self.model[0] * k))
        if h < 5 or w < 5 or w > self.max_width:
            return None
        return Strip(np.zeros((h, w), np.float32), count_words(text), len("".join(text.split())))

    def strip(self, text) -> Strip | None:
        text = text.strip()
        if self.dry:
            return self._dry_strip(text)
        raw = self._raw(text) if text else None
        if raw is None:
            return None
        ink, row = raw
        self.ref_row = self.ref_row or row
        scale = self.row_mm * self.px_mm / self.ref_row
        h, w = max(1, round(ink.shape[0] * scale)), max(1, round(ink.shape[1] * scale))
        if h < 5 or w < 5 or w > self.max_width:
            return None
        alpha = cv2.resize(ink, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        return Strip(np.clip(alpha, 0, 1) ** 0.75, count_words(text), len("".join(text.split())))


def line_boxes(words, ink, cfg: GenCfg):
    """Boxes of text rows: words grouped by row, each box tight around the ink in its padded row."""
    rows = []
    for _, (x0, y0, x1, y1), _ in sorted(words, key=lambda w: (w[1][1] + w[1][3]) / 2):
        c = (y0 + y1) / 2
        if rows and abs(c - rows[-1][4]) < cfg.box_row_tol * (rows[-1][3] - rows[-1][1]):
            r = rows[-1]
            rows[-1] = [min(r[0], x0), min(r[1], y0), max(r[2], x1), max(r[3], y1), r[4]]
        else:
            rows.append([x0, y0, x1, y1, c])
    h, w = ink.shape
    out = []
    for x0, y0, x1, y1, _ in rows:
        pad = int(cfg.box_pad * (y1 - y0))
        X0, Y0, X1, Y1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, w), min(y1 + pad, h)
        ys, xs = np.nonzero(ink[Y0:Y1, X0:X1] > cfg.box_ink)
        if len(ys):
            out.append((X0 + xs.min(), Y0 + ys.min(), X0 + xs.max() + 1, Y0 + ys.max() + 1))
    return out


def new_writer(rng, cfg: GenCfg):
    """A writer of a natural person type and style (no medical, hurried, child or decorative hands)."""
    import hwsynth as H

    w = H.sample_writer(rng.choice(cfg.natural_persons), rng.choice(cfg.natural_styles), seed=rng.randrange(2**31))
    from .handfonts import Restricted

    w.fonts = Restricted(w.fonts)
    return w


COMMON = ("Common", "Inherited", "Unknown", None)


def script_of(ch):
    s = char_script(ch)
    return "Latin" if s in COMMON else s


def pick_fonts(w, text, prefer=None):
    """One font per script that has every character of the page"""
    by = {}
    for ch in set(text):
        if not ch.isspace():
            by.setdefault(script_of(ch), set()).add(ch)
    mapping = {}
    for sc, chars in by.items():
        own = w.font(sc)
        cands = (
            ([own] if own else []) + list(w.fonts.candidates(sc, w.font_class)) + list(w.fonts.candidates(sc, "any"))
        )
        name = (prefer or {}).get(sc)
        if name:
            cands = [f for f in cands if str(f).endswith(name)] + cands
        need = "".join(chars)
        mapping[sc] = next((f for f in cands if f and w.fonts.covers(f, need)), None)
        if mapping[sc] is None:
            cov = lambda f: sum(ord(c) in w.fonts.cmap.get(f, ()) for c in need)
            best = max((f for f in cands if f), key=cov, default=None)
            if best is not None and cov(best) >= 0.9 * len(need):
                mapping[sc] = best
    return {k: v for k, v in mapping.items() if v}


def strip_missing(w, text):
    """The text without the characters the writer's fonts lack (they would be empty boxes) and without orphaned vowel signs."""
    out, start = [], True
    for ch in text:
        if ch.isspace():
            out.append(ch)
            start = True
            continue
        cm = w.fonts.cmap.get(w.font(script_of(ch)), ())
        if (unicodedata.category(ch)[0] == "M" and (start or unicodedata.combining(out[-1]) == 9)) or (
            cm and ord(ch) not in cm
        ):
            start = True
            continue
        out.append(ch)
        start = False
    return "".join(out)
