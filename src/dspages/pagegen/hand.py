"""Handwriting of forms and notes: one writer at one font size."""

import random

import numpy as np
from hwsynth.config import GlyphCfg
from hwsynth.fonts import em_per_xh
from hwsynth.scripts import text_script

from .canvas import PAGE_MM, Hand, Strip, count_words
from .config import PAGE, GenCfg


def line_pitch(prints):
    """Distance between consecutive printed lines"""
    ys = sorted((b[1] + b[3]) / 2 for b in prints)
    gaps = [b - a for a, b in zip(ys, ys[1:]) if b - a > 2]
    return float(np.percentile(gaps, 30)) if gaps else 14.0


def writing_height(prints, page_h, rng: random.Random):
    """Font height in px of the handwriting on a printed page of height `page_h`"""
    pitch = line_pitch(prints)
    h = rng.uniform(*PAGE.pitch_share) * pitch
    if h < PAGE.min_hw * page_h:
        h = PAGE.min_hw * page_h * (1 + rng.uniform(0, PAGE.min_jitter))
    return min(h, PAGE.max_hw * page_h)


class FontHand(Hand):
    """`em_px` is the font height on a page `page_h` px high (A4 proportions)."""

    def __init__(self, cfg: GenCfg, rng: random.Random, page_h, max_width, em_px):
        px_mm = page_h / PAGE_MM[1]
        super().__init__(cfg, rng, px_mm, max_width, em_px / px_mm)

    def _raw(self, text):
        """The ink of the line cropped to its extent, at page resolution."""
        if text not in self.cache:
            import hwsynth as H

            written = self._cover(text)
            script = text_script(written)
            script = "Latin" if script == "Common" else script
            raw = None
            font = self.writer.font(script)
            if font is None:
                self.failures["no font"] += 1
            else:
                xh = self.row_mm / em_per_xh(font, script, GlyphCfg().px)
                self.writer.p["size"] = xh
                pen = self.writer.pen  # hwsynth writes no x-height below 5 pen widths / 4 pen edges (page.write): small
                pen.width, pen.edge = (
                    min(pen.width, xh / 5),
                    min(pen.edge, xh / 4),
                )  # writing takes a finer pen instead of growing
                s = H.write(
                    written,
                    self.writer,
                    width_mm=min(1800.0, 14 + 14.0 * len(text)),
                    dpi=self.px_mm * 25.4,
                    ruled=False,
                    scan=False,
                    image=False,
                )
                ys, xs = np.nonzero(s.ink > self.cfg.crop_ink)
                if len(ys):
                    raw = s.ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
                elif any(c.isdigit() for c in text) and self._localize(text) != text:
                    raw = self._raw(self._localize(text))
                else:
                    self.failures["no ink"] += 1
            self.cache[text] = raw
        return self.cache[text]

    def strip(self, text) -> Strip | None:
        text = text.strip()
        ink = self._raw(text) if text else None
        if ink is None or min(ink.shape) < 5 or ink.shape[1] > self.max_width:
            return None
        return Strip(np.clip(ink, 0, 1).astype(np.float32) ** 0.75, count_words(text), len("".join(text.split())))

    def char_px(self, sample: str):
        """Mean width in px of one character, from a sample line."""
        if "char_px" not in self.cache:
            st = self.strip(sample[:40])
            self.cache["char_px"] = st.alpha.shape[1] / max(1, len(sample[:40])) if st is not None else 10.0
        return self.cache["char_px"]

    @property
    def pen(self):
        """Stroke width in px."""
        return max(1.0, self.writer.pen.width * self.px_mm)
