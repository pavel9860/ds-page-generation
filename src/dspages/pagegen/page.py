"""One handwritten page of a layout"""

import logging
import random
import unicodedata
from collections import Counter
from dataclasses import dataclass

from hwsynth.scripts import text_script

from . import paper
from .canvas import Canvas, Hand, readable
from .config import MIN_COVER, ROW_MM, RULED, GenCfg  # noqa: F401
from .config import PAGE_MARGIN as MARGIN
from .layouts import LAYOUTS

log = logging.getLogger(__name__)

WORDS, GLYPHS, LAST_WORDS = 100, 400, 50


@dataclass
class HandwrittenPage:
    img: object
    words: int
    glyphs: int
    coverage: float
    row_mm: float
    failures: Counter


def is_rtl(texts):
    letters = [c for t in texts[:20] for c in t if c.isalpha()]
    return bool(letters) and sum(unicodedata.bidirectional(c) in ("R", "AL") for c in letters) > len(letters) / 2


def enough(c: Canvas, last: bool):
    """A page holds enough when it has 100 words or 400 glyphs; on the last attempt 50 words are accepted."""
    return c.coverage() >= MIN_COVER and (c.words >= WORDS or c.glyphs >= GLYPHS or (last and c.words >= LAST_WORDS))


def handwritten_page(layout, pool, cfg: GenCfg, rng: random.Random, tries=2) -> HandwrittenPage | None:
    """A page of the layout; each try draws size, fill, writer and texts afresh."""
    for _ in range(tries):
        if page := _page(layout, pool, cfg, rng):
            return page
    return None


def _page(layout, pool, cfg: GenCfg, rng: random.Random) -> HandwrittenPage | None:
    """The row height is the largest the layout can hold"""
    fill = rng.uniform(0.3, 1.0)
    target = rng.randint(WORDS, 2 * WORDS)
    ruled = rng.random() < RULED[layout]
    first_mm = rng.uniform(15, 30)
    pool = {k: rng.sample(v, len(v)) for k, v in pool.items()}
    sample = pool.get("para") or pool.get("item") or pool.get("heading") or []
    if not sample:
        return None
    script = Counter(text_script(t) for t in sample[:30]).most_common(1)[0][0]

    def canvas(row, dry):
        c = Canvas(cfg, rng, ruled_mm=1.55 * row if ruled else None, first_mm=first_mm, dry=dry)
        c.y_max = c.H * (0.07 + 0.86 * fill)
        return c

    lo, hi = ROW_MM
    c = canvas(lo, True)
    hand = Hand(cfg, rng, c.px_mm, c.W - 20, lo, c)
    hand.rtl, hand.script = is_rtl(sample), script
    hand.align = rng.choices(["start", "justify"], [0.6, 0.4])[0]
    hand.calibrate(sample[0])
    if hand.model is None:
        return None
    pick = rng.random()
    state = rng.getstate()

    def lay(row, dry):
        nonlocal fill
        rng.setstate(state)
        c = canvas(row, dry)
        rng.setstate(state)
        hand.canvas, hand.row_mm, hand.dry, hand.drift = c, row, dry, 0.0
        LAYOUTS[layout](hand, pool, target, rng)
        hand.dry = False
        return c

    def holds(row):
        c = lay(row, True)
        return c.coverage() >= MIN_COVER * 1.05 and (c.words >= MARGIN * WORDS or c.glyphs >= MARGIN * GLYPHS)

    if not holds(lo):
        fill = 1.0
        if not holds(lo):
            return None
    top = hi
    if not holds(hi):
        top = lo
        for step in (0.5, 0.25, 0.125, 0.0625):
            if holds(top + step * (hi - lo)):
                top += step * (hi - lo)
    row_mm = lo + pick * (top - lo)
    for last in (False, True):
        c = lay(row_mm, False)
        log.debug(
            "%s: row %.1f mm (max %.1f), fill %.2f, %d words, %d glyphs, cover %.2f, failures %s",
            layout,
            row_mm,
            top,
            fill,
            c.words,
            c.glyphs,
            c.coverage(),
            dict(hand.failures),
        )
        if enough(c, last):
            img = readable(paper.degrade(c.img, rng, cfg, cfg.p_scan_hw), c.img)
            return HandwrittenPage(img, c.words, c.glyphs, c.coverage(), row_mm, hand.failures)
        got = max(c.words / WORDS, c.glyphs / GLYPHS, 0.3)
        row_mm = max(lo, row_mm * min(0.95, 0.95 * got**0.5))
        fill = 1.0
    return None
