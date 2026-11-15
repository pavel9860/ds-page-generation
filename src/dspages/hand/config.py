"""Paths of the handwritten-pages pipeline. Every one can be set by an environment variable; nothing else in the package names a path."""

import os
from dataclasses import dataclass

from ..paths import CORPUS, ROOT


@dataclass(frozen=True)
class HandPaths:
    uvtm: str = str(ROOT)
    labels: str = os.environ.get("DSPAGES_PAGETYPE_LABELS", str(CORPUS / "pagetype/data/labels.json"))

    @property
    def layouts(self):
        return f"{self.uvtm}/Layouts"

    @property
    def refine(self):
        """Folders moved out of Layouts (same relative paths): certificates, flyers_clean, tables_clean, ..."""
        return os.environ.get("DSPAGES_REFINE", f"{self.uvtm}/Refine_DS")

    @property
    def manifest(self):
        """SQLite manifest of the corpus: plan (documents), lang_res (languages), old_meta (topics)."""
        return f"{self.layouts}/manifest.sqlite"

    @property
    def doc_types(self):
        """Page type of every document from the embedding classifier (jsonl: path, label, p)."""
        return f"{self.uvtm}/Labels/embedding/emb_out/doc_types.jsonl"

    @property
    def tessdata(self):
        return os.environ.get("DSPAGES_TESSDATA", f"{self.layouts}/archive/tessdata")


PATHS = HandPaths()


# ocr
MIN_CONF = 82  # mean word confidence of a line
NO_SPACE = {"th": (0x0E00, 0x0E7F), "lo": (0x0E80, 0x0EFF), "km": (0x1780, 0x17FF), "my": (0x1000, 0x109F)}

# page
SCRIPT_SCALE = {
    "Latin": 1.0,
    "Cyrillic": 1.0,
    "Greek": 1.0,
    "Han": 0.85,
    "Hangul": 0.85,
    "Hiragana": 0.85,
}  # other scripts: 0.7

# paper
FORMATS = {"a4": 210.0, "a5": 148.0, "b5": 176.0}  # sheet widths [mm]: the page always fills the 1024 x 1440 image
TONES = (
    (0.985, 0.985, 0.98),
    (0.97, 0.965, 0.94),
    (0.96, 0.95, 0.90),
    (0.98, 0.97, 0.93),
    (0.93, 0.93, 0.92),
    (0.99, 0.99, 1.0),
    (0.95, 0.95, 0.97),
)
LEGAL = ((0.98, 0.93, 0.62), (0.97, 0.91, 0.55), (0.99, 0.95, 0.72))
RULE = {
    "blue": (0.55, 0.70, 0.90),
    "gray": (0.66, 0.66, 0.68),
    "green": (0.62, 0.80, 0.66),
    "purple": (0.72, 0.62, 0.86),
    "cyan": (0.55, 0.82, 0.90),
}

# plan
ENGLISH = 0.40
GROUPS = {
    "cyrillic": 0.18,
    "cjk": 0.12,
    "other": 0.18,
    "latin": 0.12,
}  # shares of all pages (LayoutCfg.script_mix); English is the Latin rest
IAM_SHARE = 0.35  # pages written by a persona fitted to a real writer
FMT_W = {"a4": 0.55, "a5": 0.25, "b5": 0.20}  # sheet formats: A4 and notebook pages
PAPER_BY = {  # (notebook, mode) -> paper kinds and shares
    (False, "normal"): {
        "blank": 0.16,
        "lined": 0.20,
        "squared": 0.14,
        "dotted": 0.06,
        "legal": 0.06,
        "seyes": 0.08,
        "margin": 0.14,
        "spiral": 0.16,
    },
    (False, "dense"): {"blank": 0.25, "dotted": 0.15, "squared": 0.15, "lined": 0.25, "margin": 0.10, "spiral": 0.10},
    (False, "table"): {"blank": 0.4, "dotted": 0.2, "squared": 0.4},
    (True, "normal"): {"lined": 0.30, "margin": 0.20, "spiral": 0.22, "squared": 0.12, "seyes": 0.10, "dotted": 0.06},
    (True, "dense"): {"lined": 0.35, "margin": 0.20, "spiral": 0.20, "squared": 0.15, "dotted": 0.10},
    (True, "table"): {"squared": 0.5, "dotted": 0.2, "lined": 0.3},
}
XH_RANGE = {
    (False, "normal"): (2.0, 3.0),
    (False, "dense"): (2.0, 2.3),
    (False, "table"): (2.0, 2.4),  # x-height [mm] on paper without rules; font size 4 mm = 2 mm x-height at least
    (True, "normal"): (2.0, 3.4),
    (True, "dense"): (2.0, 2.4),
    (True, "table"): (2.0, 2.6),
}
KIND_W = {
    "book": 0.22,
    "paper": 0.22,
    "article": 0.16,
    "letter": 0.14,
    "magazine": 0.10,
    "lecture": 0.15,
    "recipe": 0.005,
    "poem": 0.005,
}

# rare
LOST = 0.5  # garbled: more than this share of the blocks' tags lost in the translation (tools/seed_texts.whole)
PER_DOC = 2  # pages tried per document (a document still gives one page to the set)

# structure
MAX_COLS = 4

# texts
SYMBOL_BULLET = {
    0xB7: "\u2022",
    0xA7: "\u25aa",
    0xD8: "\u27a2",
    0xFC: "\u2713",
    0xA8: "\u25fb",
    0xE0: "\u2192",
    0xB0: "\u00b0",
    0xB1: "\u00b1",
}

# util
CHAR_W = {
    "Latin": 1.8,
    "Cyrillic": 2.0,
    "Greek": 2.2,
    "Han": 3.6,
    "Hiragana": 3.6,
    "Katakana": 3.6,
    "Hangul": 3.8,
    "Arabic": 2.5,
    "Devanagari": 2.6,
    "Bengali": 2.4,
    "Tamil": 2.3,
    "Telugu": 1.95,
    "Thai": 1.4,
}  # x-heights per character, measured
MIN_XH_SCRIPT = {"Han": 2.3, "Hangul": 2.3, "Arabic": 2.0}  # complex glyphs smear below these x-heights
MAX_GLYPH_MM = 15.0  # tallest word of a page (ascender to descender)
