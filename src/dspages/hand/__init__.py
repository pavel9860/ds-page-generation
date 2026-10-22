"""Handwritten pages: real digital-born text, with its structure, written by hwsynth writers on papers of many kinds.

make(plan, items) -> (rgb image, metadata): one page; see `python -m dspages.hand` for the pipeline (texts -> plan -> run).
"""


def make(plan, items):
    from .page import make as _make
    return _make(plan, items)
