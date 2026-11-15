"""Data locations of multilang, under dspages.paths.ROOT (MULTILANG_* / HWDET_* variables still override)."""

from ..paths import CORPUS, LAYOUTS, ROOT, env

WORK = env("WORK", ROOT / "handwriting", "MULTILANG_WORK", "HWDET_WORK")
DATA = env("DATA", WORK / "handwriting_detection", "MULTILANG_DATA", "HWDET_DATA")
LANGDS = DATA / "langds"
GENERATION = env("GENERATION", WORK / "generation", "MULTILANG_GENERATION", "HWDET_GENERATION")
TRANSLATION = GENERATION / "translation"
FILLING = GENERATION / "handwriting" / "forms"
MODELS = env("MODELS", "models", "MULTILANG_MODELS")
RESULTS = env("RESULTS", LAYOUTS / "test", "MULTILANG_RESULTS")
UVTM = ROOT
__all__ = [
    "CORPUS",
    "LAYOUTS",
    "ROOT",
    "UVTM",
    "WORK",
    "DATA",
    "LANGDS",
    "GENERATION",
    "TRANSLATION",
    "FILLING",
    "MODELS",
    "RESULTS",
]
