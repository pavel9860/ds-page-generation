"""Data locations of multilang. Set MULTILANG_ROOT (corpus root) or the single overrides; defaults sit under ./data."""
import os
from pathlib import Path


def _env(name, default):
    return Path(os.environ.get(f"MULTILANG_{name}") or os.environ.get(f"HWDET_{name}") or default)


ROOT = _env("ROOT", os.environ.get("HWDET_UVTM", "data"))
WORK = _env("WORK", ROOT / "handwriting")
DATA = _env("DATA", WORK / "handwriting_detection")
LANGDS = DATA / "langds"
LAYOUTS = _env("LAYOUTS", ROOT / "Layouts")
GENERATION = _env("GENERATION", WORK / "generation")
TRANSLATION = GENERATION / "translation"
FILLING = GENERATION / "handwriting" / "forms"
CORPUS = _env("CORPUS", ROOT / "corpus_collecting")            # page-type classifier and its item list
MODELS = _env("MODELS", "models")                                # detector weights (picture filter)
RESULTS = _env("RESULTS", LAYOUTS / "test")
UVTM = ROOT                                                      # legacy name
