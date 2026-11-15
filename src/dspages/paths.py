"""Data locations, from the environment (or a .env file in the working directory or DSPAGES_ENV); nothing else names a path.
DSPAGES_ROOT      data root (Layouts/, Labels/, GridPages/, handwriting/ ...), default ./data
DSPAGES_CORPUS    corpus_collecting checkout (page-type classifier, URL index, language models), default <root>/corpus_collecting
DSPAGES_LAYOUTS   the PDF corpus, default <root>/Layouts
DSPAGES_HWDET_WEIGHTS  handwriting detector weights (hwdet-s), default models/hwdet-s-1.0.0.pt"""

import os
from pathlib import Path


def _load_env(path):
    if path.is_file():
        for line in path.read_text().splitlines():
            k, sep, v = line.strip().partition("=")
            if sep and not k.startswith("#"):
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


for _f in (os.environ.get("DSPAGES_ENV", ".env"), Path(__file__).resolve().parents[2] / ".env"):
    _load_env(Path(_f))


def env(name, default, *legacy):
    """Path from DSPAGES_<name> (or a legacy variable), else default."""
    for k in (f"DSPAGES_{name}", *legacy):
        if os.environ.get(k):
            return Path(os.environ[k])
    return Path(default)


ROOT = env("ROOT", "data", "DSPAGES_UVTM", "MULTILANG_ROOT", "HWDET_UVTM")
CORPUS = env("CORPUS", ROOT / "corpus_collecting", "MULTILANG_CORPUS")
LAYOUTS = env("LAYOUTS", ROOT / "Layouts", "MULTILANG_LAYOUTS", "HWDET_LAYOUTS")
INFO = LAYOUTS / "archive" / "info"
HWDET_WEIGHTS = env("HWDET_WEIGHTS", "models/hwdet-s-1.0.0.pt")
