"""Font discovery by real cmap coverage."""
import glob
import os
import pickle
from functools import lru_cache
from pathlib import Path

from fontTools.ttLib import TTFont


# Font family-name blocklists are a losing game -- font collections ship
# dozens of single-script fonts (e.g. NotoSansElbasan, an extinct
# Albanian script) whose names don't obviously signal "not general body
# text". Only trust a named allowlist of known general-purpose families,
# verified against their real cmap (not a rendered-bitmap heuristic,
# which a font can satisfy with generic placeholder glyphs).
_TRUSTED_FAMILY_TOKENS = (
    "dejavusans", "dejavuserif", "liberationsans", "liberationserif", "liberationmono",
    "notosans-", "notoserif-", "notomono", "freeserif", "freesans", "freemono",
    "roboto", "opensans", "lato-", "ptsans", "ptserif", "sourcesans", "sourceserif",
    "arial", "times", "verdana", "tahoma", "calibri", "cambria", "georgia", "consolas", "ubuntu",
)
_REQUIRED_CODEPOINTS = tuple(ord(c) for c in "AaZz09АаЯяЁёΑαωω")
_MIN_SCRIPT_CMAP_SIZE = 40


_CMAP_CACHE = {}
_CMAP_FILE = Path("~/.cache/dspages/cmaps.pkl").expanduser()
_cmap_dirty = False


def _load_cmaps():
    """Font cmaps cached on disk as {path: (mtime, cmap)}; reading every font's cmap takes seconds."""
    if not _CMAP_CACHE and _CMAP_FILE.exists():
        try:
            with open(_CMAP_FILE, "rb") as f:
                _CMAP_CACHE.update(pickle.load(f))
        except Exception:
            pass


def _save_cmaps():
    global _cmap_dirty
    if _cmap_dirty:
        _CMAP_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _CMAP_FILE.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(_CMAP_CACHE, f)
        os.replace(tmp, _CMAP_FILE)
        _cmap_dirty = False


def _font_cmap(path: str) -> set:
    global _cmap_dirty
    _load_cmaps()
    mtime = os.path.getmtime(path)
    hit = _CMAP_CACHE.get(path)
    if hit is None or hit[0] != mtime:
        try:
            cmap = frozenset(TTFont(path, fontNumber=0, lazy=True).getBestCmap() or ())
        except Exception:
            cmap = frozenset()
        hit = _CMAP_CACHE[path] = (mtime, cmap)
        _cmap_dirty = True
    return hit[1]


def _is_general_font(path: str) -> bool:
    name = os.path.basename(path).lower()
    if not any(tok in name for tok in _TRUSTED_FAMILY_TOKENS):
        return False
    cmap = _font_cmap(path)
    return all(cp in cmap for cp in _REQUIRED_CODEPOINTS)


def _is_script_font(path: str) -> bool:
    """Any Noto Sans/Serif script-specific family (Bengali, Devanagari,
    Arabic, CJK, ...) with real, non-trivial glyph coverage -- accepted
    without hand-listing every script, since _pick_font checks actual
    text codepoints against each font's own cmap at render time."""
    name = os.path.basename(path).lower()
    if not (name.startswith("notosans") or name.startswith("notoserif")):
        return False
    return len(_font_cmap(path)) >= _MIN_SCRIPT_CMAP_SIZE


@lru_cache(maxsize=4)
def find_fonts(font_dirs: tuple) -> tuple:
    """.ttf/.ttc/.otf files of known families, verified via their real cmap: Latin + Cyrillic + Greek for the
    general body-text families, any real glyph coverage for Noto's script-specific families."""
    dirs = [os.path.expanduser(d) for d in font_dirs]
    candidates = sorted(p for d in dirs if os.path.isdir(d) for ext in ("ttf", "ttc", "otf")
                        for p in glob.glob(f"{d}/**/*.{ext}", recursive=True))
    found = tuple(p for p in candidates if _is_general_font(p) or _is_script_font(p))
    _save_cmaps()
    if not found:
        raise FileNotFoundError(f"no supported fonts under {dirs}")
    return found
