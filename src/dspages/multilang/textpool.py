"""Text pool: lines per language from the layout PDFs and written-by-hand style texts (translate/<code>.json)"""

import json
import pickle
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import pymupdf

from .config import Config


def lang_of(path):
    s = path.stem.rsplit("_", 1)[-1]
    return s if 2 <= len(s) <= 3 and s.isalpha() else None


def page_lines(page, min_font):
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            t = "".join(s["text"] for s in ln["spans"]).strip()
            if t and any(s["text"].strip() and s["size"] > min_font for s in ln["spans"]):
                yield t


def clean(t, cfg: Config):
    """A line of words: mostly letters, none of the characters that legacy-encoded or OCR text fills with."""
    body = [c for c in t if not c.isspace()]
    return (
        bool(body)
        and sum(c.isalpha() for c in body) / len(body) >= cfg.text_min_alpha
        and not set(t) & set(cfg.text_bad)
    )


def lang_lines(args):
    """Lines of the first PDFs of one language."""
    lang, files, cfg = args
    random.Random(lang).shuffle(files)
    lo, hi = cfg.text_line_len
    lines = []
    for p in files[: cfg.text_files_per_lang]:
        with pymupdf.open(p) as doc:
            for pg in list(doc)[: cfg.text_pages_per_file]:
                lines += [t for t in page_lines(pg, cfg.text_min_font) if lo <= len(t) <= hi]
    return lang, lines


def build(cfg: Config):
    from hwsynth.cli import coverage
    from hwsynth.scripts import text_script

    hand = {s for s, kind in coverage().items() if kind == "hand"}
    by = {}
    for d in cfg.bg_folders:
        for p in sorted((cfg.layouts / d).glob("*.pdf")):
            if (lang := lang_of(p)) and lang not in ("mul", "xx"):
                by.setdefault(lang, []).append(p)
    with ProcessPoolExecutor(cfg.workers) as ex:
        found = dict(ex.map(lang_lines, [(lang, files, cfg) for lang, files in sorted(by.items())]))
    pool, skip = {}, []
    for lang, lines in found.items():
        if len(lines) >= cfg.text_min_lines:
            script = Counter(text_script(t) for t in lines[:200]).most_common(1)[0][0]
            same = [t for t in lines if text_script(t) == script and clean(t, cfg)]
            if script not in hand:
                skip.append(lang)
            elif len(same) >= cfg.text_min_lines:
                pool[lang] = same[: cfg.text_max_lines]
    for code in cfg.no_font_langs:
        pool.pop(code, None)
    for f in sorted((cfg.data / "translate").glob("*.json")):
        pool[cfg.translated_key.get(f.stem, f.stem)] = json.loads(f.read_text())
    by_script = {}
    for lang, lines in pool.items():
        if not lang.startswith("script:"):
            by_script.setdefault(Counter(text_script(t) for t in lines[:100]).most_common(1)[0][0], []).extend(lines)
    for sc, lines in by_script.items():
        if sc in hand and f"script:{sc}" not in pool and sc not in cfg.language_scripts:
            pool[f"script:{sc}"] = random.Random(sc).sample(lines, min(len(lines), cfg.text_max_lines))
    cfg.data.mkdir(parents=True, exist_ok=True)
    pickle.dump(pool, open(cfg.text_pool, "wb"))
    (cfg.data / cfg.skip_langs).write_text(json.dumps(sorted(skip)))
    return pool


def load(cfg: Config):
    return pickle.load(open(cfg.text_pool, "rb"))
