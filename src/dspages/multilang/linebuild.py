"""Builds langds/linebank/<lang>.json from the digital text pages of the corpus (language from the file name)."""
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pymupdf
from hwsynth.pagegen.source import well_formed

from .buckets import BUCKETS
from .config import LANGDS as D
from .config import LINEBANK as C
from .langcheck import clean_ok, repair
from .linebank import BANK, key, phrases
from .realpages import parse_lang, text_layer_ok
from .translate import paragraphs


def scan(files, lang, pages, rng, seen, lines):
    """Adds the lines of up to `pages` digital pages of `files`; returns the pages used."""
    got = tried = 0
    for path in files:
        if got >= pages or tried >= C.tried_factor * pages:
            break
        with pymupdf.open(path) as doc:
            for i in random.Random(path).sample(range(len(doc)), min(len(doc), C.page_samples)):
                tried += 1
                pg = doc[i]
                if got >= pages or pg.rect.width > C.max_page_width or len(pg.get_text("text")) < C.min_page_chars or not text_layer_ok(pg):
                    continue
                got += 1
                for p in paragraphs(pg, C.max_paragraphs):
                    text = repair(p["text"])
                    if p["role"] in ("p", "li") and clean_ok(text, lang, lexicon=False) and well_formed(text):
                        for line in phrases(text, lang):
                            k = key(line)
                            if k and k not in seen:
                                seen.add(k)
                                lines.append(" ".join(line.split()))
    return got


def lang_bank(job):
    """(layouts, lang, out_dir, pages, seed) -> (lang, lines)"""
    layouts, lang, out, pages, seed = job
    f = Path(out) / f"{lang}.json"
    if f.exists():
        return lang, len(json.loads(f.read_text())["w"])
    rng, root = random.Random(f"{lang}{seed}"), Path(layouts)
    files = lambda dirs: [str(root / d / n) for d in dirs if (root / d).is_dir() for n in os.listdir(root / d)
                          if n.lower().endswith(".pdf") and parse_lang(n) == lang]
    own = files(D.folders["text"])
    rng.shuffle(own)
    seen, lines = set(), []
    got = scan(own, lang, pages, rng, seen, lines)
    wide = len(lines) < C.min_lines
    if wide:
        more = [x for x in files(sorted(d for d in os.listdir(root) if d not in C.skip_folders)) if x not in set(own)]
        rng.shuffle(more)
        got += scan(more, lang, pages, rng, seen, lines)
    Path(out).mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"w": lines, "pages": got, "wide": wide}, ensure_ascii=False))
    return lang, len(lines)


def build(layouts, out=BANK, pages=C.pages_per_lang, seed=0, workers=10, langs=None):
    langs = langs or sorted({l for ls in BUCKETS.values() for l in ls.split()} - C.skip_langs)
    with ProcessPoolExecutor(workers) as ex:
        return dict(ex.map(lang_bank, [(str(layouts), l, str(out), pages, seed) for l in langs], chunksize=1))
