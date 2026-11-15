"""More forms from the corpus for the languages short of forms
python scripts/forms/add_corpus_forms.py LANG=N [LANG=N ...]"""
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

import pymupdf
from dspages.multilang import filledforms as F
from dspages.multilang.detlayouts import near_a4
from dspages.multilang.langpages import detect_type
from dspages.multilang.paths import FILLING, LAYOUTS
from dspages.multilang.realpages import parse_lang

FOLDERS = ("forms", "forms_clean", "bureaucracy")


def pages(lang, have):
    out = []
    for d in FOLDERS:
        for f in sorted(os.listdir(LAYOUTS / d)) if (LAYOUTS / d).is_dir() else []:
            if f.lower().endswith(".pdf") and parse_lang(f) == lang:
                try:
                    with pymupdf.open(LAYOUTS / d / f) as doc:
                        out += [(str(LAYOUTS / d / f), i, lang) for i in range(min(len(doc), 3))
                                if near_a4(doc[i].rect.width, doc[i].rect.height) and detect_type(doc[i]) == "forms" and (str(LAYOUTS / d / f), i) not in have]
                except Exception:
                    pass
    return out


def prep(job):
    path, page, lang = job
    r = F.prep_one((path, page, lang, str(FILLING), 0))
    return r


def main():
    need = {k: int(v) for k, v in (a.split("=") for a in sys.argv[1:])}
    meta = json.loads((FILLING / "meta.json").read_text())
    have = {(m["path"], int(m["page"])) for m in meta.values()}
    items = []
    with ProcessPoolExecutor(6) as ex:
        for lang, n in need.items():
            cand = pages(lang, have)
            random.Random(lang).shuffle(cand)
            got = [r for r in ex.map(prep, cand[:3 * n]) if r][:n]
            for item, m in got:
                meta[m["fid"]] = m
                items.append(item)
            print(lang, "candidates", len(cand), "prepared", len(got), "of", n)
    first = len(list(FILLING.glob("in_*.json")))
    for k in range(0, len(items), 4):
        (FILLING / f"in_{first + k // 4:04d}.json").write_text(json.dumps(items[k:k + 4], ensure_ascii=False))
    (FILLING / "meta.json").write_text(json.dumps(meta))
    print(len(items), "forms added")


if __name__ == "__main__":
    main()
