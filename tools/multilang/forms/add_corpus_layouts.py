"""More layouts from the corpus for the languages short of them
python scripts/forms/add_corpus_layouts.py LANG=N [LANG=N ...]"""
import hashlib
import json
import os
import random
import sys

import pymupdf
from dspages.multilang.detlayouts import near_a4
from dspages.multilang.paths import LANGDS, LAYOUTS
from dspages.multilang.realpages import parse_lang, text_layer_ok

SKIP = ("handwritten", "filled_forms", "sheet_music", "maps", "maps_blueprints", "blueprints", "images")


def pages(lang, used):
    out = []
    for d in sorted(os.listdir(LAYOUTS)):
        if (LAYOUTS / d).is_dir() and d not in SKIP:
            for f in sorted(os.listdir(LAYOUTS / d)):
                if f.lower().endswith(".pdf") and parse_lang(f) == lang:
                    try:
                        with pymupdf.open(LAYOUTS / d / f) as doc:
                            for i in range(min(len(doc), 3)):
                                pg = doc[i]
                                if (str(LAYOUTS / d / f), i) not in used and near_a4(pg.rect.width, pg.rect.height) and len(pg.get_text("words")) > 80 and text_layer_ok(pg):
                                    out.append((str(LAYOUTS / d / f), i))
                    except Exception:
                        pass
    return out


def main():
    need = {k: int(v) for k, v in (a.split("=") for a in sys.argv[1:])}
    used = {(r["src"], r["page"]) for r in map(json.loads, open(LANGDS / "det_layouts/layouts.jsonl"))}
    f = LANGDS / "det_extra_corpus.jsonl"
    rows = [json.loads(l) for l in open(f)] if f.exists() else []
    have = {r["id"] for r in rows}
    for lang, n in need.items():
        cand = pages(lang, used | {(r["src"], r["page"]) for r in rows})
        random.Random(lang).shuffle(cand)
        for p, i in cand[:n]:
            rid = hashlib.sha1(f"{p}{i}".encode()).hexdigest()[:12]
            if rid not in have:
                rows.append(dict(id=rid, lang=lang, type="text", src=p, page=i, scan=False, lang_status="ok", script=None, boxes=[{"k": "print", "b": [0, 0, 1, 1]}]))
        print(lang, "found", len(cand), "pages, listed", min(n, len(cand)))
    f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


if __name__ == "__main__":
    main()
