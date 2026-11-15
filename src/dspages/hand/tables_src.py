"""Tables of digital-born PDFs (pages the embedding classifier calls tables, p >= 0.7) as rows of cell texts, for handwritten tables:
PyMuPDF find_tables, small tables only (2-7 columns, 3-14 rows, short cells), most cells filled, the script of the language.

python -m dspages.hand tables --out texts_tables.jsonl [--workers 6] [--per-lang 80]
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict
from multiprocessing import Pool

import pymupdf

from . import corpus
from . import texts as T
from .langs import EXPECT, script_ok


def candidates(per_lang, min_p=0.7):
    """Documents whose page type is `tables` (p >= min_p), with a digital text layer and a confident language."""
    db = corpus.connect()
    typ = {d for d, (lab, p) in corpus.page_types(min_p).items() if lab == "tables"}
    lang = corpus.languages(db)
    out = defaultdict(list)
    for d, path, pages in corpus.pdf_documents(db):
        if d in typ and d in lang:
            out[lang[d]].append(dict(doc=d, lang=lang[d], kind="table", path=str(path), pages=pages))
    rng = random.Random(0)
    for v in out.values():
        rng.shuffle(v)
    return {k: v[:per_lang] for k, v in out.items()}


def tidy(c):
    c = re.sub(r"\s+", " ", (c or "").replace("\n", " ")).strip()
    return c


def read(job):
    c, seed = job
    rng = random.Random(seed)
    res = []
    try:
        pdf = pymupdf.open(c["path"])
        n = len(pdf)
        for pg in rng.sample(range(n), min(n, 3)):
            for tb in pdf[pg].find_tables().tables:
                rows = [[tidy(x) for x in r] for r in tb.extract()]
                rows = [r for r in rows if any(r)]
                if not (3 <= len(rows) <= 14):
                    continue
                ncol = max(len(r) for r in rows)
                rows = [r + [""] * (ncol - len(r)) for r in rows]
                keep = [j for j in range(ncol) if sum(bool(r[j]) for r in rows) >= 0.5 * len(rows)]  # no empty columns
                rows = [[r[j] for j in keep] for r in rows]
                if not (2 <= len(keep) <= 7):
                    continue
                cells = [x for r in rows for x in r]
                if max(len(x) for x in cells) > 40 or sum(bool(x) for x in cells) < 0.7 * len(cells):
                    continue
                text = " ".join(cells)
                if len(re.findall(r"\w", text)) < 20 or (c["lang"] in EXPECT and not script_ok(text, c["lang"], 0.6)):
                    continue
                res.append(
                    dict(
                        c,
                        items=[dict(k="table", rows=rows)],
                        text=text,
                        page=pg,
                        pages=n,
                        path=os.path.relpath(c["path"], T.LAYOUTS),
                    )
                )
                break
    except Exception as e:
        print("table source failed:", c["path"], repr(e)[:120], flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-lang", type=int, default=80)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    cand = candidates(a.per_lang)
    rng = random.Random(1)
    jobs = [(c, rng.getrandbits(32)) for v in cand.values() for c in v]
    rng.shuffle(jobs)
    print(len(jobs), "documents", flush=True)
    n = 0
    with Pool(a.workers, maxtasksperchild=100) as pool, open(a.out, "w") as f:
        for i, rs in enumerate(pool.imap_unordered(read, jobs, chunksize=4)):
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
            if i % 300 == 0:
                print(i, "/", len(jobs), n, "tables", flush=True)
    print(n, "tables")


if __name__ == "__main__":
    main()
