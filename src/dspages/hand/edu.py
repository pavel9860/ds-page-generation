"""Long passages (2500-5500 characters, consecutive blocks over a few pages) of education books, textbooks, lectures and academic
documents of many topics, for dense full pages (diary, lecture notes). Topics from the manifest (old_meta.topic), digital text layer,
language confident, passages checked for the script of the language. kind = lecture.

python -m dspages.hand edu --out texts_edu.jsonl [--workers 6] [--per-lang 150]
"""
import argparse
import json
import os
import random
from collections import defaultdict
from multiprocessing import Pool

import pymupdf

from . import corpus
from . import texts as T
from .langs import EXPECT, script_ok
from .structure import to_items
from .util import flat_items, walk_text

EDU = {"education", "textbook", "academic", "mathematics", "history", "physics_and_astronomy", "psychology", "social_sciences",
       "social-sciences", "arts_and_humanities", "arts-and-humanities", "computer_science", "computer-science", "medicine",
       "engineering", "earth_and_planetary_sciences", "agricultural_and_biological_sciences", "environment", "reference",
       "economics_econometrics_and_finance", "culture", "healthcare", "technology", "manual", "agriculture", "law"}


def candidates(per_lang):
    db = corpus.connect()
    lang = corpus.languages(db)
    out = defaultdict(list)
    for d, topic, pages, path in db.execute("select m.hash,m.topic,p.pages,p.new_path from old_meta m join plan p on p.doc=m.hash "
                                              "where m.digital=1 and p.cls='text'"):
        if topic in EDU and d in lang and (pages or 0) >= 4 and path.lower().endswith(".pdf"):
            out[lang[d]].append(dict(doc=d, lang=lang[d], topic=topic, kind="lecture", path=f"{T.LAYOUTS}/{path}", pages=pages))
    rng = random.Random(0)
    res = {}
    for lg, v in out.items():
        rng.shuffle(v)
        by = defaultdict(list)
        for c in v:
            by[c["topic"]].append(c)
        order = []                                         # topics in turn: the topics are equally represented
        while any(by.values()) and len(order) < per_lang:
            for k in list(by):
                if by[k]:
                    order.append(by[k].pop())
        res[lg] = order
    return res


def long_passage(job):
    c, seed = job
    rng = random.Random(seed)
    native = c["lang"] in EXPECT
    target = rng.choice((2500, 3200, 4000, 5000, 5500))
    try:
        pdf = pymupdf.open(c["path"])
        n = len(pdf)
        lo = int(0.1 * n) if n >= 20 else 0
        hi = max(n - int(0.08 * n) - 3, lo + 1) if n >= 20 else max(n - 2, 1)
        pg = rng.randrange(lo, hi)
        items, size = [], 0
        for p in range(pg, min(pg + 6, n)):
            bl = T.blocks_of(pdf[p])
            if not items:
                bl = bl[next((i for i, b in enumerate(bl) if b[0] in ("h", "p")), len(bl)):]
            page_items = to_items(bl)
            if page_items and items:
                page_items[0]["gap"] = 0                   # the next page goes on without a break
            items += page_items
            size += sum(len(b[1]) for b in bl)
            if size >= target:
                break
        while items and items[-1]["k"] == "h":
            items.pop()
        if size < 1800 or sum(i["k"] == "p" for i in flat_items(items)) < 3:
            return None
        text = "\n".join(walk_text(items))
        if native and not script_ok(text, c["lang"], 0.85):
            return None
        return dict(c, items=items, text=text, page=pg, pages=n, path=os.path.relpath(c["path"], T.LAYOUTS))
    except Exception as e:
        print("passage failed:", c["path"], repr(e)[:120], flush=True)
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-lang", type=int, default=150)
    ap.add_argument("--en", type=int, default=2200)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    cand = candidates(a.en)
    rng = random.Random(1)
    jobs = []
    for lg, cs in cand.items():
        cap = a.en if lg == "en" else a.per_lang
        jobs += [(c, rng.getrandbits(32)) for c in cs[:cap]]
    rng.shuffle(jobs)
    print(len(jobs), "documents,", {k: len(v) for k, v in sorted(cand.items(), key=lambda kv: -len(kv[1]))[:20]}, flush=True)
    n = 0
    with Pool(a.workers, maxtasksperchild=100) as pool, open(a.out, "w") as f:
        for i, r in enumerate(pool.imap_unordered(long_passage, jobs, chunksize=4)):
            if r:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
            if i % 500 == 0:
                print(i, "/", len(jobs), n, "passages", flush=True)
    print(n, "passages")


if __name__ == "__main__":
    main()
