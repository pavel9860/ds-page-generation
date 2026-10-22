"""Generates the pages of a plan shard: jpg + one meta line each (word boxes in px). Resumable, process pool.

python -m dspages.hand run --plan plan.jsonl --out OUT [--shard 0 --shards 4] [--workers 4] [--limit N]
"""
import argparse
import json
import os
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import cv2


KEEP = ("i", "lang", "type", "persona", "style", "pen", "paper", "kind", "doc", "page", "seed", "wseed", "fill", "ocr", "fmt", "dense", "table")


def work(job):
    plan, out = job
    from . import page
    f = Path(out) / "pages" / f"{plan['i']:05d}.jpg"
    if f.exists():
        return plan["i"], None
    t = time.time()
    try:
        img, meta = page.make(plan, plan["items"])
    except Exception:
        return plan["i"], "ERR " + traceback.format_exc()[-400:]
    cv2.imwrite(str(f), img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    keep = {k: plan.get(k) for k in KEEP}
    return plan["i"], json.dumps(dict(keep, **meta, sec=round(time.time() - t, 1)), ensure_ascii=False)


def main(argv=None):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cv2.setNumThreads(1)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    (Path(a.out) / "pages").mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(a.plan)]
    rows = [r for r in rows if r["i"] % a.shards == a.shard]
    rows = rows[: a.limit] if a.limit else rows
    todo = [r for r in rows if not (Path(a.out) / "pages" / f"{r['i']:05d}.jpg").exists()]
    print(len(rows), "pages in shard,", len(todo), "to do", flush=True)
    t0, n = time.time(), 0
    with Pool(a.workers, maxtasksperchild=20) as pool, open(Path(a.out) / f"meta_{a.shard}.jsonl", "a") as f, \
            open(Path(a.out) / f"errors_{a.shard}.log", "a") as fe:
        for i, res in pool.imap_unordered(work, [(r, a.out) for r in todo]):
            n += 1
            if res and res.startswith("ERR"):
                fe.write(f"{i} {res}\n")
                fe.flush()
            elif res:
                f.write(res + "\n")
                f.flush()
            if n % 20 == 0:
                print(f"{n}/{len(todo)}  {(time.time() - t0) / n:.1f}s/page", flush=True)


if __name__ == "__main__":
    main()
