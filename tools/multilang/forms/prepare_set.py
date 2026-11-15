"""Prepares a set of forms with the current field finder for the agents"""
import json
import random
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from dspages.multilang import filledforms as F
from dspages.multilang.buckets import NAMES
from dspages.multilang.config import Config
from dspages.multilang.paths import FILLING
from dspages.pagegen import forms

HERE = Path(__file__).parent


def prep(job):
    path, page, lang, out = job
    r = F.prep_one((path, page, lang, out, 0))
    if r is None:
        return None
    item, meta = r
    spots = forms.render_form(path, int(page), replace(Config(), long_side=1754)).spots
    item["language"] = NAMES.get(lang, lang)
    item["boxes"] = [dict(b, label=spots[b["i"]]["label"]) for b in item["boxes"]]
    return item, meta


def main():
    out, n = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 100
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((FILLING / "meta.json").read_text())
    ok = json.loads((FILLING / "usable.json").read_text())
    by = defaultdict(list)
    for fid, m in meta.items():
        if ok.get(fid) == "ok":
            by[m["lang"]].append(m)
    rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    for v in by.values():
        rng.shuffle(v)
    jobs = []
    while len(jobs) < 2 * n and any(by.values()):
        for lang in sorted(by):
            if by[lang]:
                m = by[lang].pop()
                jobs.append((m["path"], int(m["page"]), lang, str(out)))
    res, seen = [], set()
    with ProcessPoolExecutor(6) as ex:
        for r in ex.map(prep, jobs, chunksize=2):
            if r and r[1]["fid"] not in seen and len(r[0]["boxes"]) >= 4 and len(res) < n:
                seen.add(r[1]["fid"])
                res.append(r)
    (out / "meta.json").write_text(json.dumps({m["fid"]: m for _, m in res}))
    for k in range(0, len(res), 4):
        (out / f"in_{k // 4:04d}.json").write_text(json.dumps([i for i, _ in res[k:k + 4]], ensure_ascii=False))
    shutil.copy(HERE / "INSTRUCTIONS.md", out / "INSTRUCTIONS.md")
    print(len(res), "forms,", len({m['lang'] for _, m in res}), "languages,", -(-len(res) // 4), "batches ->", out)


if __name__ == "__main__":
    main()
