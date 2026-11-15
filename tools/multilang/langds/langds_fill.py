import os

os.environ["OMP_NUM_THREADS"] = "1"
import collections
import json
import math
import sys
from pathlib import Path

from dspages.multilang import filledforms as F
from dspages.multilang import langds as G

if __name__ == "__main__":
    D = Path(sys.argv[1])
    out = D / "langds"
    real = [json.loads(l) for l in open(out / "real.jsonl", encoding="utf-8")]
    quota, _ = G.plan(real, int(sys.argv[2]))
    work = out / "forms_work"
    meta = json.loads((work / "meta.json").read_text())
    have = collections.Counter()
    vals = {}
    for f in sorted(work.glob("out_*.json")):
        vals.update(json.loads(f.read_text()))
    for fid, m in meta.items():
        if fid in vals:
            have[m["lang"]] += 1
    variants = {l: max(1, math.ceil(quota.get((l, "filled"), 3) / n)) for l, n in have.items()}
    print(
        sum(have.values()),
        "forms with answers in",
        len(have),
        "languages; variants up to",
        max(variants.values()),
        flush=True,
    )
    rows = F.fill_all(str(out), variants)
    with open(out / "filled.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = collections.Counter(r["lang"] for r in rows)
    print(
        len(rows),
        "filled pages;",
        len(c),
        "languages; languages below 3:",
        [l for l, n in c.items() if n < 3],
        flush=True,
    )
