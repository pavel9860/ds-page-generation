import os

os.environ["OMP_NUM_THREADS"] = "1"
import collections
import json
import sys
from pathlib import Path

from dspages.multilang import langds as G
from dspages.multilang import synthpages as P

if __name__ == "__main__":
    D = Path(sys.argv[1])
    n = int(sys.argv[2])
    real = [json.loads(l) for l in open(D / "langds" / "real.jsonl", encoding="utf-8")]
    quota, chosen = G.plan(real, n)
    langs = sorted({l for l, _ in quota})
    pools = {
        (l, part): P.load_pools(l, D / "langds", D / "bucketing", part) for l in langs for part in ("train", "val")
    }
    jobs = G.synth_jobs(quota, chosen, D / "langds", pools)
    print(len(chosen), "real chosen;", len(jobs), "synthetic jobs;", len(set(langs) - set(pools)), "x", flush=True)
    json.dump([r for r in chosen], open(D / "langds" / "chosen_real.json", "w"), ensure_ascii=False)
    rows = G.run_synth(jobs)
    with open(D / "langds" / "synth.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        len(rows), "synthetic pages made of", len(jobs), dict(collections.Counter(r["type"] for r in rows)), flush=True
    )
