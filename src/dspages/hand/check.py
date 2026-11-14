"""Plan balance and leakage check: shares of every factor, Cramer's V between factors (0 = independent), repeated texts.

python -m dspages.hand check plan.jsonl
"""
import argparse
import itertools
import json
from collections import Counter

import numpy as np

from .util import walk_text


def cramers_v(a, b):
    """Bias-corrected Cramer's V of two categorical sequences."""
    ca, cb = sorted(set(a), key=str), sorted(set(b), key=str)
    t = np.zeros((len(ca), len(cb)))
    for x, y in zip(a, b):
        t[ca.index(x), cb.index(y)] += 1
    e = t.sum(1)[:, None] * t.sum(0)[None] / t.sum()
    chi = ((t - e) ** 2 / np.maximum(e, 1e-9)).sum()
    bias = (len(ca) - 1) * (len(cb) - 1) / (len(a) - 1)
    return float(np.sqrt(max(chi / len(a) - bias, 0) / max(min(len(ca), len(cb)) - 1, 1)))


def factors(rows):
    band = lambda x, edges: int(np.searchsorted(edges, x))
    f = {k: [r[k] for r in rows] for k in ("lang", "type", "style", "pen", "paper", "kind", "fmt")}
    f["iam"] = [r["persona"].startswith("iam") for r in rows]
    f["mode"] = [("table" if r.get("table") else "dense" if r.get("dense") else "normal") for r in rows]
    f["fill"] = [band(r["fill"], [.3, .6, .85]) for r in rows]
    f["len"] = [band(len(" ".join(walk_text(r["items"]))), [300, 600, 1000, 1500]) for r in rows]
    return f


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("plan")
    rows = [json.loads(l) for l in open(ap.parse_args(argv).plan)]
    f = factors(rows)
    print(len(rows), "pages")
    for k in ("type", "style", "pen", "paper", "kind", "fmt", "mode"):
        c = Counter(f[k])
        print(f"{k:6} min {min(c.values())} max {max(c.values())}", dict(c.most_common(3)))
    print("languages:", len(set(f["lang"])), Counter(f["lang"]).most_common(6))
    top = sorted(((cramers_v(f[a], f[b]), a, b) for a, b in itertools.combinations(f, 2)), reverse=True)[:8]
    print("largest Cramer V:", [(round(v, 3), a, b) for v, a, b in top])
    first = Counter(" ".join(next(walk_text(r["items"]), "").split())[:60] for r in rows)
    print("repeated first lines:", sum(c > 1 for c in first.values()), "| documents used more than once:", sum(c - 1 for c in Counter(r["doc"] for r in rows).values()))


if __name__ == "__main__":
    main()
