"""What the manifest holds against a plan's quotas: pages per script group and language, and what is short.

python tools/inventory.py [--preset full] [--n 100000] [--manifest PATH]
Every sample uses its own page (no repeats): a group covers its quota n * weight with raster pages first, then
book pages; the rest is the shortfall.
"""
import argparse
import json
from collections import Counter, defaultdict
from dataclasses import replace

from dspages.conditions import entry_pools, script
from dspages.config import get_preset
from dspages.layout.sources import is_pdf, load_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="full")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--manifest", default=None)
    a = ap.parse_args()
    P = get_preset(a.preset)
    if a.manifest:
        P = replace(P, paths=replace(P.paths, manifest=a.manifest))
    man = load_manifest(P.paths.manifest)
    pools = entry_pools(P, man)
    eligible = {i for raster, _ in pools.values() for i in raster}

    langs = defaultdict(Counter)
    for i, e in enumerate(man):
        kind = "book" if e["kind"] == "book_text" else ("pdf" if is_pdf(e) else "image")
        langs[(script(e, P.layout.scripts), e["language"])][kind + ("" if kind == "book" or i in eligible
                                                                    else "_too_small")] += 1

    rows, short = [], {}
    print(f"{'group':10s} {'quota':>8s} {'raster':>8s} {'books':>8s} {'covered':>8s} {'short':>8s}  raster share")
    for g, w in P.layout.script_mix:
        raster, books = pools.get(g, ([], []))
        q = round(a.n * w)
        cov = min(q, len(raster) + len(books))
        short[g] = q - cov
        rows.append(dict(group=g, quota=q, raster=len(raster), books=len(books), covered=cov, short=q - cov))
        print(f"{g:10s} {q:8d} {len(raster):8d} {len(books):8d} {cov:8d} {q - cov:8d}  "
              f"{min(len(raster), q) / max(q, 1):.0%}")
    print(f"\ntotal: {sum(r['covered'] for r in rows)} of {a.n} samples covered without repeats; "
          f"short {sum(short.values())}")
    print("\npages per language (script group, language: kinds)")
    for (g, lang), c in sorted(langs.items(), key=lambda kv: (kv[0][0], -sum(kv[1].values()))):
        print(f"  {g:9s} {str(lang):8s} " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))
    print(json.dumps(dict(n=a.n, groups=rows), indent=1))


if __name__ == "__main__":
    main()
