"""Builds the line banks from the corpus (dspages.multilang.linebuild)"""

import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
from dspages.multilang import linebuild
from dspages.multilang.config import LINEBANK
from dspages.multilang.paths import LAYOUTS

if __name__ == "__main__":
    pages = int(sys.argv[1]) if len(sys.argv) > 1 else LINEBANK.pages_per_lang
    t = time.time()
    res = linebuild.build(LAYOUTS, pages=pages, langs=sys.argv[2:] or None)
    n = sorted(res.values())
    print(
        f"{len(res)} languages in {time.time() - t:.0f}s; lines min {n[0]}, median {n[len(n) // 2]}, max {n[-1]}; under {LINEBANK.min_lines}: "
        f"{sorted(l for l, k in res.items() if k < LINEBANK.min_lines)}"
    )
