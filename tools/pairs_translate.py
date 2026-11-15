"""The translations of the pairs layouts only (shift_test's translate_forms and translate_texts, its stages before the images), Google through one
browser (tools/seed_texts.py), into the build's PAIRS_DIR:
  PAIRS_DIR=<build dir> python tools/pairs_translate.py PLAN.csv [--workers 6]"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "tools")
import shift_test as ST

if __name__ == "__main__":
    w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 6
    L = pd.read_csv(sys.argv[1], index_col=0)  # the plan (its pages counted here, not written back)
    ST.translate_forms(L, 1)
    if "pages" not in L:
        L["pages"] = np.nan
    T = L[L.kind == "text"]
    for _, (i, n) in ST.pooled(ST._pages, {i: (i, r.to_dict()) for i, r in T.iterrows() if pd.isna(r.pages)}, w):
        L.loc[i, "pages"] = n
    T = L[L.kind == "text"]
    ST.translate_texts(
        [(i, r.path, int(r.page) + k, r.lang, r.tl) for i, r in T.iterrows() for k in range(int(r.pages))], 1
    )
    print("translations done", flush=True)
