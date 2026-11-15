"""Accuracy of the rule-based page detector (dspages.hand.pagekind: fast line features + reader features of the text layer, no image model)
on the labelled pages matched to their PDFs: grouped cross-validation, forms_blank / forms_filled against the other kinds.
python tools/detector_rule_check.py [--classes ...] [--seeds 3]"""

import argparse

import numpy as np
import pagekind_eval as PE

ap = argparse.ArgumentParser()
ap.add_argument("--classes", default="forms_blank,forms_filled,formatted,bureaucracy,tables,plain")
ap.add_argument("--seeds", type=int, default=3)
a = ap.parse_args()
classes = a.classes.split(",")
X, Y, E, D, y, g, recs, t = PE.load(PE.SRC, classes, False, False)
print(
    f"{len(y)} pages, {len(set(g))} documents, {dict(zip(classes, np.bincount(y)))}; fast features {1000 * np.median(t[:, 0]):.0f} ms median, "
    f"reader {1000 * np.median(t[:, 1]):.0f} ms median",
    flush=True,
)
XY = np.hstack([X, Y])
for name, (make, Z) in {
    "fast format, logreg": (lambda: PE.Model("lr"), X),
    "fast + reader format, xgb": (lambda: PE.Model("xgb"), XY),
}.items():
    P = [PE.cv_proba(make, Z, y, g, s) for s in range(a.seeds)]
    PE.report(name, P, y, classes)
    PE.standard(name, P, y, classes)
