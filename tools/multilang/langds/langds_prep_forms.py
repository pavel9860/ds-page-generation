import os

os.environ["OMP_NUM_THREADS"] = "1"
import collections
import glob
import json
import random
import sys

from dspages.multilang import filledforms as F

if __name__ == "__main__":
    out = sys.argv[1]
    by = collections.defaultdict(list)
    for l in open(out + "/real.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["type"] in ("form", "filled") and r["src"].lower().endswith(".pdf"):
            by[r["lang"]].append((r["src"], r["page"], r["lang"]))
    bg = []
    for l, v in sorted(by.items()):
        random.Random(l).shuffle(v)
        bg += v[:6]
    for f in sorted(glob.glob(out + "/forms_blank/*.pdf")):
        bg.append((f, 0, os.path.basename(f).rsplit("_", 1)[0]))
    print(len(bg), "backgrounds", flush=True)
    print(F.prepare(bg, out, workers=5), "forms prepared", flush=True)
