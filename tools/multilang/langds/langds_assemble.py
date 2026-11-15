import os

os.environ["OMP_NUM_THREADS"] = "1"
import collections
import json
import shutil
import sys
from pathlib import Path

from dspages.multilang import langds as G

D = Path(sys.argv[1])
out = D / "langds"
per = int(sys.argv[2])
rows = json.load(open(out / "chosen_real.json", encoding="utf-8"))
for name in ("synth.jsonl", "filled.jsonl"):
    f = out / name
    if f.exists():
        rows += [json.loads(l) for l in open(f, encoding="utf-8")]
rows = [r for r in rows if (out / r["file"]).exists()]
cnt = G.counts(rows)
print("classes", len(cnt), "items", len(rows))
sizes = sorted(sum(v.values()) for v in cnt.values())
print("items per class: min", sizes[0], "median", sizes[len(sizes) // 2], "max", sizes[-1])
rows = G.balance(rows, per)
rep = G.report(rows)
print(
    "after balance:",
    rep["items"],
    "items;",
    rep["n_short"],
    "language-type cells below",
    G.MIN_PER_TYPE,
    rep["short"][:25],
)
cnt = G.counts(rows)
print("per type:", dict(collections.Counter(r["type"] for r in rows)))
print(
    "real/synthetic:",
    dict(collections.Counter("synthetic" if r.get("lang_status") == "synthetic" else "real" for r in rows)),
)
ds = D / "langds_ds"
if ds.exists():
    shutil.rmtree(ds)
ds.mkdir()
print(G.assemble(out, rows, ds))
