"""Blind labelling of border-case candidates (tools/pagekind_border.py pick) by agents, with the montages and prompt of the corpus page-type
recheck (corpus_collecting pagetype.recheck: numbered 2 x 2 montages, 24 pages per agent, the class descriptions of the labelled set).

  batches  DIR/img/*.jpg -> DIR/agent/batch_NNN/{m1..m6.jpg, in.json, prompt.txt}
  collect  DIR/agent/batch_*/out.json -> DIR/labels.json {k: class | "skip"} (labels outside plain / poems / formatted / tables /
           forms_blank / forms_filled, and "other", are "skip"); then: python tools/pagekind_border.py label --out DIR

python tools/pagekind_agents.py batches|collect --out DIR
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from dspages import paths as _P

sys.path.insert(0, f"{_P.CORPUS}")
from pagetype import recheck as RC  # noqa: E402

SIX = (
    "plain",
    "poems",
    "formatted",
    "tables",
    "forms_blank",
    "forms_filled",
    "bureaucracy",
    "certificates",
    "flyers",
    "magazines",
    "scientific",
    "maps",
    "blueprints",
    "music",
)  # labels kept from the agents; anything else is "skip"


def prompt(d):
    """The recheck prompt, with a confidence per page: out.json {"<page id>": ["<label>", <confidence 1-5>]}."""
    return RC.prompt(d).replace(
        'Write out.json in the same folder: {"<page id>": "<label>"} for every id in in.json.',
        'Write out.json in the same folder: {"<page id>": ["<label>", <confidence>]} for every id in in.json, '
        "confidence an integer 1-5 (5: certain, 3: could be another class, 1: guess).",
    )


def batches(out):
    c = [json.loads(l) for l in open(out / "candidates.jsonl")]
    items = [(str(r["k"]), out / "img" / f"{r['k']:04d}.jpg") for r in c]
    dirs = RC.make_batches(items, work=out / "agent")
    for d in dirs:
        (d / "prompt.txt").write_text(prompt(d))
    print(len(dirs), "batches in", out / "agent")


def collect(out, review=0.10):
    """labels.json from the agents; the least confident `review` share of the pages -> review.json (ids, lowest confidence first) and
    review sheets DIR/review_NN.jpg for a second look; corrections in DIR/review_labels.json {k: class | "skip"} override the agents."""
    lab, conf, n = {}, {}, Counter()
    for f in sorted((out / "agent").glob("batch_*/out.json")):
        for k, v in json.loads(f.read_text()).items():
            v, c = (v, 3) if isinstance(v, str) else (v[0], int(v[1]))
            lab[k], conf[k] = (v if v in SIX else "skip"), c
            n[v] += 1
    low = sorted(conf, key=lambda k: (conf[k], int(k)))[: int(review * len(conf))]
    json.dump(low, open(out / "review.json", "w"))
    for s in range(0, len(low), 10):
        RC.montage(
            [out / "img" / f"{int(k):04d}.jpg" for k in low[s : s + 10]],
            out / f"review_{s // 10:02d}.jpg",
            cell=400,
            cols=5,
        )
    fix = out / "review_labels.json"
    if fix.exists():
        lab.update(json.load(open(fix)))
    json.dump(lab, open(out / "labels.json", "w"), indent=0)
    print(len(lab), "labels:", dict(n), "| confidence:", dict(Counter(conf.values())), "| to review:", len(low))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("batches", "collect"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"batches": batches, "collect": collect}[a.cmd](Path(a.out))
