"""Plan of the handwritten pages: every factor gets its exact share (largest remainder), factors are paired by independent shuffles (no factor
predicts another), passages are matched to the capacity of a page by length only.

python -m dspages.hand plan --texts texts.jsonl ... --out plan.jsonl --n 10000
"""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np

import hwsynth as H
from hwsynth.config import TOOLS
from hwsynth.scripts import text_script
from hwsynth.styles import personas

from . import paper as paper_mod
from .langs import EXPECT, NO_TEXT_LAYER, SKIP_SCRIPTS, group_of, implausible, script_ok
from .paper import paper
from .util import CHAR_W, MAX_XH, MIN_XH

ENGLISH = 0.40
GROUPS = {"cyrillic": 0.18, "cjk": 0.12, "other": 0.18, "latin": 0.12}     # shares of all pages (LayoutCfg.script_mix); English is the Latin rest
IAM_SHARE = 0.35                                       # pages written by a persona fitted to a real writer
FMT_W = {"a4": 0.55, "a5": 0.25, "b5": 0.20}           # sheet formats: A4 and notebook pages
DENSE_SHARE, TABLE_SHARE = 0.30, 0.05
PAPER_BY = {                                           # (notebook, mode) -> paper kinds and shares
    (False, "normal"): {"blank": 0.16, "lined": 0.20, "squared": 0.14, "dotted": 0.06, "legal": 0.06, "seyes": 0.08, "margin": 0.14, "spiral": 0.16},
    (False, "dense"): {"blank": 0.25, "dotted": 0.15, "squared": 0.15, "lined": 0.25, "margin": 0.10, "spiral": 0.10},
    (False, "table"): {"blank": 0.4, "dotted": 0.2, "squared": 0.4},
    (True, "normal"): {"lined": 0.30, "margin": 0.20, "spiral": 0.22, "squared": 0.12, "seyes": 0.10, "dotted": 0.06},
    (True, "dense"): {"lined": 0.35, "margin": 0.20, "spiral": 0.20, "squared": 0.15, "dotted": 0.10},
    (True, "table"): {"squared": 0.5, "dotted": 0.2, "lined": 0.3},
}
XH_RANGE = {(False, "normal"): (1.1, 3.0), (False, "dense"): (1.2, 1.8), (False, "table"): (1.2, 2.3),   # x-height [mm] on paper without rules
            (True, "normal"): (1.4, 3.4), (True, "dense"): (1.25, 2.1), (True, "table"): (1.4, 2.5)}
KIND_W = {"book": 0.22, "paper": 0.22, "article": 0.16, "letter": 0.14, "magazine": 0.10, "lecture": 0.15, "recipe": 0.005, "poem": 0.005}


def balanced(values, weights, n, rng):
    """n items, value i exactly round(n * share) times (largest remainder), shuffled."""
    w = np.asarray(weights, float) / np.sum(weights)
    c = np.floor(w * n).astype(int)
    for i in np.argsort(-(w * n - c))[: n - c.sum()]:
        c[i] += 1
    out = [v for v, k in zip(values, c) for _ in range(k)]
    rng.shuffle(out)
    return out


def languages(texts, n):
    """{language: pages}: English 40 %, the other shares by script group, equal over the languages of a group (capped by the passages a
    language has: water filling)."""
    langs = defaultdict(list)
    for i, t in enumerate(texts):
        if t["kind"] != "table":                       # a language needs text passages, tables alone are not enough
            langs[t["lang"]].append(i)
    group = {lg: Counter(t["script"] for t in (texts[i] for i in ix)).most_common(1)[0][0] for lg, ix in langs.items()}
    quota = {"en": int(round(ENGLISH * n))}
    for g, share in GROUPS.items():
        members = [lg for lg in langs if group_of(group[lg]) == g and lg != "en" and group[lg] not in SKIP_SCRIPTS]
        pool = {lg: len(langs[lg]) for lg in members}
        left = int(round(share * n))
        while left > 0 and members:
            each = max(left // len(members), 1)
            for lg in list(members):
                take = min(each, pool[lg] - quota.get(lg, 0), left)
                quota[lg] = quota.get(lg, 0) + take
                left -= take
            members = [lg for lg in members if quota[lg] < pool[lg] and left > 0]
    return quota


def load_texts(files):
    """Passages of the jsonl files that are real text of their language: script, no cross-document boilerplate, no table of numbers cut out
    of its table, no scrambled text layers."""
    texts, seen = [], set()
    for f in files:
        for line in open(f):
            t = json.loads(line)
            key = {" ".join(t["text"].split())[k:k + 60] for k in (0, 200, 500) if len(t["text"]) > k + 60}
            if key & seen or t["lang"] in NO_TEXT_LAYER or not script_ok(t["text"], t["lang"]):
                continue
            digits = sum(c.isdigit() for c in t["text"])
            if t["kind"] != "table" and digits > 0.3 * max(sum(c.isalnum() for c in t["text"]), 1):
                continue
            seen |= key
            t["script"] = text_script(t["text"])
            texts.append(t)
    return implausible(texts)


def draw_factors(n, quota, table_langs, rng):
    """n rows with the independent factors: type, style, pen, language, format, mode (normal / dense / table), writer, seeds."""
    types, styles, pens = list(H.PERSONS), list(H.STYLES), list(TOOLS)
    cols = {"type": balanced(types, np.ones(len(types)), n, rng), "style": balanced(styles, np.ones(len(styles)), n, rng),
            "pen": balanced(pens, np.ones(len(pens)), n, rng), "lang": balanced(list(quota), list(quota.values()), n, rng),
            "iam": balanced([1, 0], [IAM_SHARE, 1 - IAM_SHARE], n, rng), "fmt": balanced(list(FMT_W), list(FMT_W.values()), n, rng),
            "dense": balanced([1, 0], [DENSE_SHARE, 1 - DENSE_SHARE], n, rng), "tab": balanced([1, 0], [TABLE_SHARE, 1 - TABLE_SHARE], n, rng)}
    iam = defaultdict(list)
    for e in personas():
        iam[e["type"]].append(e["page"])
    seeds = rng.permutation(10 * n)[: 3 * n].reshape(3, n) + 1000
    rows = []
    for i in range(n):
        r = dict(i=i, **{k: v[i] for k, v in cols.items()})
        r["seed"], r["wseed"], r["pseed"] = (int(s[i]) for s in seeds)
        r["persona"] = f"iam:{iam[r['type']][int(rng.integers(len(iam[r['type']])))]}" if r.pop("iam") and iam[r["type"]] else r["type"]
        r["dense"] = bool(r["dense"]) and not r["tab"]
        r["table"] = bool(r.pop("tab")) and r["lang"] in table_langs
        rows.append(r)
    return rows


def set_geometry(rows, rng):
    """Paper, fill, x-height and the characters a page holds (_cap) of every row; papers by format and mode with exact shares."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["fmt"], "table" if r["table"] else "dense" if r["dense"] else "normal")].append(r)
    for (fmt, mode), rs in groups.items():
        wts = PAPER_BY[(fmt != "a4", mode)]
        for r, p in zip(rs, balanced(list(wts), list(wts.values()), len(rs), rng)):
            r["paper"] = p
    for r in rows:
        mode = "dense" if r["dense"] else "table" if r["table"] else "normal"
        lo, hi = XH_RANGE[(r["fmt"] != "a4", mode)]
        r["fill"] = float(rng.uniform(0.88, 1.0) if r["dense"] else rng.uniform(0.5, 0.9) if r["table"] else rng.uniform(0.6, 1.0))
        r["xh"] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        u = rng.random()
        r["xh_ratio"] = float(rng.uniform(0.16, 0.27) if r["dense"] else rng.uniform(0.22, 0.5) if u < 0.8 else rng.uniform(0.5, 0.62) if u < 0.9 else rng.uniform(0.18, 0.22))
        r["dy"] = float(rng.uniform(0.0, 0.5))
        pp = paper(r["paper"], np.random.default_rng(r["seed"]), r["fmt"], r["dense"])
        xh = float(np.clip(pp.pitch * r["xh_ratio"], MIN_XH, MAX_XH)) if pp.pitch else r["xh"]
        lines = max(int(round(((pp.size[1] - pp.bottom - pp.first) / (pp.pitch or 3.3 * xh) + 1) * r["fill"])), 1)
        r["_cap"] = 1.1 * lines * (pp.size[0] - pp.left - pp.right) / (xh * CHAR_W.get(EXPECT.get(r["lang"], "Latin"), 1.8))


def assign_texts(rows, texts, quota, rng):
    """A passage of the language to every row: the shortest one that fills the page (longest page first), of the source kind drawn for the
    row, from an unused document; short passages are completed by more of the same document."""
    kinds = Counter(t["kind"] for t in texts)
    kw = {k: v for k, v in KIND_W.items() if kinds[k]}
    for r, k in zip(rows, balanced(list(kw), list(kw.values()), len(rows), rng)):
        r["kind_t"] = "table" if r["table"] else "lecture" if r["dense"] and rng.random() < 0.75 and kinds["lecture"] else k
    by = defaultdict(list)
    for i, t in enumerate(texts):
        by[t["lang"]].append(i)
    used = set()
    for lg in quota:
        full = sorted(by[lg], key=lambda i: len(texts[i]["text"]))
        prose = lambda: [i for i in full if texts[i]["kind"] != "table"]
        pool, tabs = prose(), [i for i in full if texts[i]["kind"] == "table"]
        for r in sorted((r for r in rows if r["lang"] == lg), key=lambda r: -r["_cap"]):
            if r["table"]:
                t = texts[tabs[int(rng.integers(len(tabs)))]]
                r.update(items=list(t["items"]), doc=t["doc"], kind="table", page=t["page"], ocr=False)
            else:
                pool = pool or prose()                    # small pools are used up by long pages: passages are used again

                def pick(ok):
                    ix = [j for j, i in enumerate(pool) if ok(texts[i])]
                    return next((j for j in ix if len(texts[pool[j]]["text"]) >= 0.9 * r["_cap"]), ix[-1] if ix else None)
                j = next(j for j in (pick(lambda t: t["kind"] == r["kind_t"] and t["doc"] not in used), pick(lambda t: t["doc"] not in used),
                                     pick(lambda t: t["kind"] == r["kind_t"]), pick(lambda t: True)) if j is not None)
                t = texts[pool.pop(j)]
                used.add(t["doc"])
                items, extra = list(t["items"]), 0
                while sum(len(i["t"]) for i in items) < 0.85 * r["_cap"] and pool and extra < (8 if r["dense"] else 5):
                    same = [k for k, i in enumerate(pool) if texts[i]["doc"] == t["doc"]]
                    items += texts[pool.pop(same[0] if same else len(pool) - 1)]["items"]
                    extra += 1
                r.update(items=items, doc=t["doc"], kind=t["kind"], page=t["page"], ocr=bool(t.get("ocr")))
            del r["_cap"], r["kind_t"]


def make_plan(texts, n, seed=1):
    """The plan: n page specs with their passages (list of dict)."""
    rng = np.random.default_rng(seed)
    quota = languages(texts, n)
    rows = draw_factors(n, quota, {t["lang"] for t in texts if t["kind"] == "table"}, rng)
    set_geometry(rows, rng)
    assign_texts(rows, texts, quota, rng)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--texts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    paper_mod.DRAW = False                         # planning needs the geometry of the papers, not their pictures
    texts = load_texts(a.texts)
    rows = make_plan(texts, a.n, a.seed)
    with open(a.out, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"{a.n} pages from {len(texts)} passages;", Counter(r["lang"] for r in rows).most_common(8))
    print(Counter(r["kind"] for r in rows))


if __name__ == "__main__":
    main()
