"""Plan of the handwritten pages: every factor gets its exact share (largest remainder), factors are paired by independent shuffles (no factor
predicts another), passages are matched to the capacity of a page by length only.

python -m dspages.hand plan --texts texts.jsonl ... --out plan.jsonl --n 10000
"""

import argparse
import json
import re
from collections import Counter, defaultdict

import hwsynth as H
import numpy as np
from hwsynth.config import TOOLS
from hwsynth.scripts import text_script
from hwsynth.styles import personas

from ..ingest.lang import garbled
from . import paper as paper_mod
from .config import ENGLISH, FMT_W, GROUPS, IAM_SHARE, KIND_W, PAPER_BY, XH_RANGE  # noqa: F401
from .langs import (
    EXPECT,
    ILLEGIBLE_TYPES,
    LEGIBLE_STYLES,
    NO_TEXT_LAYER,
    PRINT_HANDS,
    PRINT_ONLY_NOT,
    SKIP_SCRIPTS,
    group_of,
    script_ok,
)
from .paper import paper
from .util import CHAR_W, MAX_XH, MIN_XH, MIN_XH_SCRIPT, walk_text

DENSE_SHARE, TABLE_SHARE = 0.30, 0.05


def fitted_tune(rng, spread=0.12):
    """Writer genes fitted to the distributions of real lines (fitted_tune.json: component sizes, joins, word gaps), each moved a little per page."""
    import json
    from pathlib import Path

    base = json.loads((Path(__file__).parent / "fitted_tune.json").read_text())["tune"]
    wide = dict(
        ws=(0.8, 1.5),
        ws_ln=(0.05, 0.4),
        asc=(0.75, 2.2),
        desc=(0.75, 2.2),
        cap=(0.9, 1.8),
        lv_s=(0, 0.35),
        lv_dy=(0, 0.2),
        lv_break=(0, 0.3),
        lv_warp=(0, 0.2),
        ls_sd=(0, 0.1),
        lv_gap=(0, 0.4),
        size_sd=(0, 0.1),
        connect=(0, 1),
        speed=(0, 1),
        slant_sd=(0, 4),
        base_sd=(0, 0.09),
    )
    return {
        k: float(min(max(v + rng.normal(0, spread * (wide[k][1] - wide[k][0])), wide[k][0]), wide[k][1]))
        for k, v in base.items()
        if k in wide
    }


def balanced(values, weights, n, rng):
    """n items, value i exactly round(n * share) times (largest remainder), shuffled."""
    w = np.asarray(weights, float) / np.sum(weights)
    c = np.floor(w * n).astype(int)
    for i in np.argsort(-(w * n - c))[: n - c.sum()]:
        c[i] += 1
    out = [v for v, k in zip(values, c) for _ in range(k)]
    rng.shuffle(out)
    return out


def languages(texts, n, as_is=False):
    """{language: pages}: English 40 %, the other shares by script group, equal over the languages of a group (capped by the passages a
    language has: water filling)."""
    langs = defaultdict(list)
    for i, t in enumerate(texts):
        if t["kind"] != "table":  # a language needs text passages, tables alone are not enough
            langs[t["lang"]].append(i)
    group = {lg: Counter(t["script"] for t in (texts[i] for i in ix)).most_common(1)[0][0] for lg, ix in langs.items()}
    if as_is:  # a set of seeds: every passage once, the languages as the set has them
        return {lg: len(ix) for lg, ix in langs.items()}
    quota = (
        {"en": int(round(ENGLISH * n))} if "en" in langs else {}
    )  # no English passages (a set of seeds): no English pages
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


def usable(t, repeats=False):
    """The passage is real text of its language: script, no scrambled text layer, no repeated lines, no table of numbers cut out of its table.
    A verified (Google-translated) text comes from no PDF layer, and the Latin names and codes Google keeps do not count against its script."""
    v = t.get("verified")
    if (
        (t["lang"] in NO_TEXT_LAYER and not v)
        or not script_ok(re.sub("[A-Za-z]", "", t["text"]) if v else t["text"], t["lang"])
        or (garbled(t["text"]) and not v)
    ):
        return False
    lines = [x.strip() for x in t["text"].split("\n") if len(x.strip()) > 12]
    if not repeats and len(lines) - len(set(lines)) > 0.05 * max(
        len(lines), 1
    ):  # a passage that repeats its own lines (footnotes, running text) teaches repeated phrases
        return False
    digits = sum(c.isdigit() for c in t["text"])
    return repeats or t["kind"] == "table" or digits <= 0.3 * max(sum(c.isalnum() for c in t["text"]), 1)


def load_texts(files, repeats=False):
    """Passages of the jsonl files that are real text of their language: script, no cross-document boilerplate, no table of numbers cut out
    of its table, no scrambled text layers."""
    texts, seen = [], set()
    for f in files:
        for line in open(f):
            t = json.loads(line)
            key = {" ".join(t["text"].split())[k : k + 60] for k in (0, 200, 500) if len(t["text"]) > k + 60}
            if key & seen and not repeats or not usable(t, repeats):  # a seed set (repeats) keeps every passage
                continue
            seen |= key
            t["script"] = text_script(t["text"])
            texts.append(t)
    return texts


def draw_factors(n, quota, table_langs, rng):
    """n rows with the independent factors: type, style, pen, language, format, mode (normal / dense / table), writer, seeds."""
    types, styles, pens = [t for t in H.PERSONS if t not in ILLEGIBLE_TYPES], list(LEGIBLE_STYLES), list(TOOLS)
    cols = {
        "type": balanced(types, np.ones(len(types)), n, rng),
        "style": balanced(styles, np.ones(len(styles)), n, rng),
        "pen": balanced(pens, np.ones(len(pens)), n, rng),
        "lang": balanced(list(quota), list(quota.values()), n, rng),
        "iam": balanced([1, 0], [IAM_SHARE, 1 - IAM_SHARE], n, rng),
        "fmt": balanced(list(FMT_W), list(FMT_W.values()), n, rng),
        "dense": balanced([1, 0], [DENSE_SHARE, 1 - DENSE_SHARE], n, rng),
        "tab": balanced([1, 0], [TABLE_SHARE, 1 - TABLE_SHARE], n, rng),
    }
    iam = defaultdict(list)
    for e in personas():
        iam[e["type"]].append(e["page"])
    seeds = rng.permutation(10 * n)[: 3 * n].reshape(3, n) + 1000
    rows = []
    for i in range(n):
        r = dict(i=i, **{k: v[i] for k, v in cols.items()})
        r["seed"], r["wseed"], r["pseed"] = (int(s[i]) for s in seeds)
        r["persona"] = (
            f"iam:{iam[r['type']][int(rng.integers(len(iam[r['type']])))]}"
            if r.pop("iam") and iam[r["type"]]
            else r["type"]
        )
        r["dense"] = bool(r["dense"]) and not r["tab"]
        r["table"] = bool(r.pop("tab")) and r["lang"] in table_langs
        if (
            EXPECT.get(r["lang"], "Latin") not in PRINT_ONLY_NOT
        ):  # no fitted or cursive writer: print-like hands only, drawn per page
            r["type"] = r["persona"] = PRINT_HANDS[int(rng.integers(len(PRINT_HANDS)))]
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
        r["fill"] = float(
            rng.uniform(0.8, 1.0) if r["dense"] else rng.uniform(0.5, 0.9) if r["table"] else rng.uniform(0.5, 1.0)
        )
        lo = max(lo, MIN_XH_SCRIPT.get(EXPECT.get(r["lang"], "Latin"), 0))
        r["xh"] = float(np.exp(rng.uniform(np.log(lo), np.log(max(hi, lo * 1.2)))))
        u = rng.random()
        r["xh_ratio"] = float(
            rng.uniform(0.16, 0.27)
            if r["dense"]
            else rng.uniform(0.22, 0.5)
            if u < 0.8
            else rng.uniform(0.5, 0.62)
            if u < 0.9
            else rng.uniform(0.18, 0.22)
        )
        r["dy"] = float(rng.uniform(0.0, 0.5))
        r["tune"] = fitted_tune(rng)
        u = float(
            rng.uniform(0.3, 1.0)
        )  # word-to-word irregularity of the hand: rotation [deg], shift [x-heights], scale, shear [deg]
        r["wj"] = [round(6 * u, 2), round(0.3 * u, 3), round(0.12 * u, 3), round(8 * u, 2)]
        u = float(
            rng.uniform(0.3, 1.0)
        )  # word-to-word irregularity of the hand: rotation [deg], shift [x-heights], scale, shear [deg]
        r["wj"] = [round(6 * u, 2), round(0.3 * u, 3), round(0.12 * u, 3), round(8 * u, 2)]
        u = float(
            rng.uniform(0.3, 1.0)
        )  # word-to-word irregularity of the hand: rotation [deg], shift [x-heights], scale, shear [deg]
        r["wj"] = [round(6 * u, 2), round(0.3 * u, 3), round(0.12 * u, 3), round(8 * u, 2)]
        pp = paper(r["paper"], np.random.default_rng(r["seed"]), r["fmt"], r["dense"])
        xh = float(np.clip(pp.pitch * r["xh_ratio"], MIN_XH, MAX_XH)) if pp.pitch else r["xh"]
        lines = max(int(round(((pp.size[1] - pp.bottom - pp.first) / (pp.pitch or 3.3 * xh) + 1) * r["fill"])), 1)
        r["_cap"] = (
            1.1 * lines * (pp.size[0] - pp.left - pp.right) / (xh * CHAR_W.get(EXPECT.get(r["lang"], "Latin"), 1.8))
        )


def relayout(items, rng):
    """Page layouts of a reflowed document: two columns, a narrow side column, a text box away from the margin, two boxes at different heights.
    A leading heading stays full width; pages that already have columns are left as they are."""
    if len(items) < 4 or any(i["k"] in ("cols", "table") for i in items):
        return items
    u = rng.random()
    head = [items[0]] if items[0]["k"] == "h" else []
    body = [dict(i) for i in items[len(head) :]]
    sizes = np.cumsum([len(i.get("t", "")) + 1 for i in body])

    def split(share):
        k = int(np.searchsorted(sizes, share * sizes[-1])) + 1
        k = min(max(k, 1), len(body) - 1)
        a, b = body[:k], body[k:]
        a[0]["gap"], b[0]["gap"] = 0, 0
        return a, b

    if u < 0.30:
        a, b = split(0.5)
        cols = [dict(x=0.0, w=0.47, start=0, items=a), dict(x=0.53, w=0.47, start=0, items=b)]
    elif u < 0.40:
        a, b = split(0.28)
        cols = [dict(x=0.0, w=0.30, start=0, items=a), dict(x=0.35, w=0.65, start=0, items=b)]
    elif u < 0.52:
        body[0]["gap"] = 0
        cols = [dict(x=float(rng.uniform(0.08, 0.3)), w=float(rng.uniform(0.5, 0.65)), start=0, items=body)]
    elif u < 0.62:
        a, b = split(0.5)
        cols = [dict(x=0.0, w=0.46, start=0, items=a), dict(x=0.5, w=0.5, start=int(rng.integers(2, 7)), items=b)]
    else:
        return items
    return head + [dict(k="cols", cols=cols, gap=1 if head else 0)]


def assign_texts(rows, texts, quota, rng):
    """A passage of the language to every row: the shortest one that fills the page (longest page first), of the source kind drawn for the
    row, from an unused document; short passages are completed by more of the same document."""
    kinds = Counter(t["kind"] for t in texts)
    kw = {k: v for k, v in KIND_W.items() if kinds[k]}
    for r, k in zip(rows, balanced(list(kw), list(kw.values()), len(rows), rng)):
        r["kind_t"] = (
            "table" if r["table"] else "lecture" if r["dense"] and rng.random() < 0.75 and kinds["lecture"] else k
        )
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
                pool = pool or prose()  # small pools are used up by long pages: passages are used again

                def pick(ok):
                    ix = [j for j, i in enumerate(pool) if ok(texts[i])]
                    return next(
                        (j for j in ix if len(texts[pool[j]]["text"]) >= 0.9 * r["_cap"]), ix[-1] if ix else None
                    )

                j = next(
                    j
                    for j in (
                        pick(lambda t: t["kind"] == r["kind_t"] and t["doc"] not in used),
                        pick(lambda t: t["doc"] not in used),
                        pick(lambda t: t["kind"] == r["kind_t"]),
                        pick(lambda t: True),
                    )
                    if j is not None
                )
                t = texts[pool.pop(j)]
                used.add(t["doc"])
                items, extra = list(t["items"]), 0
                while len(" ".join(walk_text(items))) < 0.85 * r["_cap"] and pool and extra < (8 if r["dense"] else 5):
                    same = [k for k, i in enumerate(pool) if texts[i]["doc"] == t["doc"]]
                    more = [dict(i) for i in texts[pool.pop(same[0] if same else len(pool) - 1)]["items"]]
                    if more:
                        more[0]["gap"] = 1  # another passage: one blank line
                    items += more
                    extra += 1
                r.update(
                    items=relayout(items, rng), doc=t["doc"], kind=t["kind"], page=t["page"], ocr=bool(t.get("ocr"))
                )
            del r["_cap"], r["kind_t"]


def make_plan(texts, n, seed=1, as_is=False):
    """The plan: n page specs with their passages (list of dict)."""
    rng = np.random.default_rng(seed)
    quota = languages(texts, n, as_is)
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
    ap.add_argument(
        "--as-is", action="store_true", help="one page per passage, the languages as the passages have them"
    )
    a = ap.parse_args(argv)
    paper_mod.DRAW = False  # planning needs the geometry of the papers, not their pictures
    texts = load_texts(a.texts, a.as_is)
    rows = make_plan(texts, len(texts) if a.as_is else a.n, a.seed, a.as_is)
    with open(a.out, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"{len(rows)} pages from {len(texts)} passages;", Counter(r["lang"] for r in rows).most_common(8))
    print(Counter(r["kind"] for r in rows))


if __name__ == "__main__":
    main()
