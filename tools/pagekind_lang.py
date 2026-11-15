"""Language balance of the labelled page kinds: English 25 % of every class, the other 75 % spread as evenly as the pages allow over the
languages of the digital corpus with at least MIN_DOCS documents. maps / blueprints have no language target (their labelled languages
stay as they are), music is out of it.

  scan     labelled pages without a language -> contact sheets of corpus_collecting tools.langscan (whole pages, its prompt) in
           CACHE/langscan/: agents write res_<batch>.json
  read     the agents' answers -> CACHE/pagekind_lang.json {item id: language}
  plan     per class: target count per language, the pages kept (CACHE/pagekind_balance.json {class: [item ids]}) and the deficits
           (CACHE/pagekind_deficit.json {class: {language: missing}})
  cands    for the deficits: corpus pages (embedded pages 0 / 1 of emb_out) of that language that the embedding model of all labelled
           pages puts in that class, `factor` per missing page, one per document -> candidate set of tools/pagekind_border.py in DIR
           (then tools/pagekind_agents.py batches / collect, tools/pagekind_border.py label)

python tools/pagekind_lang.py scan|read|plan
python tools/pagekind_lang.py cands --out DIR [--factor 2]
"""

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).parent))
from dspages import paths as _P

sys.path.insert(0, f"{_P.CORPUS}")
import pagekind_border as B  # noqa: E402
import pagekind_inventory as I  # noqa: E402

from dspages.hand import corpus  # noqa: E402
from dspages.hand.config import PATHS  # noqa: E402

EN_SHARE = 0.25
MIN_DOCS = 100
WORK = I.V.CACHE / "langscan"
BALANCE = I.V.CACHE / "pagekind_balance.json"
DEFICIT = I.V.CACHE / "pagekind_deficit.json"
SKIP = ("music",)
NOT_DIGITAL = ("images", "recieps")  # hardly any digital-born page: no language plan, no candidates
FOLDER_FILLED = ("scientific",)  # filled from Layouts/scientific_paper by tools/pagekind_full.py


def corpus_langs():
    db = corpus.connect()
    lang = corpus.languages(db, min_conf=0.0)
    n = Counter(lang.get(d) for d, c in db.execute("select doc,cls from plan") if c == "text")
    return sorted(l for l, k in n.items() if l and k >= MIN_DOCS and l != "en")


def scan():
    from tools.langscan import Config, make_sheets

    items = [i for i in I.inventory() if i.lang is None and i.cls not in I.NO_LANG + SKIP]
    pages = WORK / "pages"
    shutil.rmtree(WORK, ignore_errors=True)
    pages.mkdir(parents=True)
    ids, names = [], {}
    for k, i in enumerate(items):
        key = f"p{k:05d}"
        f = I.ROOT / i.id
        try:
            if f.suffix.lower() == ".pdf":
                pg = pymupdf.open(f)[0]
                z = 1000 / pg.rect.width
                im = Image.frombytes(
                    "RGB", *(lambda p: ((p.w, p.h), p.samples))(pg.get_pixmap(matrix=pymupdf.Matrix(z, z)))
                )
            else:
                im = Image.open(f).convert("RGB")
            im.thumbnail((1000, 1500))
            im.convert("L").save(pages / f"{key}.png")
        except Exception:
            continue
        ids.append(key)
        names[key] = i.id
    (WORK / "names.json").write_text(json.dumps(names))
    batches = make_sheets(Config(root=Path(PATHS.layouts)), ids, WORK, kind="pages")
    print(len(ids), "pages without a language;", len(batches), "batches; agents follow", WORK / "run.txt")


def read():
    from tools.langscan import read_results

    names = json.loads((WORK / "names.json").read_text())
    dets, bad = read_results(WORK)
    out = json.load(open(I.LANG_FILE)) if I.LANG_FILE.exists() else {}
    for d in dets:
        if d.doc in names:
            out[names[d.doc]] = d.lang or "none"
    json.dump(out, open(I.LANG_FILE, "w"), indent=0)
    print(len(dets), "answers;", Counter(v for v in out.values()).most_common(15), bad)


def targets(n, have, langs):
    """{language: target} for a class of n pages: English 25 %, the rest water-filled evenly over `langs` (a language with fewer pages
    than its share keeps a target of its share: the deficit is what to add)."""
    t = {"en": round(EN_SHARE * n)}
    each = (n - t["en"]) / len(langs)
    for l in langs:
        t[l] = max(int(round(each)), 1)
    return t


def plan():
    items = I.inventory()
    langs = corpus_langs()
    keep, deficit, report = {}, {}, []
    for c in I.classes():
        its = [i for i in items if i.cls == c and i.emb is not None]
        if c in SKIP or c in I.NO_LANG:
            keep[c] = [i.id for i in its]
            continue
        by = defaultdict(list)
        for i in its:
            by[i.lang or "?"].append(i.id)
        n = sum(len(v) for l, v in by.items() if l != "?")
        t = targets(n, by, langs)
        keep[c], deficit[c] = [], {}
        for l, want in t.items():
            have = by.get(l, [])
            keep[c] += sorted(have)[:want]  # a surplus language keeps its share (file order is random by hash)
            if len(have) < want:
                deficit[c][l] = want - len(have)
        dropped = sum(len(v) for l, v in by.items() if l not in t)
        report.append((c, n, len(keep[c]), sum(deficit[c].values()), len(by.get("?", [])), dropped))
    json.dump(keep, open(BALANCE, "w"))
    json.dump(deficit, open(DEFICIT, "w"), indent=0)
    print(f"{len(langs)} languages besides English (>= {MIN_DOCS} digital documents)")
    print("class          pages  kept  missing  no-language  other-languages")
    for r in report:
        print(f"{r[0]:14s} {r[1]:5d} {r[2]:5d} {r[3]:7d} {r[4]:7d} {r[5]:7d}")


def cands(out, factor):
    out = Path(out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    items = [i for i in I.inventory() if i.emb is not None]
    names = I.classes()
    y = np.array([names.index(i.cls) for i in items])
    model = LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced").fit(
        normalize(np.array([i.emb for i in items])), y
    )
    deficit = json.load(open(DEFICIT))
    db = corpus.connect()
    lang = corpus.languages(db, min_conf=0.0)
    paths = {d: p for d, p, _ in corpus.pdf_documents(db)}
    used = {i.doc for i in items}
    man = {r["id"]: corpus.doc_id(r["path"]) for r in map(json.loads, open(I.V.UV / "emb_out" / "manifest.jsonl"))}
    ids = np.load(I.V.UV / "emb_out" / "ids.npy")
    emb = np.load(I.V.UV / "emb_out" / "embeddings.npy", mmap_mode="r")
    want = {l for d in deficit.values() for l in d}
    rows = [
        k
        for k, (d, p) in enumerate(ids)
        if man.get(int(d)) in paths and man.get(int(d)) not in used and lang.get(man.get(int(d))) in want
    ]
    P = np.vstack(
        [
            model.predict_proba(normalize(np.asarray(emb[rows[a : a + 20000]], np.float32)))
            for a in range(0, len(rows), 20000)
        ]
    )
    taken, picks = set(), []
    for c, langs in deficit.items():
        if c in NOT_DIGITAL + FOLDER_FILLED:
            continue
        ci = names.index(c)
        for l, miss in langs.items():
            ix = [j for j in np.argsort(-P[:, ci]) if lang.get(man[int(ids[rows[j], 0])]) == l][: 40 * miss * factor]
            got = 0
            for j in ix:
                d = man[int(ids[rows[j], 0])]
                if d in taken or P[j, ci] < 0.2:
                    continue
                taken.add(d)
                picks.append((c, l, d, rows[j], float(P[j, ci])))
                got += 1
                if got >= miss * factor:
                    break
    with open(out / "candidates.jsonl", "w") as f:
        k = 0
        for c, l, d, r, p in picks:
            no = int(ids[r, 1])
            try:
                B.render(str(paths[d]), no, out / "img" / f"{k:04d}.jpg")
            except Exception:  # file gone from the corpus
                continue
            f.write(
                json.dumps(
                    dict(
                        k=k,
                        doc=d,
                        lang=l,
                        pdf=str(paths[d]),
                        page=no,
                        emb_row=int(r),
                        p={c: round(p, 3)},
                        margin=None,
                        want=c,
                    )
                )
                + "\n"
            )
            k += 1
    print(
        len(picks),
        "candidates:",
        Counter(p[0] for p in picks),
        "| asked:",
        {c: sum(v.values()) for c, v in deficit.items()},
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("scan", "read", "plan", "cands"))
    ap.add_argument("--out")
    ap.add_argument("--factor", type=int, default=2)
    a = ap.parse_args()
    {"scan": scan, "read": read, "plan": plan, "cands": lambda: cands(a.out, a.factor)}[a.cmd]()
