"""Real handwritten pages of the corpus in the 3K plan's language distribution"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from sklearn.linear_model import LogisticRegression

from . import realbank
from .config import HWPICK as C
from .config import LANGDS as D
from .paths import LANGDS, UVTM

_LANG = re.compile(r"_([a-z]{2,3})\.(pdf|png|jpe?g)$")


def lang_of(path):
    m = _LANG.search(path)
    return D.alias.get(m.group(1), m.group(1)) if m else None


def render(path, page, side=C.side):
    pg = pymupdf.open(UVTM / "Layouts" / path)[page]
    k = side / max(pg.rect.width, pg.rect.height)
    px = pg.get_pixmap(matrix=pymupdf.Matrix(k, k), alpha=False, colorspace=pymupdf.csRGB)
    return np.frombuffer(px.samples, np.uint8).reshape(px.h, px.w, 3)


class Corpus:
    """The page vectors of the corpus with the language and type label of each document."""

    def __init__(self):
        self.E = np.load(UVTM / "emb_out/embeddings.npy", mmap_mode="r")
        self.ids = np.load(UVTM / "emb_out/ids.npy")
        self.docs = {}
        for line in open(UVTM / "emb_out/doc_types.jsonl"):
            r = json.loads(line)
            self.docs[r["doc"]] = r
        n = int(self.ids[:, 0].max()) + 1
        self.lang = np.array(
            [lang_of(self.docs[d]["path"]) or "" if d in self.docs else "" for d in range(n)], dtype=object
        )
        self.label = np.array([self.docs[d]["label"] if d in self.docs else "" for d in range(n)], dtype=object)

    def vectors(self, rows):
        Z = np.asarray(self.E[sorted(rows)], np.float32)
        return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-6)


def train(corpus, excluded=()):
    """The classifier; `excluded` are (path, page) of pages never trained on."""
    R, items = realbank.load_real()
    pos, neg = realbank.reviewed(items)
    pos, neg = [i for i in pos if items[i] not in excluded], [i for i in neg if items[i] not in excluded]
    first = {}
    for row, (d, _) in enumerate(corpus.ids):
        first.setdefault(int(d), row)
    pool = [d for d in range(len(corpus.lang)) if corpus.label[d] in C.neg_labels and d in first]
    other = [first[d] for d in np.random.default_rng(0).choice(pool, C.n_neg, replace=False)]
    X = np.vstack([R[pos], R[neg], corpus.vectors(other)])
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg) + len(other))]
    return LogisticRegression(max_iter=3000, C=C.c, class_weight="balanced").fit(X, y), len(pos)


def score(corpus, clf, langs):
    """({row: probability} of the pages of `langs`)."""
    rows = np.flatnonzero(np.isin(corpus.lang[corpus.ids[:, 0]], list(langs)))
    p = np.zeros(len(corpus.ids), np.float32)
    for s in range(0, len(rows), C.chunk):
        r = rows[s : s + C.chunk]
        p[r] = clf.predict_proba(corpus.vectors(r))[:, 1]
    return rows, p


def candidates(corpus, rows, p, exclude_docs):
    """{language: [row]} best page of each document first."""
    out, seen = defaultdict(list), set()
    for i in rows[np.argsort(-p[rows])]:
        d = int(corpus.ids[i, 0])
        if d in corpus.docs and d not in seen and Path(corpus.docs[d]["path"]).stem not in exclude_docs:
            seen.add(d)
            out[corpus.lang[d]].append(int(i))
    return out


def choose(cand, p, quota, n):
    chosen = {l: [i for i in cand[l] if p[i] >= C.lo][:q] for l, q in quota.items()}
    for floor in (C.hi, C.mid, C.lo):
        while sum(map(len, chosen.values())) < n:
            open_ = [
                l for l in quota if len(chosen[l]) < min(C.cap, len(cand[l])) and p[cand[l][len(chosen[l])]] >= floor
            ]
            if not open_:
                break
            l = min(open_, key=lambda l: (len(chosen[l]), l))
            chosen[l].append(cand[l][len(chosen[l])])
    return chosen


def select(out, n, excluded_pages=(), excluded_docs=()):
    """Writes OUT/<lang>_<doc>_<page>.jpg and OUT/meta.json {name"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plan = Counter(r["lang"] for r in json.loads((LANGDS / "ds3k_plan.json").read_text()) if r["kind"] == "hand")
    corpus = Corpus()
    clf, _ = train(corpus, set(excluded_pages))
    rows, p = score(corpus, clf, plan)
    cand = candidates(corpus, rows, p, set(excluded_docs))
    base, extra = divmod(n, len(plan))
    langs = sorted(plan, key=lambda l: -sum(p[i] >= C.hi for i in cand[l]))
    chosen = choose(cand, p, {l: base + (k < extra) for k, l in enumerate(langs)}, n)
    meta = {}
    for lang, picks in chosen.items():
        for i in picks:
            d, page = int(corpus.ids[i, 0]), int(corpus.ids[i, 1])
            path = corpus.docs[d]["path"]
            name = f"{lang}_{Path(path).stem.split('_')[0]}_{page}"
            try:
                cv2.imwrite(
                    str(out / f"{name}.jpg"), render(path, page)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, C.quality]
                )
            except Exception:
                continue
            meta[name] = dict(
                lang=lang,
                path=path,
                folder=path.split("/")[0],
                page=page,
                p=round(float(p[i]), 3),
                doc_label=corpus.docs[d]["label"],
            )
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return meta
