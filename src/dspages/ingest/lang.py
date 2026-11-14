"""Language of a document. 1. PDFs with a text layer: fastText lid.176 and GlotLID on the text of the first picked pages (corpus_collecting ftnp, the
label code space of corpus.langstack); accepted when both models give the same language at p >= MIN_P and the letters are in its script
(hand.langs.script_ok). 2. The rest (scans, images, texts the models disagree on): the language of the nearest corpus pages in the SigLIP 2 space
(cosine kNN over pages of documents whose language the corpus knows), vote share = p; below KNN_MIN: `unknown`. Held-out accuracy of 2 is printed by `check_knn`."""
import sqlite3
import sys
from collections import Counter

import numpy as np
import pymupdf

from ..hand.langs import script_ok
from . import embstore
from .paths import CC_ROOT, EMB, LANG_MODELS, LAYOUTS, MANIFEST_DB
from .registry import db

MIN_P, TEXT_PAGES, TEXT_CHARS, MIN_CHARS = 0.8, 3, 4000, 200
COMPARABLE = (0.5, 2.0)                                                  # garbled: the English translation's word count / the input's outside this range
KNN_K, KNN_MIN, KNN_MIN_LANG = 15, 0.5, 20


def garbled(text):
    """True for text a PDF extraction corrupted, whatever the page's language label says: Google Translate (source language auto-detected; hwdet.gtranslate.lines,
    the browser where tools/seed_texts.py is imported) gives English whose word count (ICU word breaks: Chinese, Japanese, Thai ... have no spaces) is not
    comparable with the input's (COMPARABLE). No answer from Google is an error, not a verdict."""
    from hwdet import gtranslate as GT
    t = " ".join(text.split())[:TEXT_CHARS]
    if not words(t):
        return True
    out = GT.lines([t], "en")[0]
    if not out:
        raise RuntimeError(f"garbled: no answer from Google for {t[:80]!r}")
    return not COMPARABLE[0] <= words(out) / words(t) <= COMPARABLE[1]


def words(text):
    """Number of words of the text: ICU word segments with a letter."""
    import regex
    from ..layout.linebreak import breaks
    b = (0,) + breaks(text, 1)
    return sum(bool(regex.search(r"\p{L}", text[x:y])) for x, y in zip(b, b[1:]))


def page_text(rel, pages):
    try:
        with pymupdf.open(LAYOUTS / rel) as doc:
            return " ".join(" ".join(doc[p].get_text().split()) for p in pages[:TEXT_PAGES] if p < doc.page_count)[:TEXT_CHARS]
    except Exception:
        return ""


def text_langs():
    """Docs of kind pdf without a language: lid.176 + GlotLID on the text layer. -> {state: count}."""
    sys.path.insert(0, str(CC_ROOT))
    from corpus import ftnp
    from corpus.langstack import iso
    c = db()
    todo = c.execute("select doc, path from docs where kind = 'pdf' and lang is null and state != 'unreadable'").fetchall()
    texts = [page_text(p, [r[0] for r in c.execute("select page from pages where doc = ? order by page", (d,))]) for d, p in todo]
    ok = [i for i, t in enumerate(texts) if len(t) >= MIN_CHARS]
    ans = ftnp.read([texts[i] for i in ok], {"l176": LANG_MODELS / "lid176.bin", "glot": LANG_MODELS / "glotlid.bin"})
    out = Counter()
    for i, a in zip(ok, ans):
        (l1, p1), (l2, p2) = ((iso(a[m][0][0]), a[m][1][0]) if a[m][0] else (None, 0) for m in ("l176", "glot"))
        if l1 and l1 == l2 and min(p1, p2) >= MIN_P and script_ok(texts[i], l1):
            c.execute("update docs set lang = ?, lang_src = 'text', lang_p = ? where doc = ?", (l1, float(min(p1, p2)), todo[i][0]))
            out["text"] += 1
        else:
            out["no_text_answer"] += 1
    out["no_text_layer"] = len(todo) - len(ok)
    c.commit()
    return dict(out)


def reference():
    """(L2-normalised page-0..n vectors, language) of corpus pages whose document language is known, languages with >= KNN_MIN_LANG pages."""
    with sqlite3.connect(f"file:{MANIFEST_DB}?mode=ro", uri=True) as m:
        lang = dict(m.execute("select doc, lang from plan where lang is not null and lang not like 'unsure%' and lang != 'pending'"))
    man = embstore.manifest()
    import re
    hexre = re.compile(r"(?<![0-9a-f])[0-9a-f]{12}(?![0-9a-f])")
    dl = {d: lang.get(h.group(0)) for d, p in man.items() if (h := hexre.search(p.rsplit("/", 1)[-1]))}
    ids = np.load(EMB / "ids.npy")
    L = np.array([dl.get(int(d)) or "" for d in ids[:, 0]], object)
    keep = np.flatnonzero(L != "")
    cnt = Counter(L[keep])
    keep = keep[[cnt[l] >= KNN_MIN_LANG for l in L[keep]]]
    keep = keep[np.random.default_rng(0).permutation(len(keep))[:400_000]]
    return np.sort(keep), L


class Ref:
    """The reference pages as one normalised matrix (built once)."""

    def __init__(self):
        import torch
        self.idx, self.L = reference()
        E = np.load(EMB / "embeddings.npy", mmap_mode="r")
        self.R = torch.nn.functional.normalize(torch.from_numpy(np.asarray(E[self.idx]).astype(np.float32)), dim=1)
        self.lab = self.L[self.idx]

    def vote(self, queries, drop_self=False):
        """(language, vote share) per query vector (n, 1536): the K nearest reference pages by cosine, similarity-weighted; drop_self: the nearest hit is the query."""
        import torch
        out = []
        for a in range(0, len(queries), 2048):
            Q = torch.nn.functional.normalize(torch.from_numpy(queries[a:a + 2048].astype(np.float32)), dim=1)
            s, i = (Q @ self.R.T).topk(KNN_K + int(drop_self), dim=1)
            for k in range(len(Q)):
                v = Counter()
                for sim, j in zip(s[k, int(drop_self):].tolist(), i[k, int(drop_self):].tolist()):
                    v[self.lab[j]] += max(sim, 0.0)
                top, w = v.most_common(1)[0]
                out.append((top, w / sum(v.values())))
        return out


def check_knn(n=3000):
    """Held-out accuracy of the embedding vote on corpus reference pages (the page itself left out)."""
    ref = Ref()
    pick = np.sort(np.random.default_rng(1).permutation(len(ref.idx))[:n])
    E = np.load(EMB / "embeddings.npy", mmap_mode="r")
    got = ref.vote(np.asarray(E[ref.idx[pick]]), drop_self=True)
    right = np.array([g[0] == ref.lab[i] for g, i in zip(got, pick)])
    conf = np.array([g[1] for g in got])
    acc = {t: (round(float(right[conf >= t].mean()), 4), round(float((conf >= t).mean()), 3)) for t in (0.0, 0.5, 0.7, 0.9)}
    print("kNN language, held out: {vote share >=: (accuracy, share answered)}", acc)
    return acc


def embed_langs():
    """Docs still without a language, now embedded: the summed vote of their pages. -> {source: count}."""
    c = db()
    todo = {d for (d,) in c.execute("select doc from docs where lang is null and state in ('embedded', 'merged')")}
    if not todo:
        return {}
    ref = Ref()
    ids = np.load(EMB / "ids.npy")
    rows = np.flatnonzero(np.isin(ids[:, 0], list(todo)))
    E = np.load(EMB / "embeddings.npy", mmap_mode="r")
    votes = ref.vote(np.asarray(E[rows]))
    per = {}
    for r, (lg, p) in zip(rows, votes):
        per.setdefault(int(ids[r, 0]), Counter())[lg] += p
    out = Counter()
    for d, v in per.items():
        lg, w = v.most_common(1)[0]
        p = w / sum(v.values())
        ok = p >= KNN_MIN
        c.execute("update docs set lang = ?, lang_src = 'emb_knn', lang_p = ? where doc = ?", (lg if ok else "unknown", float(p), d))
        out["emb_knn" if ok else "unknown"] += 1
    c.commit()
    return dict(out)
