"""Border cases for the page-kind labels: unlabelled digital pages whose kind among plain / poems / formatted / tables / forms_blank /
forms_filled the current models are least sure of, sampled by the language distribution of the digital-born corpus.

  pick    pages 0-1 of the digital documents that have corpus embeddings (emb_out): an embedding model over all 18 labelled classes
          keeps the pages whose two likeliest classes are both among the six; per language (quota by the corpus share) the smallest
          margins go to a shortlist, the shortlist gets formatting features, and the product of the formatting xgb and the embedding
          model picks the N pages with the smallest margin. -> OUT/candidates.jsonl, OUT/img/<k>.jpg, OUT/sheet_<s>.jpg
  label   OUT/labels.json {k: class | "skip"} (the review) -> images into pagetype_labelled/<class>/bd_<doc>_<lang>_p<page>.jpg and their
          embeddings into pagetype_labelled/cache/pagekind_new_embeddings.npz (key: file stem)

python tools/pagekind_border.py pick --out DIR [--n 300]
python tools/pagekind_border.py label --out DIR
"""

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).parent))
import pagekind_eval as V  # noqa: E402

from dspages.hand import corpus  # noqa: E402
from dspages.hand import pagekind as PK  # noqa: E402

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
ROOT = V.UV / "pagetype_labelled"
NEW_EMB = V.CACHE / "pagekind_new_embeddings.npz"


def labelled():
    """(embeddings, folder class, stem) of every labelled image that has an embedding (the folders are the labels)."""
    pe = V.page_embeddings()
    rows = [
        (pe[f.stem], c.name, f.stem)
        for c in sorted(ROOT.iterdir())
        if c.is_dir() and c.name != "cache"
        for f in c.glob("*.jpg")
        if f.stem in pe
    ]
    return np.array([r[0] for r in rows]), [r[1] for r in rows], {r[2] for r in rows}


def render(pdf, no, out, long=1024):
    page = pymupdf.open(pdf)[no]
    z = long / max(page.rect.width, page.rect.height)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(z, z), colorspace=pymupdf.csRGB, alpha=False)
    cv2.imwrite(
        str(out),
        np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, 3)[..., ::-1],
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )


def _fast(job):
    try:
        return PK.features(pymupdf.open(job[0])[job[1]])
    except Exception:
        return None


def pick(out, n, seed=0):
    out.mkdir(parents=True, exist_ok=True)
    (out / "img").mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    E, lab, stems = labelled()
    classes = sorted(set(lab))
    emb_model = LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced").fit(
        normalize(E), [classes.index(c) for c in lab]
    )
    six = [classes.index(c) for c in SIX]

    db = corpus.connect()
    lang = corpus.languages(db)
    paths = {d: p for d, p, _ in corpus.pdf_documents(db)}  # digital text layer (plan.cls == text)
    used = {stem.split("_")[1] for stem in stems if len(stem.split("_")) > 1} | {
        corpus.doc_id(json.loads(l)["pdf"]) for l in open(V.SRC)
    }
    man = {r["id"]: corpus.doc_id(r["path"]) for r in map(json.loads, open(V.UV / "emb_out" / "manifest.jsonl"))}
    ids = np.load(V.UV / "emb_out" / "ids.npy")
    keep = np.array(
        [man.get(int(d)) in paths and man.get(int(d)) in lang and man.get(int(d)) not in used for d, _ in ids]
    )
    rows = np.flatnonzero(keep)
    emb = np.load(V.UV / "emb_out" / "embeddings.npy", mmap_mode="r")
    P = np.vstack(
        [
            emb_model.predict_proba(normalize(np.asarray(emb[rows[a : a + 20000]], np.float32)))
            for a in range(0, len(rows), 20000)
        ]
    )
    top2 = np.argsort(-P, 1)[:, :2]
    in_six = np.isin(top2, six).all(1)
    margin = P[np.arange(len(P)), top2[:, 0]] - P[np.arange(len(P)), top2[:, 1]]
    print(
        f"{len(rows)} embedded pages of digital documents; {in_six.sum()} with both likeliest classes among the six",
        flush=True,
    )

    docs = [man[int(ids[r, 0])] for r in rows]
    share = Counter(lang[d] for d in set(docs))
    tot = sum(share.values())
    quota = {lg: c / tot * n for lg, c in share.items()}
    by = defaultdict(list)
    for i in np.flatnonzero(in_six):
        by[lang[docs[i]]].append(i)
    short = []
    for lg, ix in by.items():
        k = int(np.ceil(3 * quota.get(lg, 0)))
        if k:
            ix = sorted(ix, key=lambda i: margin[i])
            seen, sel = set(), []
            for i in ix:  # one page per document
                if docs[i] not in seen:
                    seen.add(docs[i])
                    sel.append(i)
                if len(sel) >= k:
                    break
            short += sel
    jobs = [(str(paths[docs[i]]), int(ids[rows[i], 1])) for i in short]
    with ProcessPoolExecutor(6) as ex:
        F = list(ex.map(_fast, jobs, chunksize=8))
    ok = [k for k, f in enumerate(F) if f is not None]
    short = [short[k] for k in ok]
    F = np.array([F[k] for k in ok])
    print(f"shortlist {len(short)} pages with formatting features", flush=True)

    # formatting model on the matched labelled pages (six classes, text layer)
    X, Y, El, D, y, g, recs, t = V.load(V.SRC, list(SIX), False, False)
    fmt = V.Model("xgb").fit(X, y)
    pf = fmt.proba(F)
    pe6 = P[short][:, six]
    pe6 /= pe6.sum(1, keepdims=True)
    comb = np.log(pf + 1e-4) + np.log(pe6 + 1e-4)
    comb = np.exp(comb - comb.max(1, keepdims=True))
    comb /= comb.sum(1, keepdims=True)
    s2 = np.sort(comb, 1)
    m2 = s2[:, -1] - s2[:, -2]
    sel_by = defaultdict(list)
    for k in np.argsort(m2):
        sel_by[lang[docs[short[k]]]].append(k)
    final = []
    for lg, q in sorted(quota.items(), key=lambda kv: -kv[1]):  # largest remainder over the languages
        final += sel_by.get(lg, [])[: int(q)]
    rest = sorted(
        (k for lg, ks in sel_by.items() for k in ks[int(quota.get(lg, 0)) :]),
        key=lambda k: (-(quota[lang[docs[short[k]]]] % 1), m2[k]),
    )
    final += rest[: n - len(final)]
    rng.shuffle(final)
    with open(out / "candidates.jsonl", "w") as f:
        for j, k in enumerate(final):
            i = short[k]
            pdf, no = jobs[ok[k]]
            render(pdf, no, out / "img" / f"{j:04d}.jpg")
            f.write(
                json.dumps(
                    dict(
                        k=j,
                        doc=docs[i],
                        lang=lang[docs[i]],
                        pdf=pdf,
                        page=no,
                        emb_row=int(rows[i]),
                        p={c: round(float(v), 3) for c, v in zip(SIX, comb[k])},
                        margin=round(float(m2[k]), 3),
                    )
                )
                + "\n"
            )
    np.savez(
        out / "embeddings.npz",
        keys=np.array([str(j) for j in range(len(final))]),
        X=np.array([np.asarray(emb[rows[short[k]]], np.float32) for k in final]),
    )  # corpus page embedding of every candidate
    print("languages:", Counter(lang[docs[short[k]]] for k in final).most_common(30))
    sheets(out)


def sheets(out, per=10):
    c = [json.loads(l) for l in open(out / "candidates.jsonl")]
    for s in range(0, len(c), per):
        ims = []
        for r in c[s : s + per]:
            im = cv2.imread(str(out / "img" / f"{r['k']:04d}.jpg"))
            h, w = im.shape[:2]
            z = min(400 / w, 560 / h)
            im = cv2.resize(im, (int(w * z), int(h * z)))
            canvas = np.full((560, 400, 3), 255, np.uint8)
            canvas[: im.shape[0], : im.shape[1]] = im
            top = sorted(r["p"].items(), key=lambda kv: -kv[1])[:2]
            cv2.rectangle(canvas, (0, 0), (400, 20), (255, 255, 255), -1)
            cv2.putText(
                canvas, f"{r['k']} {r['lang']} {top[0][0][:6]}/{top[1][0][:6]}", (3, 15), 0, 0.5, (0, 0, 255), 2
            )
            ims.append(canvas)
        while len(ims) < per:
            ims.append(np.full((560, 400, 3), 255, np.uint8))
        cv2.imwrite(str(out / f"sheet_{s // per:02d}.jpg"), np.vstack([np.hstack(ims[:5]), np.hstack(ims[5:])]))


def label(out):
    c = {r["k"]: r for r in map(json.loads, open(out / "candidates.jsonl"))}
    lab = json.load(open(out / "labels.json"))
    emb = np.load(V.UV / "emb_out" / "embeddings.npy", mmap_mode="r")
    old = dict(np.load(NEW_EMB)) if NEW_EMB.exists() else {}
    own = dict(zip(*np.load(out / "embeddings.npz").values())) if "emb_row" not in next(iter(c.values())) else None
    keys, X = list(old.get("keys", [])), list(old.get("X", np.zeros((0, 768), np.float32)))
    n = Counter()
    for k, cls in lab.items():
        r = c[int(k)]
        if cls == "skip":
            continue
        stem = f"bd_{r['doc']}_{r['lang']}_p{r['page']}"
        shutil.copy(out / "img" / f"{int(k):04d}.jpg", ROOT / cls / f"{stem}.jpg")
        if stem not in keys:
            keys.append(stem)
            X.append(own[str(r["k"])] if own else np.asarray(emb[r["emb_row"]], np.float32))
        n[cls] += 1
    np.savez(NEW_EMB, keys=np.array(keys), X=np.array(X, np.float32))
    print("added:", dict(n), "skipped:", sum(v == "skip" for v in lab.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("pick", "label", "sheets"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    {
        "pick": lambda: pick(Path(a.out), a.n),
        "label": lambda: label(Path(a.out)),
        "sheets": lambda: sheets(Path(a.out)),
    }[a.cmd]()


if __name__ == "__main__":
    main()
