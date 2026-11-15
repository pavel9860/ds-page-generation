"""Full duplicate search over the corpus documents from ALL embedded pages, a dry run with samples; nothing is moved or deleted here.

pagetype.dedup compares pages 0 / 1 only. Now that most documents have more embedded pages, a candidate pair found on pages 0 / 1 (the
same prefilter and rules as pagetype.dedup) is checked on every page both documents have embedded: the same template with other content
(forms, certificates, a series of letters) agrees on the first pages and disagrees on the rest, a real copy agrees everywhere.

  python tools/dedup_full.py search            # pairs -> WORK/pairs.parquet, counts
  python tools/dedup_full.py samples [--n 40]  # side-by-side jpgs of pairs per class -> WORK/samples/<class>/
  python tools/dedup_full.py plan              # moves -> WORK/moves.tsv (files that pagetype.dedup apply would move)

Data goes to UVTM/tests/work/dedup_full (WORK)."""

import argparse
import collections
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf

from dspages import paths as _P
from dspages.hand.config import PATHS

sys.path.insert(0, f"{_P.CORPUS}")
from pagetype import dedup  # noqa: E402
from pagetype.layouts import Layouts, load_manifest  # noqa: E402

LAYOUTS = Path(PATHS.layouts)
EMB = Path(PATHS.uvtm) / "emb_out"
WORK = Path(PATHS.uvtm) / "tests" / "work" / "dedup_full"
LABELS = f"{_P.CORPUS}/pagetype/data/labels.json"
EXTRA_OK = 0.95  # mean cosine of the pages beyond 0 / 1 for a copy; below it: same template, other content
SIDE = 420  # height of a sample page, px

pymupdf.TOOLS.mupdf_display_errors(False)


def load():
    E, ids = np.load(EMB / "embeddings.npy", mmap_mode="r"), np.load(EMB / "ids.npy")
    return E, ids, Layouts(LAYOUTS, load_manifest(EMB / "manifest.jsonl"))


def pages_of(ids):
    """{doc: {page: row}}."""
    out = collections.defaultdict(dict)
    for r, (d, p) in enumerate(ids):
        out[int(d)][int(p)] = r
    return out


def unit(E, rows):
    P = np.asarray(E[rows], np.float32)
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def candidates(E, pg, thr=dedup.LOW):
    """[(mode, a, b, cosine on the first two embedded pages)]: "one" for documents with a single embedded page, "two" for the rest."""
    out = []
    for mode, docs in (
        ("one", [d for d, p in pg.items() if len(p) == 1]),
        ("two", [d for d, p in pg.items() if len(p) >= 2]),
    ):
        k = 1 if mode == "one" else 2
        pages = [unit(E, [sorted(pg[d].items())[j][1] for d in docs]) for j in range(k)]
        print(mode, len(docs), "documents", flush=True)
        out += [(mode, docs[int(i)], docs[int(j)], float(c)) for i, j, c in dedup.pair_search(pages, thr)]
    return out


def verify(E, pg, cand):
    """Rows of every candidate pair with the agreement on the pages beyond the first two: common pages, their mean / min cosine."""
    rows = []
    for mode, a, b, v in cand:
        sa, sb = sorted(pg[a]), sorted(pg[b])
        common = sorted(set(sa) & set(sb) - set(sa[:2]) - set(sb[:2]))
        if common:
            cs = np.einsum("ij,ij->i", unit(E, [pg[a][p] for p in common]), unit(E, [pg[b][p] for p in common]))
            mean, low = float(cs.mean()), float(cs.min())
        else:
            mean = low = np.nan
        rows.append((mode, a, b, v, len(sa), len(sb), len(common), mean, low))
    return pd.DataFrame(rows, columns="mode a b cos pa pb common extra_mean extra_min".split())


def classify(P, lay):
    """Adds path a / b, same language, the old rule's verdict (pages 0 / 1 only) and the class after the check of the other pages."""
    P["path_a"], P["path_b"] = [lay.path(d) for d in P.a], [lay.path(d) for d in P.b]
    P["same_lang"] = [lay.same_lang(a, b) for a, b in zip(P.a, P.b)]
    P["rule"] = [
        dedup.RULES[m](v, s, str(pa), str(pb))
        for m, v, s, pa, pb in zip(P["mode"], P.cos, P.same_lang, P.path_a, P.path_b)
    ]
    P["both_files"] = P.path_a.notna() & P.path_b.notna() & (P.path_a != P.path_b)
    cls = np.where(
        P.common == 0,
        "first pages only",
        np.where(P.extra_mean >= EXTRA_OK, "confirmed by more pages", "template only"),
    )
    P["cls"] = np.where(P.rule & P.both_files, cls, "no (rule or file)")
    return P


def cmd_search(_):
    WORK.mkdir(parents=True, exist_ok=True)
    E, ids, lay = load()
    pg = pages_of(ids)
    P = classify(verify(E, pg, candidates(E, pg)), lay)
    P.to_parquet(WORK / "pairs.parquet")
    print(f"{len(P)} candidate pairs (cosine >= {dedup.LOW} on the first pages)")
    print(P.cls.value_counts().to_string())
    ok = P[P.cls.isin(["confirmed by more pages", "first pages only"])]
    for name, s in (
        ("pairs the old rule accepts", P[P.rule & P.both_files]),
        ("pairs after the check of the other pages", ok),
    ):
        files = set(s.path_a) | set(s.path_b)
        print(f"{name}: {len(s)} pairs, {len(files)} files involved")
    for lo in (0.0, 0.9, 0.95, 0.98, 0.99):
        s = P[(P.cls == "confirmed by more pages") & (P.extra_mean >= lo)]
        print(f"confirmed, extra pages mean >= {lo}: {len(s)} pairs")
    print("mean extra-page agreement (candidate pairs with common pages):", P.extra_mean.describe().round(3).to_dict())


def plan_files(P, lay):
    """Moves for the pairs that stand after the check: {moved file: (kept file, cosine)} (pagetype.dedup.plan_moves on the edge list)."""
    ok = P[P.cls.isin(["confirmed by more pages", "first pages only"])]
    rules = {m: (lambda v, same, a, b: True) for m in ("one", "two")}  # the old rules were applied in classify
    return dedup.plan_moves(list(zip(ok["mode"], ok.a, ok.b, ok.cos)), lay, dedup.protected_paths(lay, LABELS), rules)


def cmd_plan(_):
    _, _, lay = load()
    P = pd.read_parquet(WORK / "pairs.parquet")
    moves = plan_files(P, lay)
    pd.DataFrame([(s, k, v) for s, (k, v) in moves.items()], columns=["moved_from", "kept_copy", "cosine"]).to_csv(
        WORK / "moves.tsv", sep="\t", index=False
    )
    print(
        f"{len(moves)} files would move (kept: {len({k for k, _ in moves.values()})}); by folder:",
        collections.Counter(p.split("/")[0] for p in moves).most_common(10),
    )


def thumb(path, page=0):
    try:
        with pymupdf.open(path) as d:
            if page >= d.page_count:
                return None
            pm = d[page].get_pixmap(matrix=pymupdf.Matrix(*(2 * [SIDE / d[page].rect.height])), alpha=False)
            return pm
    except Exception:
        return None


def montage(path_a, path_b, label, out):
    import cv2

    tiles = []
    for p in (path_a, path_b):
        for pm in (thumb(p, 0), thumb(p, 2)):
            if pm is None:
                tile = np.full((SIDE, SIDE * 3 // 4, 3), 255, np.uint8)
            else:
                tile = cv2.resize(
                    np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, pm.n)[..., :3]
                    if pm.n >= 3
                    else np.repeat(np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, 1), 3, 2),
                    (SIDE * 3 // 4, SIDE),
                )
            tiles.append(np.pad(tile, ((0, 0), (0, 6), (0, 0)), constant_values=127))
    img = np.concatenate(tiles, 1)
    cv2.putText(img, label[:150], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 0, 0), 1)
    cv2.imwrite(str(out), img[..., ::-1])


def cmd_samples(a):
    P = pd.read_parquet(WORK / "pairs.parquet")
    rng = np.random.default_rng(0)
    P["band"] = pd.cut(P.cos, [0, 0.97, 0.98, 0.99, 1.01]).astype(str)
    for cls, s in P[P.cls != "no (rule or file)"].groupby("cls"):
        d = WORK / "samples" / cls.replace(" ", "_")
        d.mkdir(parents=True, exist_ok=True)
        for k, (_, r) in enumerate(s.iloc[rng.permutation(len(s))[: a.n]].iterrows()):
            label = f"{r.cos:.3f} extra {r.extra_mean:.3f} ({int(r.common)}p) | {r.path_a} | {r.path_b}"
            montage(LAYOUTS / r.path_a, LAYOUTS / r.path_b, label, d / f"{k:03d}_{r.cos:.3f}.jpg")
        print(cls, len(s), "pairs, samples in", d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("search").set_defaults(f=cmd_search)
    sub.add_parser("plan").set_defaults(f=cmd_plan)
    sp = sub.add_parser("samples")
    sp.add_argument("--n", type=int, default=40)
    sp.set_defaults(f=cmd_samples)
    a = ap.parse_args()
    a.f(a)
