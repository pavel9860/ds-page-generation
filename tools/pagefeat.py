"""Hand-built layout features of digital pdf pages (pymupdf, no rendering) to compare with the SigLIP page embedding: does the
feature set separate what the embedding cannot (scripts, ruled forms, checkboxes, column layout)?

python tools/pagefeat.py extract --n 24000     pool pages (digital, live) -> OUT/pagefeat.parquet (resumable, 6 workers)
python tools/pagefeat.py eval                  probes label / lang / script / cluster from embedding, features, both -> OUT/pagefeat_eval.csv
"""
import argparse
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import diverse_ds as D  # noqa: E402

GRID = 6
SCRIPTS = ("latin", "cyrillic", "greek", "cjk", "kana", "hangul", "arabic", "hebrew", "devanagari", "thai", "other_indic", "other")


def script_of(ch):
    o = ord(ch)
    if o < 0x250 or 0x1E00 <= o < 0x1F00:
        return 0
    if 0x400 <= o < 0x530:
        return 1
    if 0x370 <= o < 0x400 or 0x1F00 <= o < 0x2000:
        return 2
    if 0x4E00 <= o < 0xA000 or 0x3400 <= o < 0x4DC0:
        return 3
    if 0x3040 <= o < 0x3100:
        return 4
    if 0xAC00 <= o < 0xD7B0 or 0x1100 <= o < 0x1200:
        return 5
    if 0x600 <= o < 0x700 or 0x750 <= o < 0x780 or 0xFB50 <= o < 0xFE00:
        return 6
    if 0x590 <= o < 0x600:
        return 7
    if 0x900 <= o < 0x980:
        return 8
    if 0xE00 <= o < 0xE80:
        return 9
    if 0x980 <= o < 0xE00:
        return 10
    return 11


def dist(f, name, v, qs=True):
    """Count, mean, median, std (+ p10, p90, max) of the values v of one page as features name_*."""
    v = np.asarray(v, float)
    if not len(v):
        v = np.zeros(1)
        f[f"{name}_n"] = 0
    else:
        f[f"{name}_n"] = np.log1p(len(v))
    f.update({f"{name}_mean": v.mean(), f"{name}_med": np.median(v), f"{name}_std": v.std()})
    if qs:
        f.update({f"{name}_p10": np.percentile(v, 10), f"{name}_p90": np.percentile(v, 90), f"{name}_max": v.max()})


def feat(job):
    """(row, feature vector or None) of one pdf page."""
    i, path, page = job
    import pymupdf
    pymupdf.TOOLS.mupdf_display_errors(False)
    try:
        p = D.resolve(path)
        if p is None or not str(p).lower().endswith(".pdf"):
            return i, None
        d = pymupdf.open(p)
        pg = d[min(int(page), len(d) - 1)]
        W, H = pg.rect.width, pg.rect.height
        f = {"ratio": H / W}
        td = pg.get_text("dict")
        blocks = [b for b in td["blocks"] if b["type"] == 0]
        chars, sizes, bold, sc = 0, [], 0, np.zeros(len(SCRIPTS))
        occ = np.zeros((GRID, GRID))
        lefts, rights, lines_n, lh = [], [], 0, []
        lw, cpl, bw_, bh_, top, bot = [], [], [], [], [], []
        fonts = set()
        for b in blocks:
            x0, y0, x1, y1 = b["bbox"]
            bw_.append((x1 - x0) / W)
            bh_.append((y1 - y0) / H)
            occ[min(int(y0 / H * GRID), GRID - 1):min(int(y1 / H * GRID) + 1, GRID),
                min(int(x0 / W * GRID), GRID - 1):min(int(x1 / W * GRID) + 1, GRID)] += 1
            for ln in b["lines"]:
                lines_n += 1
                lefts.append(ln["bbox"][0] / W)
                rights.append(ln["bbox"][2] / W)
                lh.append(ln["bbox"][3] - ln["bbox"][1])
                lw.append((ln["bbox"][2] - ln["bbox"][0]) / W)
                cpl.append(sum(len(sp_["text"]) for sp_ in ln["spans"]))
                top.append(ln["bbox"][1] / H)
                bot.append(ln["bbox"][3] / H)
                for s in ln["spans"]:
                    t = s["text"]
                    n = len(t.strip())
                    chars += n
                    sizes += [s["size"]] * n
                    bold += n * bool(s["flags"] & 16)
                    fonts.add(s["font"])
                    for ch in t:
                        if not ch.isspace():
                            sc[script_of(ch)] += 1
        sc = sc / max(sc.sum(), 1)
        f.update({f"sc_{n}": v for n, v in zip(SCRIPTS, sc)})
        sz = np.array(sizes) if sizes else np.zeros(1)
        f.update(chars=np.log1p(chars), blocks=len(blocks), lines=lines_n, fonts=len(fonts), size_mean=sz.mean(), size_std=sz.std(),
                 size_max=sz.max(), bold=bold / max(chars, 1), line_h=np.mean(lh) / H if lh else 0,
                 text_area=sum((b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]) for b in blocks) / (W * H))
        dist(f, "span_size", sizes)
        dist(f, "line_w", lw)
        dist(f, "chars_line", cpl)
        dist(f, "block_w", bw_)
        dist(f, "block_h", bh_)
        ts = np.sort(np.array(top)) if top else np.zeros(1)
        gaps = np.diff(ts) if len(ts) > 1 else np.zeros(1)
        dist(f, "line_gap", gaps)                                       # vertical spacing of lines (leading, paragraph gaps)
        f.update(top_margin=ts.min(), bottom_margin=1 - max(bot) if bot else 1.0, white=1 - min(1.0, f["text_area"]))
        L, R = np.array(lefts) if lefts else np.zeros(1), np.array(rights) if rights else np.zeros(1)
        f.update(left_min=L.min(), right_max=R.max(), left_std=L.std(), right_std=R.std(),
                 left_modes=len(np.unique(np.round(L, 2))) / max(len(L), 1), columns=float(np.sum(np.histogram(L, bins=10, range=(0, 1))[0] > 0.1 * len(L))))
        for a in range(GRID):
            for c in range(GRID):
                f[f"occ_{a}{c}"] = occ[a, c]
        hl = vl = rects = boxes = fills = curves = 0
        hlen = 0.0
        hls, vls, rsz, itn = [], [], [], []
        draws = pg.get_drawings()
        for dr in draws:
            itn.append(len(dr["items"]))
            for it in dr["items"]:
                if it[0] == "l":
                    dx, dy = abs(it[2].x - it[1].x), abs(it[2].y - it[1].y)
                    if dy < 1 and dx > 0.05 * W:
                        hl += 1
                        hlen += dx / W
                        hls.append(dx / W)
                    elif dx < 1 and dy > 0.02 * H:
                        vl += 1
                        vls.append(dy / H)
                elif it[0] == "re":
                    r = it[1]
                    w_, h_ = r.width, r.height
                    if h_ < 1.5 and w_ > 0.05 * W:
                        hl += 1
                        hlen += w_ / W
                        hls.append(w_ / W)
                    elif w_ < 1.5 and h_ > 0.02 * H:
                        vl += 1
                        vls.append(h_ / H)
                    elif 4 < w_ < 25 and 4 < h_ < 25 and abs(w_ - h_) < 0.25 * max(w_, h_):
                        boxes += 1
                    else:
                        rects += 1
                        rsz.append(w_ * h_ / (W * H))
                    if dr.get("fill") is not None and w_ * h_ > 0.02 * W * H:
                        fills += 1
                elif it[0] == "c":
                    curves += 1
        imgs = pg.get_image_info()
        isz = [(max(0, min(W, i_["bbox"][2]) - max(0, i_["bbox"][0])) * max(0, min(H, i_["bbox"][3]) - max(0, i_["bbox"][1]))) / (W * H) for i_ in imgs]
        cover = sum(isz)
        dpi = [max(i_["width"] / max((i_["bbox"][2] - i_["bbox"][0]) / 72, 1e-3), i_["height"] / max((i_["bbox"][3] - i_["bbox"][1]) / 72, 1e-3)) for i_ in imgs]
        f.update(hlines=hl, vlines=vl, hlen=hlen, rects=rects, checkboxes=boxes, big_fills=fills, curves=curves, images=len(imgs),
                 img_cover=min(cover, 1.0), widgets=len(list(pg.widgets() or [])), links=len(pg.get_links()), annots=len(list(pg.annots() or [])),
                 drawings=len(draws), img_max=max(isz) if isz else 0.0, img_small=sum(x < 0.01 for x in isz), img_gray=sum(i_.get("colorspace", 3) == 1 for i_ in imgs),
                 img_mask=sum(bool(i_.get("has-mask")) for i_ in imgs))
        dist(f, "img_area", isz)
        dist(f, "img_dpi", np.log10(np.maximum(dpi, 1)) if dpi else [])
        dist(f, "hline_len", hls)
        dist(f, "vline_len", vls)
        dist(f, "rect_area", rsz)
        dist(f, "draw_items", itn)
        return i, f
    except Exception:
        return i, None


def cmd_extract(a):
    P = pd.read_parquet(D.OUT / "pool.parquet", columns=["path", "scan", "scope", "page"])
    cand = P[~P.scan & (P.scope == "live") & P.path.str.lower().str.endswith(".pdf")]
    pick = cand.sample(min(a.n, len(cand)), random_state=0)
    f = D.OUT / "pagefeat.parquet"
    done = pd.read_parquet(f) if f.exists() else pd.DataFrame()
    todo = [(i, r.path, r.page) for i, r in zip(pick.index, pick.itertuples()) if i not in set(done.index)]
    out, rows = {}, []
    with ProcessPoolExecutor(D.WORKERS, initializer=D._init, initargs=(D.hash_index(),)) as ex:
        for n, (i, v) in enumerate(ex.map(feat, todo, chunksize=32)):
            if v is not None:
                out[i] = v
            if (n + 1) % 4000 == 0 or n == len(todo) - 1:
                done = pd.concat([done, pd.DataFrame.from_dict(out, orient="index")])
                done.to_parquet(f)
                out = {}
                print(len(done), "of", len(pick), flush=True)


def probe(Z, y, seed=0):
    """Accuracy and macro-F1 of a logistic probe (train 70 % / test 30 %) of y from Z."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=seed)
    m = LogisticRegression(max_iter=300, C=2).fit(Z[tr], y[tr])
    p = m.predict(Z[te])
    return round((p == y[te]).mean(), 3), round(f1_score(y[te], p, average="macro"), 3)


def knn_agree(Z, E, k=10, n=4000):
    """Mean share of the k nearest neighbours (in Z) that are also among the k nearest in E: how much the two spaces agree."""
    rng = np.random.default_rng(0)
    q = rng.choice(len(Z), n, replace=False)
    a = np.argsort(-(Z[q] @ Z.T), 1)[:, 1:k + 1]
    b = np.argsort(-(E[q] @ E.T), 1)[:, 1:k + 1]
    return round(float(np.mean([len(set(x) & set(y)) / k for x, y in zip(a, b)])), 3)


def cmd_eval(_):
    from sklearn.preprocessing import StandardScaler
    F = pd.read_parquet(D.OUT / "pagefeat.parquet")
    P = pd.read_parquet(D.OUT / "pool.parquet").loc[F.index]
    X = np.load(D.OUT / "X.npy", mmap_mode="r")
    E = np.asarray(X[F.index.values], np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    G = F.copy()
    for c in G.columns:
        if c.startswith(("blocks", "lines", "hlines", "vlines", "rects", "checkboxes", "curves", "images", "links", "occ_", "widgets", "annots", "fonts", "big_", "drawings", "img_small", "img_gray", "img_mask")):
            G[c] = np.log1p(G[c])
    Z = StandardScaler().fit_transform(G.fillna(0).values).astype(np.float32)
    Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    script = F[[c for c in F.columns if c.startswith("sc_")]].idxmax(axis=1).values
    both = np.hstack([E * 4, Z / np.sqrt(Z.shape[1])]).astype(np.float32)
    lang = P.lang.where(P.lang.map(P.lang.value_counts()) >= 150, "other").values
    rows = []
    for tname, y in (("label", P.label.values), ("lang", lang), ("script", script), ("cluster_c2", P.c2.values.astype(str))):
        for sname, S in (("embedding", E), ("features", Z), ("both", both)):
            acc, f1 = probe(S, y)
            rows.append(dict(target=tname, space=sname, acc=acc, macro_f1=f1))
    R = pd.DataFrame(rows).pivot(index="target", columns="space", values=["acc", "macro_f1"])
    print(R.to_string())
    print("knn agreement features vs embedding (k=10):", knn_agree(Zn, E))
    # per-feature-group: which single groups carry label / lang information
    groups = {"script": "sc_", "grid": "occ_", "lines_rects_boxes": ("hlines", "vlines", "hlen", "rects", "checkboxes", "big_fills", "curves", "hline_", "vline_", "rect_", "draw"),
              "text_stats": ("chars", "blocks", "lines", "fonts", "size_", "span_", "bold", "line_h", "text_area", "line_w", "chars_line", "block_", "line_gap", "top_", "bottom_", "white"), "margins": ("left_", "right_", "columns"),
              "images": ("images", "img_"), "widgets": ("widgets", "links", "annots")}
    g = []
    for n, pre in groups.items():
        cols = [i for i, c in enumerate(G.columns) if c.startswith(pre)]
        for tname, y in (("label", P.label.values), ("lang", lang)):
            g.append(dict(group=n, target=tname, n_feat=len(cols), acc=probe(Z[:, cols], y)[0]))
    print(pd.DataFrame(g).pivot(index="group", columns="target", values="acc").to_string())
    R.to_csv(D.OUT / "pagefeat_eval.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--n", type=int, default=24000)
    sp.add_parser("eval")
    a = ap.parse_args()
    {"extract": cmd_extract, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
