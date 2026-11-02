"""Shape space of 3D page maps and the generator parameters behind each position.

Space: the 517 engineered features (maps3d_emb.blocks) and SigLIP 2 layer 4 of the shade image (maps3d_emb), each block
standardised and weighted 1 / sqrt(dim) (both blocks count alike), PCA to N_PC components. Uses the vectors already in
maps3d_fe / maps3d_emb (3,000 generated, 1,734 real sources).
Parameters of a generated map: its plan row and its scene as drawn (sample_ends on its geometry stream, as cli._geometry).
1. Coverage: each real map's distance to the nearest generated map against the generated maps' own nearest-neighbour
   distances (gap: beyond their 95th percentile); per real category; real k-means clusters with their coverage.
2. Parameters -> position: XGBoost regressors of the PCs from the parameters (R2 per PC), importance of each parameter
   weighted by the PC variance.
3. Grammage: shift of the predicted position over the gsm range, others held, per scene kind.
4. Position -> parameters: the generated neighbours of a target (a real cluster centre) and their parameters.

python tools/maps3d_space.py --out DIR
DIR/space.npz (pcs, items order), DIR/params.jsonl, DIR/report.json, DIR/space_*.png
"""
import argparse
import json
from pathlib import Path

import numpy as np

FE = Path("/run/media/p/D/ML_DS/UVTM/GridPages/maps3d_fe")
EMB = Path("/run/media/p/D/ML_DS/UVTM/GridPages/maps3d_emb")
UV = Path("/run/media/p/D/ML_DS/UVTM/uv_ds_2.0.0")
BLOCKS = ("base", "dog", "normgrad", "ridge", "gauss_img", "curv")
N_PC = 20
K_REAL = 12
THREADS = 3


def space():
    its = [json.loads(l) for l in open(FE / "items.jsonl")]
    parts = [np.hstack([np.load(FE / f"fb_{b}.npy") for b in BLOCKS]), np.load(EMB / "emb_siglip2_l4_shade.npy")]
    Z = []
    for X in parts:
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        Z.append(X / np.sqrt(X.shape[1]))
    from sklearn.decomposition import PCA
    pca = PCA(N_PC, random_state=0).fit(np.hstack(Z))
    return its, pca.transform(np.hstack(Z)), pca.explained_variance_ratio_


def scene_params(its):
    """Plan row + drawn scene of each generated map (None for real)."""
    from dspages.cli import PARTS
    from dspages.conditions import sample_preset
    from dspages.config import get_preset
    from dspages.geometry.scenes import sample_ends
    from dspages.geometry.surface import _swap
    P0 = get_preset("full")
    plan = [json.loads(l) for l in open(UV / "part6" / "plan.jsonl")]
    out = []
    for it in its:
        if it["real"]:
            out.append(None)
            continue
        i = int(Path(it["path"]).stem)
        s = dict(plan[i], combo=tuple(plan[i]["combo"]))
        g = sample_preset(P0, s).geometry
        sw, sh = (x * 1e-3 for x in g.sheet_mm)
        L, W = _swap(sw, sh, s["along_long"])
        b = np.radians(s["bend_dir"])
        lp, wg = L * np.cos(b) + W * abs(np.sin(b)), L * abs(np.sin(b)) + W * np.cos(b)
        sa, sb = sample_ends(np.random.default_rng([0, i, PARTS["geometry"]]), g.scene, lp, wg, s["two"], s["combo"])
        fold = [abs(np.degrees(t)) for _, t, _ in sa["folds"]]
        halves = sa["halves"] or []
        out.append(dict(
            kind=str(s["combo"][0]), gsm=float(s["gsm"]), two=bool(s["two"]), along_long=bool(s["along_long"]),
            bend_dir=float(s["bend_dir"]), shallow=bool(s["shallow"]), deep=["", "light", "medium", "heavy"].index(s["deep"]),
            crumple=bool(s["combo"][3]), n_folds=len(fold), fold_deg_sum=float(sum(fold)), fold_deg_max=float(max(fold, default=0)),
            n_valley=sum(t > 0 for _, t, _ in sa["folds"]),
            clamp_deg=float(np.degrees(sa["clamp"])) if sa["clamp"] is not None else 0.0,
            n_supports=len(sa["supports"]), support_h_mm=1e3 * max((h for _, h in sa["supports"]), default=0.0),
            book_h_mm=1e3 * max((H for _, _, H in sa["base"]), default=0.0),
            open_deg=float(np.degrees(np.mean([h["clamp"] for h in halves]))) if halves else 0.0,
            open_dh_deg=float(np.degrees(abs(halves[0]["clamp"] - halves[1]["clamp"]))) if halves else 0.0))
    return out


NUM = ("gsm", "two", "along_long", "bend_dir", "shallow", "deep", "crumple", "n_folds", "fold_deg_sum", "fold_deg_max",
       "n_valley", "clamp_deg", "n_supports", "support_h_mm", "book_h_mm", "open_deg", "open_dh_deg")
KINDS = ("False", "True", "book", "open")


def design(ps):
    return np.array([[float(p[k]) for k in NUM] + [float(p["kind"] == k) for k in KINDS] for p in ps]), \
        list(NUM) + [f"kind={k}" for k in KINDS]


def forward(X, Y):
    """XGBoost regressor per PC; 5-fold R2."""
    from sklearn.model_selection import cross_val_predict
    from xgboost import XGBRegressor
    models, r2 = [], []
    for j in range(Y.shape[1]):
        mk = lambda: XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, n_jobs=THREADS, random_state=0)
        pred = cross_val_predict(mk(), X, Y[:, j], cv=5)
        r2.append(float(1 - ((Y[:, j] - pred) ** 2).sum() / ((Y[:, j] - Y[:, j].mean()) ** 2).sum()))
        models.append(mk().fit(X, Y[:, j]))
    return models, r2


def plots(out, its, Y, ps, real, gap):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    T = TSNE(2, init="pca", random_state=0, perplexity=40).fit_transform(Y)
    gen = ~real
    cats = np.array([it.get("category", "") for it in its])
    fig, ax = plt.subplots(2, 3, figsize=(24, 16))
    a = ax[0, 0]
    a.scatter(*T[gen].T, s=3, c="#8a8a8a", label="generated")
    for j, c in enumerate(sorted(set(cats[real]))):
        a.scatter(*T[real & (cats == c)].T, s=4, color=plt.cm.tab10(j + 1), label=f"real {c}")
    a.set_title("generated and real (by category)")
    a = ax[0, 1]
    a.scatter(*T[gen].T, s=3, c="#dddddd")
    sc = a.scatter(*T[real].T, s=4, c=np.minimum(gap, 3), cmap="magma_r")
    fig.colorbar(sc, ax=a)
    a.set_title("real: distance to nearest generated / generated p95 NN distance (>1: gap)")
    for a, key, cm in ((ax[0, 2], "gsm", "viridis"), (ax[1, 0], "n_folds", "viridis"), (ax[1, 1], "deep", "viridis"),
                       (ax[1, 2], "kind", None)):
        a.scatter(*T[real].T, s=2, c="#f0d0d0")
        v = [p[key] for p in ps if p]
        if cm:
            sc = a.scatter(*T[gen].T, s=3, c=v, cmap=cm)
            fig.colorbar(sc, ax=a)
        else:
            for j, k in enumerate(KINDS):
                m = np.array([x == k for x in v])
                a.scatter(*T[gen][m].T, s=3, color=plt.cm.tab10(j),
                          label={"False": "free", "True": "clamp", "book": "book", "open": "open"}[k])
        a.set_title(f"generated by {key}" + (" (0 none, 1 light, 2 medium, 3 heavy)" if key == "deep" else ""))
    for a in ax.ravel():
        a.set_xticks([]), a.set_yticks([])
        if a.get_legend_handles_labels()[0]:
            a.legend(markerscale=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "space_tsne.png", dpi=90)
    plt.close(fig)
    np.save(out / "tsne.npy", T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
    its, Y, evr = space()
    real = np.array([it["real"] for it in its])
    gen = ~real
    ps = scene_params(its)
    (out / "params.jsonl").write_text("".join(json.dumps(p) + "\n" for p in ps))
    np.savez(out / "space.npz", pcs=Y, evr=evr, real=real)
    rep = dict(explained_variance=evr.round(4).tolist())

    # 1. coverage
    Yg, Yr = Y[gen], Y[real]
    nn = NearestNeighbors(n_neighbors=2).fit(Yg)
    d_gg = nn.kneighbors(Yg)[0][:, 1]
    d_rg, i_rg = nn.kneighbors(Yr, 1)
    ref = np.percentile(d_gg, 95)
    gap = d_rg[:, 0] / ref
    gap_all = np.zeros(len(its))
    gap_all[real] = gap
    cats = np.array([it["category"] for it in its if it["real"]])
    rep["coverage"] = dict(gen_nn_p95=float(ref), real_nn_to_gen_median=float(np.median(d_rg)),
                           real_gap_share=float((gap > 1).mean()),
                           by_category={c: dict(n=int((cats == c).sum()), gap_share=float((gap[cats == c] > 1).mean()),
                                                ratio_median=float(np.median(gap[cats == c]))) for c in sorted(set(cats))})
    km = KMeans(K_REAL, n_init=10, random_state=0).fit(Yr)
    gp = [p for p in ps if p]
    Xg, names = design(gp)
    nng = NearestNeighbors(n_neighbors=20).fit(Yg)
    clusters = []
    for c in range(K_REAL):
        m = km.labels_ == c
        d, idx = nng.kneighbors(km.cluster_centers_[c][None], 20)
        nb = [gp[k] for k in idx[0]]
        clusters.append(dict(cluster=c, n=int(m.sum()), categories={k: int((cats[m] == k).sum()) for k in sorted(set(cats[m]))},
                             gap_share=float((gap[m] > 1).mean()), centre_to_gen=float(d[0, 0] / ref),
                             neighbours_20={k: (float(np.mean([p[k] for p in nb])) if k != "kind" else
                                                {x: sum(p["kind"] == x for p in nb) for x in KINDS}) for k in ("kind",) + NUM}))
    rep["real_clusters"] = clusters

    # 2. parameters -> position
    models, r2 = forward(Xg, Yg)
    imp = np.zeros(len(names))
    for mdl, w in zip(models, evr):
        g = mdl.get_booster().get_score(importance_type="total_gain")
        v = np.array([g.get(f"f{j}", 0.0) for j in range(len(names))])
        imp += w * v / max(v.sum(), 1e-12)
    imp /= imp.sum()
    rep["forward_r2_per_pc"] = [round(x, 3) for x in r2]
    rep["forward_r2_weighted"] = float(np.dot(r2, evr) / evr.sum())
    rep["param_importance"] = dict(sorted(((n, round(float(v), 4)) for n, v in zip(names, imp)), key=lambda t: -t[1]))

    # 3. grammage: predicted shift over the gsm range, per kind, in units of the generated spread (rms distance to centre)
    spread = float(np.sqrt(((Yg - Yg.mean(0)) ** 2).sum(1).mean()))
    gi = names.index("gsm")
    lo, hi = np.percentile(Xg[:, gi], (2, 98))
    gsm_eff = {}
    for k in KINDS:
        m = Xg[:, names.index(f"kind={k}")] == 1
        A, B = Xg[m].copy(), Xg[m].copy()
        A[:, gi], B[:, gi] = lo, hi
        dA = np.stack([mdl.predict(A) for mdl in models], 1)
        dB = np.stack([mdl.predict(B) for mdl in models], 1)
        gsm_eff[{"False": "free", "True": "clamp", "book": "book", "open": "open"}[k]] = dict(
            n=int(m.sum()), shift_over_spread=float(np.linalg.norm(dB - dA, axis=1).mean() / spread))
    rep["gsm"] = dict(range=[float(lo), float(hi)], corr_with_pcs=[round(float(np.corrcoef(Xg[:, gi], Yg[:, j])[0, 1]), 3)
                                                                   for j in range(N_PC)], by_kind=gsm_eff)
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    plots(out, its, Y, ps, real, gap_all[real])
    print(json.dumps({k: rep[k] for k in ("coverage", "forward_r2_weighted", "param_importance", "gsm")}, indent=1))


if __name__ == "__main__":
    main()
