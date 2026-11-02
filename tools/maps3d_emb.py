"""Vectors of 3D page maps (grid3d (3, 360, 256) X, Y, Z [mm]): which tell bends, folds and creases apart, and the real vs
generated distribution.

Each map is put in the page frame (realmaps.frame: Z along the mean page normal, X along the columns, lowest node at Z = 0).
Images (224 x 224, gray): `height` Z / DZ_MAX (DZ_MAX the largest frame Z over all maps: 10,000 real, 23,198 generated in
uv_ds_2.0.0 part5 + part6); `shade` Lambert shading of the surface normals under a light at LIGHT (azimuth, elevation).
Vectors: SigLIP 2 of each image (pooled output, mean patch tokens of layers 4 / 8); `feat` engineered surface features
(blocks(), 517); `feat+<best>` feat joined with the best SigLIP vector.
Scores (5-fold stratified XGBoost, balanced accuracy): generated maps by the plan fields folds (combo[2]), shallow, deep,
and by the folds as drawn (n_folds() > 0);
real maps by category; real vs generated.

python tools/maps3d_emb.py --out DIR [--gen 3000]
DIR/items.jsonl, DIR/emb_<name>.npy, DIR/scores.json, DIR/map_<name>.png (t-SNE: real vs generated, real by category)
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage
from scipy.fft import dctn

from dspages.prep.realmaps import frame

UV = Path("/run/media/p/D/ML_DS/UVTM/uv_ds_2.0.0")
REAL = Path("/run/media/p/D/ML_DS/UVTM/GridPages/maps3d_256x360")
MODEL = "google/siglip2-base-patch16-224"
LAYERS = (4, 8)
DZ_MAX = 73.13                              # mm: largest frame Z over all maps (real m0056_02; generated max 72.25)
LIGHT = (135.0, 30.0)                       # deg: azimuth from +X, elevation above the page plane
STEP = (210.0 / 255, 297.0 / 359)           # mm: nominal node steps across, down
SCALES = (1.0, 6.0)                         # nodes: Gaussian smoothing of the curvature (creases, bends)


def n_folds(i, spec, seed=0):
    """Folds of sample i as drawn (its scene from its geometry stream, as cli._geometry: a plan with folds can get none,
    those near a support line or past the angle budget are dropped)."""
    from dspages.cli import PARTS
    from dspages.conditions import sample_preset
    from dspages.config import get_preset
    from dspages.geometry.scenes import sample_ends
    from dspages.geometry.surface import _swap
    spec = dict(spec, combo=tuple(spec["combo"]))
    g = sample_preset(get_preset("full"), spec).geometry
    sw, sh = (x * 1e-3 for x in g.sheet_mm)
    L, W = _swap(sw, sh, spec["along_long"])
    b = np.radians(spec["bend_dir"])
    lp, wg = L * np.cos(b) + W * abs(np.sin(b)), L * abs(np.sin(b)) + W * np.cos(b)
    sa, _ = sample_ends(np.random.default_rng([seed, i, PARTS["geometry"]]), g.scene, lp, wg, spec["two"], spec["combo"])
    return len(sa["folds"])


def items(n_gen, seed=0):
    rnd = random.Random(seed)
    plan = [json.loads(l) for l in open(UV / "part6" / "plan.jsonl")]
    gen = sorted(UV.glob("part[56]/*/*.npz"))
    rnd.shuffle(gen)
    out = []
    for f in gen[:n_gen]:
        p = plan[int(f.stem)]
        out.append(dict(path=str(f), real=False, folds=bool(p["combo"][2]), n_folds=n_folds(int(f.stem), p),
                        shallow=bool(p["shallow"]), deep=p["deep"] or "none"))
    for s in map(json.loads, open(REAL / "sources.jsonl")):
        out.append(dict(path=str(REAL / "sources" / f"s{s['source']:04d}.npz"), real=True, category=s["category"],
                        source=s["source"]))
    return out


def load(path):
    """(360, 256, 3) [mm] in the page frame, lowest node at Z = 0."""
    return frame(np.moveaxis(np.load(path)["grid3d"].astype(np.float64), 0, -1))


def forms(G, sigma):
    """Normals and curvatures (k1 >= k2, Gaussian K, mean H [1/mm]) of the grid surface G smoothed by sigma nodes."""
    if sigma:
        G = np.stack([ndimage.gaussian_filter(G[..., k], sigma, mode="nearest") for k in range(3)], -1)
    du, dv = STEP
    Gu, Gv = np.gradient(G, du, axis=1), np.gradient(G, dv, axis=0)
    Guu, Guv = np.gradient(Gu, du, axis=1), np.gradient(Gu, dv, axis=0)
    Gvv = np.gradient(Gv, dv, axis=0)
    n = np.cross(Gu, -Gv)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    E, F, Gg = (Gu * Gu).sum(-1), (Gu * Gv).sum(-1), (Gv * Gv).sum(-1)
    L, M, N = (Guu * n).sum(-1), (Guv * n).sum(-1), (Gvv * n).sum(-1)
    det = np.maximum(E * Gg - F ** 2, 1e-12)
    K = (L * N - M ** 2) / det
    H = (E * N - 2 * F * M + Gg * L) / (2 * det)
    r = np.sqrt(np.maximum(H ** 2 - K, 0))
    return n, H + r, H - r, K, H


def shade(G):
    n = forms(G, 0)[0]
    a, e = np.radians(LIGHT)
    return np.clip(n @ np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)]), 0, 1)


def images(G, size=224):
    def gray(x):
        x = cv2.resize(x.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
        return np.repeat((np.clip(x, 0, 1) * 255).astype(np.uint8)[..., None], 3, -1)
    return gray(G[..., 2] / DZ_MAX), gray(shade(G))


def features(G):
    """Engineered features of a page-frame grid G [mm]: height (level, spread, contact with the table), edges and
    corners (lifted sides), tilt of the normals, curvature at a crease and a bend scale (curvedness and shape index
    histograms after Koenderink, mean |H|, |K| (developability), share of crease-like nodes), anisotropy of the slopes
    (structure tensor), radial spectrum of Z, low DCT of Z (coarse shape), row / column profiles (bend axis)."""
    Z = G[..., 2]
    f = [Z.max(), Z.mean(), Z.std(), *np.percentile(Z, (10, 50, 90)), (Z < 1).mean(), (Z < 5).mean()]
    f += [*np.sort([Z[0].mean(), Z[-1].mean(), Z[:, 0].mean(), Z[:, -1].mean()]),
          *np.sort([Z[0, 0], Z[0, -1], Z[-1, 0], Z[-1, -1]]), Z[140:220, 100:156].mean()]
    for s in SCALES:
        n, k1, k2, K, H = forms(G, s)
        tilt = np.degrees(np.arccos(np.clip(n[..., 2], -1, 1)))
        c = np.sqrt((k1 ** 2 + k2 ** 2) / 2)
        si = 2 / np.pi * np.arctan2(k1 + k2, k1 - k2)
        f += [tilt.mean(), np.percentile(tilt, 90), tilt.max(), *np.histogram(tilt, (0, 5, 10, 20, 30, 45, 90))[0] / tilt.size]
        f += [*np.histogram(np.log10(c + 1e-6), 8, (-5, 0))[0] / c.size,
              *np.histogram(si, 9, (-1, 1), weights=c)[0] / max(c.sum(), 1e-12),
              np.abs(H).mean(), np.abs(K).mean(), np.percentile(c, 99), c.max(),
              *[(c > t).mean() for t in (0.01, 0.03, 0.1, 0.3)]]
    gy, gx = np.gradient(ndimage.gaussian_filter(Z, 2.0))
    J = np.array([(gx * gx).mean(), (gy * gy).mean(), (gx * gy).mean()])
    tr, d = J[0] + J[1], np.sqrt((J[0] - J[1]) ** 2 + 4 * J[2] ** 2)
    f += [d / max(tr, 1e-12), abs(np.cos(np.arctan2(2 * J[2], J[0] - J[1])))]
    P = np.abs(np.fft.rfft2(Z - Z.mean())) ** 2
    fy, fx = np.meshgrid(np.fft.fftfreq(Z.shape[0]), np.fft.rfftfreq(Z.shape[1]), indexing="ij")
    r, band = np.hypot(fy, fx), np.geomspace(0.003, 0.5, 11)
    f += [np.log10(P[(r >= a) & (r < b)].mean() + 1e-9) for a, b in zip(band[:-1], band[1:])]
    f += list((dctn(cv2.resize(Z.astype(np.float32), (32, 32), interpolation=cv2.INTER_AREA), norm="ortho")[:6, :6]
               / DZ_MAX).ravel())
    f += [Z.mean(0).std(), Z.mean(1).std(), np.abs(np.diff(Z.mean(0))).sum(), np.abs(np.diff(Z.mean(1))).sum()]
    return np.array(f, np.float32)


def ms_stats(a):
    a = np.abs(a)
    return [a.mean(), a.std(), *np.percentile(a, (50, 90, 99, 99.9)), a.max()]

def radon_feats(img, n_ang=36):
    """Line integrals of img over n_ang directions: long straight ridges (folds) give high narrow peaks."""
    img = cv2.resize(img.astype(np.float32), (90, 128), interpolation=cv2.INTER_AREA)
    pad = int(np.ceil(np.hypot(*img.shape))) + 2
    P = np.zeros((pad, pad), np.float32); y0, x0 = (pad - 128) // 2, (pad - 90) // 2
    P[y0:y0 + 128, x0:x0 + 90] = img; ones = np.zeros_like(P); ones[y0:y0 + 128, x0:x0 + 90] = 1
    peaks = []
    for a in np.linspace(0, 180, n_ang, endpoint=False):
        R = cv2.getRotationMatrix2D((pad / 2, pad / 2), a, 1)
        s = cv2.warpAffine(P, R, (pad, pad)).sum(0); n = cv2.warpAffine(ones, R, (pad, pad)).sum(0)
        ok = n > 20
        prof = s[ok] / n[ok]
        peaks.append(prof)
    mx = np.array([p.max() for p in peaks]); md = np.array([np.median(p) for p in peaks])
    allp = np.concatenate(peaks)
    top = np.sort(allp)[::-1]
    base = img.mean() + 1e-9
    return [mx.max() / base, (mx / (md + 1e-9)).max(), mx.std() / base, top[:5].mean() / base,
            np.percentile(allp, 99) / base, (mx.max() - np.median(mx)) / base]

def blocks(G):
    """Feature blocks of a page-frame grid G: base (features()), dog (band-pass residuals of Z by scale), normgrad
    (gradient of the normal field and slopes by scale), ridge (Hessian line strength, ridge components and their extent,
    Radon peaks of long straight ridges), gauss_img (normal histogram), curv (principal curvatures, K, shape index by
    scale)."""
    Z = G[..., 2]
    out = {"base": features(G)}
    # B: band-pass (DoG) residuals of Z: crease roughness by scale
    f = []
    sm = [ndimage.gaussian_filter(Z, s) for s in (0.5, 1, 2, 4, 8, 16, 32)]
    for a, b in zip(sm[:-1], sm[1:]):
        r = a - b
        f += ms_stats(r) + [((np.abs(r) > t).mean()) for t in (0.02, 0.05, 0.1, 0.2, 0.5)] + [float(((r - r.mean()) ** 4).mean() / (r.var() ** 2 + 1e-12))]
    out["dog"] = np.array(f, np.float32)
    # C: normal-field gradient |dn| (bending) at scales; slope (gradient magnitude) stats
    f = []
    for s in (0, 1, 2, 4, 8):
        n = forms(G, s)[0]
        dn = np.sqrt(sum(np.gradient(n[..., k], axis=a) ** 2 for k in range(3) for a in (0, 1)))
        f += ms_stats(dn) + [(dn > t).mean() for t in (0.01, 0.03, 0.1, 0.3)]
    for s in (1, 4, 16):
        gy, gx = np.gradient(ndimage.gaussian_filter(Z, s))
        f += ms_stats(np.hypot(gx, gy))
    out["normgrad"] = np.array(f, np.float32)
    # D: Hessian ridge / valley (Frangi-like) at scales, ridge length & straightness (connected components)
    f = []
    for s in (1, 2, 4, 8):
        Hrr, Hrc, Hcc = [s * s * ndimage.gaussian_filter(Z, s, order=o) for o in ((2, 0), (1, 1), (0, 2))]
        tmp = np.sqrt((Hrr - Hcc) ** 2 + 4 * Hrc ** 2)
        l1, l2 = 0.5 * (Hrr + Hcc + tmp), 0.5 * (Hrr + Hcc - tmp)          # l1 >= l2
        lam = np.where(np.abs(l1) > np.abs(l2), l1, l2); other = np.where(np.abs(l1) > np.abs(l2), l2, l1)
        line = np.abs(lam) * np.exp(-(other / (lam + 1e-12)) ** 2 * 4)   # strong one-direction curvature
        f += ms_stats(line)
        thr = max(np.percentile(line, 95), 1e-6)
        for t in (0.02, 0.1):
            mask = line > t
            lab, k = ndimage.label(mask)
            if k:
                sizes = np.bincount(lab.ravel())[1:]
                objs = ndimage.find_objects(lab)
                ext = np.array([max(o[0].stop - o[0].start, o[1].stop - o[1].start) for o in objs])
                big = sizes >= 20
                f += [k, mask.mean(), sizes.max(), ext.max(), (ext > 100).sum(), (ext > 200).sum(),
                      (ext[big] ** 2 / sizes[big]).max() if big.any() else 0, big.sum()]
            else:
                f += [0] * 8
        f += radon_feats(line)
    out["ridge"] = np.array(f, np.float32)
    # E: Gaussian image (normal histogram): piecewise-flat folded pages give few clusters
    f = []
    for s in (1, 4):
        n = forms(G, s)[0]
        h = np.histogram2d(n[..., 0].ravel(), n[..., 1].ravel(), 24, ((-1, 1), (-1, 1)))[0] / n[..., 0].size
        p = h[h > 0]
        C = np.cov(np.stack([n[..., 0].ravel(), n[..., 1].ravel()]))
        ev = np.linalg.eigvalsh(C)
        f += [-(p * np.log(p)).sum(), (h > 0.01).sum(), (h > 0.05).sum(), h.max(), *ev, ev[0] / (ev[1] + 1e-12)]
    out["gauss_img"] = np.array(f, np.float32)
    # F: principal curvature magnitudes / shape index at more scales
    f = []
    for s in (0.5, 2, 4, 12):
        _, k1, k2, K, H = forms(G, s)
        c = np.sqrt((k1 ** 2 + k2 ** 2) / 2); kk = np.maximum(np.abs(k1), np.abs(k2))
        si = 2 / np.pi * np.arctan2(k1 + k2, k1 - k2)
        f += ms_stats(kk) + ms_stats(K) + [np.abs(np.minimum(np.abs(k1), np.abs(k2))).mean() / (kk.mean() + 1e-12)]
        f += list(np.histogram(si, 9, (-1, 1), weights=c)[0] / max(c.sum(), 1e-12))
    out["curv"] = np.array(f, np.float32)
    return out


def embed(its, bs=32):
    from transformers import AutoModel
    torch.set_num_threads(6)
    m = AutoModel.from_pretrained(MODEL).vision_model.eval()
    out = {f"siglip2_{k}_{im}": [] for im in ("height", "shade") for k in ("final", *[f"l{n}" for n in LAYERS])}
    out["feat"] = []
    for i in range(0, len(its), bs):
        Gs = [load(it["path"]) for it in its[i:i + bs]]
        ims = list(zip(*(images(G) for G in Gs)))
        for name, batch in zip(("height", "shade"), ims):
            x = (torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float() / 255 - 0.5) / 0.5
            with torch.no_grad():
                o = m(pixel_values=x, output_hidden_states=True)
            out[f"siglip2_final_{name}"].append(o.pooler_output.numpy())
            for n in LAYERS:
                out[f"siglip2_l{n}_{name}"].append(o.hidden_states[n].mean(1).numpy())
        out["feat"].append(np.stack([np.concatenate(list(blocks(G).values())) for G in Gs]))
        print(f"{i + len(Gs)}/{len(its)}", flush=True)
    return {k: np.concatenate(v) for k, v in out.items()}


def xgb_score(X, y, folds=5, seed=0):
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier
    y = np.unique(np.asarray([str(v) for v in y]), return_inverse=True)[1]
    pred = np.empty_like(y)
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.5,
                          tree_method="hist", n_jobs=6, random_state=seed)
        m.fit(X[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
        pred[te] = m.predict(X[te])
    return float(balanced_accuracy_score(y, pred))


def scores(its, embs):
    gen = np.array([not it["real"] for it in its])
    g, r = np.flatnonzero(gen), np.flatnonzero(~gen)
    res = {}
    for name, E in embs.items():
        s = {t: xgb_score(E[g], [its[i][t] for i in g]) for t in ("folds", "shallow", "deep")}
        s["folds_drawn"] = xgb_score(E[g], [its[i]["n_folds"] > 0 for i in g])
        s["real_category"] = xgb_score(E[r], [its[i]["category"] for i in r])
        s["real_vs_gen"] = xgb_score(E, gen.astype(int))
        res[name] = s
        print(name, s, flush=True)
    return res


def _norm(E):
    E = (E - E.mean(0)) / (E.std(0) + 1e-6)
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def plot(its, E, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    Y = TSNE(2, init="pca", random_state=0, perplexity=40).fit_transform(
        PCA(min(50, E.shape[1]), random_state=0).fit_transform(_norm(E)))
    real = np.array([it["real"] for it in its])
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].scatter(*Y[~real].T, s=3, c="#8a8a8a", label=f"generated ({(~real).sum()})")
    ax[0].scatter(*Y[real].T, s=3, c="#d6402b", label=f"real ({real.sum()})")
    ax[0].set_title("real vs generated")
    lab = np.array([it["category"] for it in its if it["real"]])
    ax[1].scatter(*Y[~real].T, s=2, c="#e4e4e4")
    for j, v in enumerate(sorted(set(lab))):
        ax[1].scatter(*Y[real][lab == v].T, s=3, color=plt.cm.tab10(j), label=f"{v} ({(lab == v).sum()})")
    ax[1].set_title("real by category")
    for a in ax:
        a.legend(markerscale=4, fontsize=8)
        a.set_xticks([]), a.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen", type=int, default=3000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    its = items(a.gen)
    (out / "items.jsonl").write_text("".join(json.dumps(it) + "\n" for it in its))
    embs = embed(its)
    res = scores(its, embs)
    best = max((k for k in res if k.startswith("siglip2")), key=lambda k: np.mean(list(res[k].values())[:4]))
    embs[f"feat+{best}"] = np.hstack([embs["feat"], embs[best]])
    res.update(scores(its, {f"feat+{best}": embs[f"feat+{best}"]}))
    (out / "scores.json").write_text(json.dumps(res, indent=1))
    for k, v in embs.items():
        np.save(out / f"emb_{k}.npy", v)
        plot(its, v, out / f"map_{k}.png", k)


if __name__ == "__main__":
    main()
