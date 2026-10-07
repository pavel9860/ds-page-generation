"""Distributions, limits and errors of a generated run.

python tools/analyze.py RUN_DIR [--preset full] [--manifest M] [--n_check 300] [--workers 6]
Plan and part metas: distributions and samples outside the limits. With the manifest: the page conditions of the
raster pages used (fill, picture share, skew) and the repetition of pages, restyled pages and documents. On n_check
samples: UV maps (the warped image mapped back to the flat page, tools/uv_check.py).
Writes RUN_DIR/report/*.png and RUN_DIR/report/summary.json.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dspages.config import get_preset  # noqa: E402

INK, MUTED, BAR = "#1f2328", "#6e7781", "#3a6fd8"


def _style(ax, title):
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.grid(True, color="#d0d7de", lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7, colors=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)


def _hist(ax, v, title, limit=None):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    ax.hist(v, bins=30, color=BAR, edgecolor="white", lw=0.5)
    if limit is not None:
        ax.axvline(limit, color=INK, lw=1, ls="--")
    _style(ax, f"{title}  (n={len(v)})")


def _bars(ax, counter, title, top=15):
    items = counter.most_common(top)[::-1]
    ax.barh([str(k) for k, _ in items], [c for _, c in items], color=BAR, height=0.7)
    _style(ax, title)


def _grid(n, out, name, draw):
    rows = (n + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(13, 3.2 * rows), squeeze=False)
    draw(axes.ravel())
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=110)
    plt.close(fig)


def _metas(d):
    """Metas only (np.load reads an .npz member on access)."""
    out = []
    for p in sorted(d.glob("*.npz")) if d.exists() else []:
        with np.load(p) as f:
            out.append(json.loads(str(f["meta"])))
    return out


def _uv(path):
    import uv_check
    return uv_check.check(path)


def repetition(plan, man):
    """Uses per raster page, per book page and per source document."""
    raster = Counter(s["entry"] for s in plan if man[s["entry"]]["kind"] == "raster")
    snip = Counter(s["entry"] for s in plan if man[s["entry"]]["kind"] == "refont")
    docs = Counter(man[s["entry"]]["source_path"] for s in plan)
    return raster, snip, docs


def conditions(plan, man, c):
    """Raster pages used that break the manifest's page conditions."""
    used = [man[i] for i in sorted({s["entry"] for s in plan}) if man[i]["kind"] == "raster"]
    return dict(raster_pages_used=len(used),
                fill_below_min=sum(e.get("fill", 1) < c.min_fill for e in used),
                pictures_over_max=sum(e.get("picture_share", 0) > c.max_image for e in used),
                skew_ge_3deg=sum(abs(e.get("deskew_deg", 0)) >= 3 for e in used)), used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--preset", default="full")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--n_check", type=int, default=300)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    run, P = Path(a.run), get_preset(a.preset)
    out = run / "report"
    out.mkdir(exist_ok=True)
    plan = [json.loads(line) for line in open(run / "plan.jsonl")]
    lay, geo = _metas(run / "layouts"), _metas(run / "geometry")
    ren = [json.loads(line) for line in open(run / "meta.jsonl")] if (run / "meta.jsonl").exists() else []
    lay = lay or [m["layout"] for m in ren]
    cam = P.render.camera
    summary = dict(n_plan=len(plan), n_layouts=len(lay), n_geometry=len(geo), n_samples=len(ren))

    _grid(9, out, "plan", lambda ax: [
        _bars(ax[0], Counter(str(tuple(s["combo"])) for s in plan), "combo (clamp, supports, folds, crumple)"),
        _bars(ax[1], Counter(s["gsm"] for s in plan), "paper g/m2"),
        _bars(ax[2], Counter(s["deep"] or "-" for s in plan), "deep crease level"),
        _hist(ax[3], [s["bend_dir"] for s in plan], "bend direction [deg]"),
        _bars(ax[6], Counter(("sharp" if s["sharp"] else "area") for s in plan), "light"),
        _bars(ax[7], Counter(f"flat={s['flat']} two={s['two']}" for s in plan), "near-flat / two profiles"),
        _bars(ax[8], Counter(f"{b[0]:.2f}-{b[1]:.2f}" for s in plan for b in [s["margin_bin"]]), "margin bin")])

    if lay:
        _grid(6, out, "layouts", lambda ax: [
            _bars(ax[0], Counter(m["kind"] for m in lay), "source kind"),
            _bars(ax[1], Counter(m["language"] for m in lay), "language"),
            _bars(ax[2], Counter(e for m in lay for e in m["effects"]), "paper effects applied"),
            _hist(ax[3], [m.get("fill") for m in lay], "window fill", P.layout.min_fill),
            _hist(ax[4], [m["margin"] for m in lay], "margin (fraction of sheet)"),
            _hist(ax[5], [m.get("font_pt") for m in lay], "font size [pt]")])
        summary["layout_fill_below_min"] = [i for i, m in enumerate(lay) if m.get("fill", 1) < P.layout.min_fill]

    if geo:
        _grid(6, out, "geometry", lambda ax: [
            _bars(ax[0], Counter(m["label"] for m in geo), "scene"),
            _hist(ax[1], [100 * m["box_strain"] for m in geo], "membrane strain [%]", 0.1),
            _bars(ax[2], Counter(m["paper"]["gsm"] for m in geo), "paper g/m2"),
            _hist(ax[3], [m["n_creases"] for m in geo], "creases per page"),
            _hist(ax[4], [len(m["scene_a"]["folds"]) for m in geo], "folds per page"),
            _hist(ax[5], [h * 1e3 for m in geo for _, h in m["scene_a"]["supports"]], "support height [mm]")])
        summary["strain_over_0.1pct"] = [i for i, m in enumerate(geo) if m["box_strain"] > 0.001 and not m["cond"]["deep"]]

    if ren:
        v = [m["view"] for m in ren]
        _grid(9, out, "render", lambda ax: [
            _hist(ax[0], [x["dist_mm"] for x in v], "camera distance [mm]"),
            _hist(ax[1], [x["tilt_deg"] for x in v], "camera tilt [deg]"),
            _hist(ax[2], [x["fov_deg"] for x in v], "vertical field of view [deg]"),
            _hist(ax[3], [x["max_incidence_deg"] for x in v], "max view incidence [deg]", cam.incidence_deg),
            _hist(ax[4], [x["min_px_per_mm"] for x in v], "min image scale on the page [px/mm]"),
            _hist(ax[5], [x["fill"] for x in v], "camera target: page bbox / frame, tighter side"),
            _hist(ax[6], [m["page_frac"] for m in ren], "page area / frame area (page_frac)"),
            _bars(ax[7], Counter(e for m in ren for e in m["effects"]), "camera effects applied"),
            _bars(ax[8], Counter(("sharp" if m["light"]["sharp"] else "area") if m["light"] else "none" for m in ren),
                  "light")])
        summary["over_incidence"] = [i for i, x in enumerate(v) if x["max_incidence_deg"] > cam.incidence_deg]
        summary["px_per_mm_q"] = np.quantile([x["min_px_per_mm"] for x in v], [0, 0.05, 0.5]).round(2).tolist()

    if a.manifest:
        from dspages.layout.sources import load_manifest
        man = load_manifest(a.manifest)
        cond, used = conditions(plan, man, P.layout)
        raster, snip, docs = repetition(plan, man)
        summary.update(cond, raster_page_max_uses=max(raster.values(), default=0),
                       raster_pages_repeated=sum(v > 1 for v in raster.values()),
                       restyled_pages_repeated=sum(v > 1 for v in snip.values()),
                       docs_used=len(docs), doc_uses_q=np.quantile(list(docs.values()), [0.5, 0.9, 0.99, 1]).tolist())
        _grid(6, out, "pages", lambda ax: [
            _hist(ax[0], [e.get("fill") for e in used], "fill of the raster pages used", P.layout.min_fill),
            _hist(ax[1], [e.get("picture_share") for e in used], "picture share of the raster pages used",
                  P.layout.max_image),
            _hist(ax[2], [e.get("deskew_deg", 0) for e in used], "stored deskew [deg]", 3.0),
            _bars(ax[3], Counter(min(v, 5) for v in raster.values()), "uses per raster page (5 = 5+)"),
            _bars(ax[4], Counter(min(v, 5) for v in snip.values()), "uses per restyled page (5 = 5+)"),
            _hist(ax[5], list(docs.values()), "samples per source document")])

    paths = sorted((run / "samples").glob("*.npz"))
    if paths and a.n_check:
        from concurrent.futures import ProcessPoolExecutor
        pick = [paths[i] for i in np.random.default_rng(0).choice(len(paths), min(a.n_check, len(paths)), False)]
        with ProcessPoolExecutor(a.workers) as pool:
            uv = list(pool.map(_uv, pick))
        shift = [r["shift_rms_px"] for r in uv if r["shift_rms_px"] is not None]
        summary.update(n_checked=len(pick),
                       uv_seen_q=np.quantile([r["seen"] for r in uv], [0, 0.5]).round(3).tolist(),
                       uv_shift_rms_px_q=np.quantile(shift, [0.5, 0.9, 1]).round(3).tolist(),
                       uv_rmse_gain_q=np.quantile([r["rmse_gain"] for r in uv], [0.5, 0.9]).round(1).tolist())
        _grid(2, out, "checks", lambda ax: [
            _hist(ax[0], shift, "UV back map: RMS tile shift [px]"),
            _hist(ax[1], [r["seen"] for r in uv], "UV back map: share of the page seen")])

    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
