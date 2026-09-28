"""Distributions, limits and errors of a generated run.

python tools/analyze.py RUN_DIR [--preset full]
Writes RUN_DIR/report/*.png and RUN_DIR/report/summary.json (counts, quantiles, samples outside the limits).
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dspages import io  # noqa: E402
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
    return [io.load(p)[1] for p in sorted(d.glob("*.npz"))] if d.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--preset", default="full")
    a = ap.parse_args()
    run, P = Path(a.run), get_preset(a.preset)
    out = run / "report"
    out.mkdir(exist_ok=True)
    plan = [json.loads(line) for line in open(run / "plan.jsonl")]
    lay, geo, ren = _metas(run / "layouts"), _metas(run / "geometry"), _metas(run / "samples")
    cam = P.render.camera
    summary = dict(n_plan=len(plan), n_layouts=len(lay), n_geometry=len(geo), n_samples=len(ren))

    _grid(9, out, "plan", lambda ax: [
        _bars(ax[0], Counter(str(tuple(s["combo"])) for s in plan), "combo (clamp, supports, folds, crumple)"),
        _bars(ax[1], Counter(s["gsm"] for s in plan), "paper g/m2"),
        _bars(ax[2], Counter(s["deep"] or "-" for s in plan), "deep crease level"),
        _hist(ax[3], [s["bend_dir"] for s in plan], "bend direction [deg]"),
        _bars(ax[4], Counter(f"{b[0]:.0f}-{b[1]:.0f}" for s in plan for b in [s["dist_bin"]]), "camera distance bin [mm]"),
        _bars(ax[5], Counter(f"{b[0]:.0f}-{b[1]:.0f}" for s in plan for b in [s["tilt_bin"]]), "camera tilt bin [deg]"),
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
            _hist(ax[5], [m["page_frac"] for m in ren], "page share of the photo"),
            _hist(ax[6], [x["fill"] for x in v], "page fill of the frame"),
            _bars(ax[7], Counter(e for m in ren for e in m["effects"]), "camera effects applied"),
            _bars(ax[8], Counter(("sharp" if m["light"]["sharp"] else "area") if m["light"] else "none" for m in ren),
                  "light")])
        summary["over_incidence"] = [i for i, x in enumerate(v) if x["max_incidence_deg"] > cam.incidence_deg]
        summary["px_per_mm_q"] = np.quantile([x["min_px_per_mm"] for x in v], [0, 0.05, 0.5]).round(2).tolist()

    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
