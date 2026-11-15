"""Per-sample visual check of a run: flat | warped with UV arrows and the flat's outline (red: cut by the frame) |
unwarped | UV map | height map | shaded surface; files of cut samples end in _cut. Filtered-out manifest pages (a manifest's .rejected.jsonl) as page thumbnails, one folder per reason.

python tools/viz.py RUN_DIR [--manifest MANIFEST] [--workers 6]
Writes RUN_DIR_viz/samples/<i>.jpg and, with --manifest, RUN_DIR_viz/filtered_out/<reason>/<name>_p<page>.jpg.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dspages import io  # noqa: E402

ARROWS = 14  # arrows per side of the UV grid


def back_map(warped, grid2d, page_hw):
    """Flat page (page_hw) sampled from the warped image through grid2d (photo x, y in [-1, 1], align_corners=True,
    of each page grid node), as grid_sample would."""
    ph, pw = page_hw
    g = cv2.resize(np.moveaxis(grid2d, 0, -1), (pw, ph), interpolation=cv2.INTER_LINEAR)
    h, w = warped.shape
    return cv2.remap(
        warped.astype(np.float32),
        (g[..., 0] + 1) * 0.5 * (w - 1),
        (g[..., 1] + 1) * 0.5 * (h - 1),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def shaded(map3d):
    n = np.cross(np.gradient(map3d, axis=1), np.gradient(map3d, axis=0))
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    light = np.array([-0.4, -0.5, 0.77])
    return np.clip(np.abs(n @ light), 0, 1)


def uv_image(grid2d):
    """grid2d as colour: photo x red, photo y green (each in [-1, 1] -> 0..1)."""
    g = np.clip((np.moveaxis(grid2d, 0, -1) + 1) / 2, 0, 1)
    return np.dstack([g[..., 0], g[..., 1], np.zeros(g.shape[:2])])


def sample_figure(path, out, meta):
    a, _ = io.load(path)
    flat, warped, g2, map3d = a["flat"], a["warped"], a["grid2d"].astype(np.float32), np.moveaxis(a["grid3d"], 0, -1)
    cut = bool(meta.get("view", {}).get("cut"))
    fig, ax = plt.subplots(2, 3, figsize=(15, 13))
    ax = ax.ravel()
    ax[0].imshow(flat, cmap="gray", vmin=0, vmax=255)
    ax[0].set_title(
        f"flat {flat.shape[1]}x{flat.shape[0]}, part of the sheet u {meta['flat_part'][0]:.3f}-{meta['flat_part'][2]:.3f} "
        f"v {meta['flat_part'][1]:.3f}-{meta['flat_part'][3]:.3f}"
        if "flat_part" in meta
        else "flat",
        fontsize=9,
    )
    ax[1].imshow(warped, cmap="gray", vmin=0, vmax=255)
    h, w = g2.shape[1:]
    ys, xs = np.mgrid[h // (2 * ARROWS) : h : h // ARROWS, w // (2 * ARROWS) : w : w // ARROWS]
    ph, pw = flat.shape
    sx, sy = (warped.shape[1] - 1) / 2, (warped.shape[0] - 1) / 2
    px, py = (g2[0, ys, xs] + 1) * sx, (g2[1, ys, xs] + 1) * sy
    ax[1].quiver(
        px,
        py,
        xs * (pw - 1) / (w - 1) - px,
        ys * (ph - 1) / (h - 1) - py,
        color="#e4572e",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.003,
    )
    edge = np.concatenate([g2[:, 0, :], g2[:, :, -1], g2[:, -1, ::-1], g2[:, ::-1, 0]], 1)
    ax[1].plot((edge[0] + 1) * sx, (edge[1] + 1) * sy, color="#d00000" if cut else "#2e86de", lw=1.5)
    ax[1].set_xlim(0, warped.shape[1] - 1)
    ax[1].set_ylim(warped.shape[0] - 1, 0)
    ax[1].set_title(
        f"warped {warped.shape[1]}x{warped.shape[0]}; outline: the flat's part"
        + (" (CUT by the frame)" if cut else ""),
        fontsize=9,
        color="#d00000" if cut else "black",
    )
    back = back_map(warped, g2, flat.shape)
    ax[2].imshow(back, cmap="gray", vmin=0, vmax=255)
    ax[2].set_title("unwarped: warped mapped back through grid2d", fontsize=9)
    ax[3].imshow(uv_image(g2))
    ax[3].set_title(f"UV map grid2d {w}x{h} (red: photo x, green: photo y)", fontsize=9)
    Z = map3d[..., 2]
    im = ax[4].imshow(Z, cmap="viridis")
    fig.colorbar(im, ax=ax[4], fraction=0.046, label="Z mm")
    ax[4].set_title(f"3D map grid3d {w}x{h}: height", fontsize=9)
    ax[5].imshow(shaded(map3d), cmap="gray", vmin=0, vmax=1)
    ax[5].set_title("3D map: shaded surface", fontsize=9)
    for x in ax:
        x.axis("off")
    v, L = meta.get("view", {}), meta.get("layout", {})
    fig.suptitle(
        f"{path.stem}  {'CUT' if cut else 'whole page'}  {L.get('kind', '')} {L.get('category', '')}  flat scale "
        f"{meta.get('flat_scale', '')}  tilt {v.get('tilt_deg', 0):.0f}  page in photo {meta.get('page_frac', 0):.2f}",
        fontsize=12,
        color="#d00000" if cut else "black",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def rejected_thumb(r, out):
    import pymupdf as fitz

    p = r["source_path"]
    if p.lower().endswith(".pdf"):
        with fitz.open(p) as d:
            pg = d[r["page_index"]]
            z = 800 / max(pg.rect.width, pg.rect.height)
            pix = pg.get_pixmap(matrix=fitz.Matrix(z, z), colorspace=fitz.csGRAY)
            g = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
    else:
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        g = cv2.resize(g, None, fx=800 / max(g.shape), fy=800 / max(g.shape), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out), g, [cv2.IMWRITE_JPEG_QUALITY, 85])


def _job(args):
    kind, a, out = args
    try:
        if kind == "sample":
            sample_figure(a[0], out, a[1])
        else:
            rejected_thumb(a, out)
    except Exception as e:
        return f"{a}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    run = Path(a.run)
    viz = run.with_name(run.name + "_viz")
    (viz / "samples").mkdir(parents=True, exist_ok=True)
    metas = {}
    if (run / "meta.jsonl").exists():
        with open(run / "meta.jsonl") as f:
            metas = {r["sample"]: r for r in map(json.loads, f)}
    jobs = [
        (
            "sample",
            (p, metas.get(p.stem, {})),
            viz / "samples" / f"{p.stem}{'_cut' if metas.get(p.stem, {}).get('view', {}).get('cut') else ''}.jpg",
        )
        for p in sorted((run / "samples").glob("*.npz"))
    ]
    if a.manifest:
        with open(Path(a.manifest).with_suffix(".rejected.jsonl")) as f:
            for r in map(json.loads, f):
                d = viz / "filtered_out" / r["reason"].replace(" ", "_")
                d.mkdir(parents=True, exist_ok=True)
                jobs.append(("rejected", r, d / f"{Path(r['source_path']).stem[:60]}_p{r['page_index']}.jpg"))
    with ProcessPoolExecutor(a.workers) as pool:
        for err in pool.map(_job, jobs, chunksize=4):
            if err:
                print(err, flush=True)
    print(viz)


if __name__ == "__main__":
    main()
