"""Per-sample visual check of a run: flat | warped with UV arrows | height map | profiles | shaded surface | back-mapped
flat. Filtered-out manifest pages (a manifest's .rejected.jsonl) as page thumbnails, one folder per reason.

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

ARROWS = 14                                         # arrows per side of the UV grid


def back_map(warped, uv, page_hw):
    """Flat page (page_hw) sampled from the warped image through the UV map (uv of each warped pixel, [0, 1] over
    the page): the UV map is inverted by scattered linear interpolation. NaN where the page is not seen."""
    from scipy.interpolate import griddata
    ph, pw = page_hw
    h, w = uv.shape[:2]
    ok = np.isfinite(uv[..., 0])
    ys, xs = np.nonzero(ok)
    src = np.stack([uv[ok, 0] * (pw - 1), uv[ok, 1] * (ph - 1)], -1)
    dst = np.stack([(xs + 0.5) * warped.shape[1] / w - 0.5, (ys + 0.5) * warped.shape[0] / h - 0.5], -1)
    gy, gx = np.mgrid[0:ph, 0:pw]
    m = griddata(src, dst, (gx, gy), method="linear").astype(np.float32)
    out = cv2.remap(warped.astype(np.float32), m[..., 0], m[..., 1], cv2.INTER_LINEAR, borderValue=np.nan)
    out[~np.isfinite(m[..., 0])] = np.nan
    return out


def shaded(map3d):
    n = np.cross(np.gradient(map3d, axis=1), np.gradient(map3d, axis=0))
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    light = np.array([-0.4, -0.5, 0.77])
    return np.clip(np.abs(n @ light), 0, 1)


def sample_figure(path, out):
    a, meta = io.load(path)
    flat, warped, uv, map3d = a["flat"], a["warped"], a["uv"].astype(np.float32), a["map3d"].astype(np.float32)
    fig, ax = plt.subplots(2, 3, figsize=(15, 13))
    ax = ax.ravel()
    ax[0].imshow(flat, cmap="gray", vmin=0, vmax=255)
    ax[0].set_title("flat")
    ax[1].imshow(warped, cmap="gray", vmin=0, vmax=255)
    h, w = uv.shape[:2]
    ys, xs = np.mgrid[h // (2 * ARROWS):h:h // ARROWS, w // (2 * ARROWS):w:w // ARROWS]
    u, v = uv[ys, xs, 0], uv[ys, xs, 1]
    ok = np.isfinite(u)
    px, py = (xs + 0.5) * warped.shape[1] / w, (ys + 0.5) * warped.shape[0] / h
    ph, pw = flat.shape
    ax[1].quiver(px[ok], py[ok], (u[ok] * pw - px[ok]), (v[ok] * ph - py[ok]), color="#e4572e",
                 angles="xy", scale_units="xy", scale=1, width=0.003)
    ax[1].set_title("warped, arrows: warped pixel -> its flat position")
    Z = map3d[..., 2]
    im = ax[2].imshow(Z, cmap="viridis")
    fig.colorbar(im, ax=ax[2], fraction=0.046, label="Z mm")
    ax[2].set_title("height map")
    ax[3].plot(map3d[Z.shape[0] // 2, :, 0], Z[Z.shape[0] // 2], label="middle row (Z vs X)")
    ax[3].plot(map3d[:, Z.shape[1] // 2, 1], Z[:, Z.shape[1] // 2], label="middle column (Z vs Y)")
    ax[3].set_aspect("equal", adjustable="datalim")
    ax[3].legend(fontsize=8)
    ax[3].set_title("profiles, mm")
    ax[4].imshow(shaded(map3d), cmap="gray", vmin=0, vmax=1)
    ax[4].set_title("shaded surface")
    back = back_map(warped, uv, flat.shape)
    ax[5].imshow(back, cmap="gray", vmin=0, vmax=255)
    ax[5].set_title("warped mapped back through UV")
    for x in ax[[0, 1, 2, 4, 5]]:
        x.axis("off")
    fig.suptitle(f"{path.stem}  {meta.get('view', '')}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=60)
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
        (sample_figure if kind == "sample" else rejected_thumb)(a, out)
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
    jobs = [("sample", p, viz / "samples" / f"{p.stem}.jpg") for p in sorted((run / "samples").glob("*.npz"))]
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
