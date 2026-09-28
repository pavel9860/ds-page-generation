"""Plot shallow-crease pages: height maps with centerlines, Lambertian shading (display only), timing.

    python scripts/creases/plot_shallow.py --n 12 --seed 0 --out scripts/creases/out
"""
import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from creases import shallow as S  # noqa: E402


def shade(w, h, elev=30.0, tilt=225.0):
    """Lambertian shading change of a height map under a distant light (display only)."""
    gy, gx = np.gradient(w, h)
    el, ti = np.radians(elev), np.radians(tilt)
    l = np.array([np.cos(el) * np.cos(ti), np.cos(el) * np.sin(ti), np.sin(el)])
    return (-l[0] * gx - l[1] * gy + l[2]) / np.sqrt(gx ** 2 + gy ** 2 + 1) / l[2] - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w", type=float, default=210.0)
    ap.add_argument("--h", type=float, default=297.0)
    ap.add_argument("--pitch", type=float, default=0.5)
    ap.add_argument("--out", default=str(Path(__file__).parent / "out"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    S.render(S.sample(np.random.default_rng(a.seed), a.w, a.h), a.w, a.h, a.pitch)      # JIT warm-up
    pages, ts = [], []
    for s in range(a.seed, a.seed + a.n):
        t = time.perf_counter()
        cr = S.sample(np.random.default_rng(s), a.w, a.h)
        pages.append((cr, S.render(cr, a.w, a.h, a.pitch)))
        ts.append(time.perf_counter() - t)
    print(f"{a.n} pages: median {np.median(ts) * 1e3:.1f} ms, max {max(ts) * 1e3:.1f} ms per page")
    cols = min(a.n, 6); rows = -(-a.n // cols)
    for name, render in [("height", False), ("shading", True)]:
        fig, ax = plt.subplots(rows, cols, figsize=(4 * cols, 5.6 * rows), constrained_layout=True, squeeze=False)
        for axi, (cr, w) in zip(ax.flat, pages):
            if render:
                axi.imshow(shade(w, a.pitch), extent=[0, a.w, a.h, 0], cmap="gray", vmin=-0.3, vmax=0.3)
            else:
                im = axi.imshow(w, extent=[0, a.w, a.h, 0], cmap="RdBu_r", vmin=-S.W_MAX, vmax=S.W_MAX)
                for c in cr:
                    axi.plot(c['line'][:, 0], c['line'][:, 1], "k-" if c['group'] >= 0 else "g-", lw=0.5)
            axi.set_xlim(0, a.w); axi.set_ylim(a.h, 0)
            ng = len({c['group'] for c in cr if c['group'] >= 0})
            axi.set_title(f"{len(cr)} creases, {ng} groups, [{w.min():.2f}, {w.max():.2f}] mm", fontsize=9)
        for axi in ax.flat[len(pages):]:
            axi.axis("off")
        if not render:
            fig.colorbar(im, ax=ax, shrink=0.5, label="height [mm] (+ mountain, - valley); black group, green single")
        fig.savefig(out / f"shallow_{name}.png", dpi=70)
        plt.close(fig)
    print(f"saved {out}/shallow_height.png, shallow_shading.png")


if __name__ == "__main__":
    main()
