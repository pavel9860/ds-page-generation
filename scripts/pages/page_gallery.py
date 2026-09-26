import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from pages import PageConfig, make_page
from pages.page import support_lines

OUT = os.path.join(os.path.dirname(__file__), "out")


def heat(ax, fig, X, Y, Z, title, cmap="viridis"):
    pc = ax.pcolormesh(X * 1e3, Y * 1e3, Z * 1e3, shading="auto", cmap=cmap)
    fig.colorbar(pc, ax=ax, label="mm")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")


def render(p, name):
    U, V = np.meshgrid(p.u, p.v)
    fig, axs = plt.subplots(2, 3, figsize=(19, 11))
    ax = axs[0, 0]
    for k, (x, z) in enumerate(p.slices):
        ax.plot(x * 1e3, z * 1e3, lw=1.5, label=f"slice {k}")
    ax.axhline(0.0, color="#777", lw=1)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title(f"1. solved profiles ({p.kind})", fontsize=10)
    heat(axs[0, 1], fig, U, V, p.z0, "2. base surface z0(u, v)")
    heat(axs[0, 2], fig, U, V, p.sag,
         f"3. shell sag, support v={p.support_span[0] * 1e3:.0f}..{p.support_span[1] * 1e3:.0f} mm, "
         f"min {p.sag.min() * 1e3:.2f} mm", "magma")
    for pu in support_lines(p.kind, p.params_a, p.u[-1]) or [0.0]:
        axs[0, 2].plot([pu * 1e3] * 2, [s * 1e3 for s in p.support_span], color="r", lw=4)
    heat(axs[1, 0], fig, U, V, p.Z, f"4. z0 + sag on (u, v), min {p.Z.min() * 1e3:.3f} mm")
    heat(axs[1, 1], fig, p.X, p.Y, p.Z, "5. final z(x, y)")
    heat(axs[1, 2], fig, U[:, 1:], V[:, 1:], np.abs(p.strain_u) * 1e-3 * 1e6,
         f"6. |strain u| 1e-6, max u {np.nanmax(np.abs(p.strain_u)):.1e} v {np.nanmax(np.abs(p.strain_v)):.1e}", "inferno")
    fig.suptitle(f"{name}: profile along {'long' if p.along_long else 'short'} side, "
                 f"A={np.round(p.params_a, 4)} B={np.round(p.params_b, 4)}")
    fig.tight_layout()
    path = os.path.join(OUT, f"{name}.png")
    fig.savefig(path, dpi=75)
    plt.close(fig)
    return path


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cfg = PageConfig()
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    runs = [(kind, False, along, frac) for kind in ("clamp", "center_support", "two_support")
            for along, frac in ((True, 0.1), (False, 1.0))] + [("center_support", True, True, None)]
    for kind, two, along, frac in runs:
        p = make_page(rng, cfg, kind=kind, two_profiles=two, along_long=along, support_frac=frac)
        name = f"page_{kind}_{'two' if two else 'one'}_{'long' if along else 'short'}_{'full' if frac == 1.0 else 'part'}"
        print(render(p, name), "sag min %.2f mm, z min %.3f mm, strain u %.1e v %.1e" % (
            p.sag.min() * 1e3, p.Z.min() * 1e3, np.nanmax(np.abs(p.strain_u)), np.nanmax(np.abs(p.strain_v))))
