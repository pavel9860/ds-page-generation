"""A4 page draped over a long-edge ridge, 30mm peak height at the center
of the long side, tapering to 0 at the two short edges. No sag: each
cross-section perpendicular to the ridge is an independent 1D heavy
elastica (see src/plate_bend.py)."""
import sys
sys.path.insert(0, "/run/media/me/D/BUISNESS/pages_generation/src")

import numpy as np
import matplotlib.pyplot as plt
from plate_bend import edge_support_page

PAGE_W = 0.210   # short side (m) -- x, direction the beam bends across
PAGE_H = 0.297   # long side (m)  -- y, direction the ridge runs along
RIDGE_X = PAGE_W / 2   # ridge sits along the long-side centerline
PEAK_H = 0.030

X, Y, Z = edge_support_page(PAGE_W, PAGE_H, RIDGE_X, PEAK_H, ny=61, nx=220)

print(f"z range: {Z.min()*1000:.2f}..{Z.max()*1000:.2f} mm")

# ---- heatmap ----
fig, ax = plt.subplots(figsize=(6.2, 8.4))
im = ax.pcolormesh(X * 1000, Y * 1000, Z * 1000, shading="gouraud", cmap="viridis")
cbar = fig.colorbar(im, ax=ax, label="z (mm)")
ax.set_xlabel("x -- short side (mm)")
ax.set_ylabel("y -- long side (mm)")
ax.set_title("A4 page over long-edge-centerline ridge, 30mm peak (no sag)")
ax.set_aspect("equal")
fig.tight_layout()
out1 = "/run/media/me/D/BUISNESS/pages_generation/scripts/paper_bend/edge_support_30mm_heatmap.png"
fig.savefig(out1, dpi=160)
print("saved", out1)

# ---- cross-section profiles at several y ----
fig2, ax2 = plt.subplots(figsize=(8, 4.5))
y_fracs = [0.5, 0.4, 0.3, 0.2, 0.1, 0.02]
ny = X.shape[0]
for frac in y_fracs:
    i = int(round(frac * (ny - 1)))
    y_mm = Y[i, 0] * 1000
    ax2.plot(X[i] * 1000, Z[i] * 1000, lw=2, label=f"y={y_mm:.0f}mm")
ax2.axhline(0, color="#718096", lw=1)
ax2.set_xlabel("x -- short side (mm)")
ax2.set_ylabel("z (mm)")
ax2.set_title("Cross-section profiles at various positions along the ridge")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
fig2.tight_layout()
out2 = "/run/media/me/D/BUISNESS/pages_generation/scripts/paper_bend/edge_support_30mm_profiles.png"
fig2.savefig(out2, dpi=160)
print("saved", out2)

# ---- along-ridge profile (peak height vs y) ----
fig3, ax3 = plt.subplots(figsize=(8, 3.5))
mid_x_idx = X.shape[1] // 2
ax3.plot(Y[:, 0] * 1000, Z[:, mid_x_idx] * 1000, lw=2, color="#c53030")
ax3.set_xlabel("y -- long side (mm)")
ax3.set_ylabel("z at ridge (mm)")
ax3.set_title("Ridge height profile along the long side")
ax3.grid(alpha=0.3)
fig3.tight_layout()
out3 = "/run/media/me/D/BUISNESS/pages_generation/scripts/paper_bend/edge_support_30mm_ridge_profile.png"
fig3.savefig(out3, dpi=160)
print("saved", out3)
