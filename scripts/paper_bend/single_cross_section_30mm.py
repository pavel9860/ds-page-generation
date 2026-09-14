"""Single 1D cross-section: 210mm page, 30mm-high support at the middle."""
import sys
sys.path.insert(0, "/run/media/me/D/BUISNESS/pages_generation/src")

import numpy as np
import matplotlib.pyplot as plt
from plate_bend import cross_section_profile

W = 0.210        # page width (m)
X_SUPPORT = W / 2
H = 0.030        # support height (m)

x, z, v0, ell = cross_section_profile(H, X_SUPPORT, W / 2, guess=(0.0, 0.05))

print(f"v0={v0:.4f}, ell={ell*1000:.2f}mm (contact/lifted arc length from the ridge)")
print(f"z range: {z.min()*1000:.3f}..{z.max()*1000:.3f} mm")

fig, ax = plt.subplots(figsize=(7, 4))
ax.axhline(0, color="#718096", lw=1, label="table (z=0)")
ax.plot(x * 1000, z * 1000, color="#2b6cb0", lw=2.2)
ax.scatter([X_SUPPORT * 1000], [H * 1000], color="#c53030", zorder=5, s=60,
           label=f"support ({H*1000:.0f}mm)")
ax.set_xlabel("x (mm)")
ax.set_ylabel("z (mm)")
ax.set_title("210mm page over a 30mm center support (1D elastica, shooting)")
ax.set_aspect("equal", adjustable="datalim")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
out = "/run/media/me/D/BUISNESS/pages_generation/scripts/paper_bend/single_cross_section_30mm.png"
fig.savefig(out, dpi=160)
print("saved", out)
