"""A4 page over a 30mm-high knife-edge ridge centered on the short side,
long side, solved with the crease_bend_solver-style 1D elastica ODE
(src/edge_support_1d.py). Plots the resulting cross-section."""
import sys
sys.path.insert(0, "/run/media/me/D/BUISNESS/pages_generation/src")

import matplotlib.pyplot as plt
from edge_support_1d import build_a4_edge_support_profile

profile = build_a4_edge_support_profile(ridge_h=0.030, ridge_x=0.075)

print(f"ridge_h={profile.ridge_h*1000:.1f}mm at x={profile.ridge_x*1000:.1f}mm")
print(f"  left:  ell={profile.ell_left*1000:.2f}mm of {profile.ridge_x*1000:.2f}mm material, "
      f"touched_down={profile.touched_down_left}")
print(f"  right: ell={profile.ell_right*1000:.2f}mm of "
      f"{(profile.page_w-profile.ridge_x)*1000:.2f}mm material, "
      f"touched_down={profile.touched_down_right}")
print(f"z range: {profile.z.min()*1000:.3f}..{profile.z.max()*1000:.3f} mm")

fig, ax = plt.subplots(figsize=(7, 4))
ax.axhline(0, color="#718096", lw=1, label="table (z=0)")
ax.plot(profile.x * 1000, profile.z * 1000, color="#2b6cb0", lw=2.2)
ax.scatter([profile.ridge_x * 1000], [profile.ridge_h * 1000], color="#c53030",
           zorder=5, s=60, label=f"ridge ({profile.ridge_h*1000:.0f}mm)")
ax.set_xlabel("x (mm)")
ax.set_ylabel("z (mm)")
ax.set_title("A4 short-side cross-section over a 30mm center ridge (1D elastica)")
ax.set_aspect("equal", adjustable="datalim")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
out = "/run/media/me/D/BUISNESS/pages_generation/scripts/paper_bend/a4_edge_support_1d.png"
fig.savefig(out, dpi=160)
print("saved", out)
