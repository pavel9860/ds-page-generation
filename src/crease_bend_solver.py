"""
Same physical model as nonlinear_asymmetric.py (coupled arms, genuinely
free/unsupported crease: z, M continuous; slope jump prescribed; shear
continuity via ell_s=ell_l), but solved by SHOOTING (solve_ivp + fsolve)
instead of solve_bvp collocation, which was diverging.

Shooting unknowns: theta_l0, z0 (shared crease height), v0 (shared crease
moment/curvature), ell (shared lifted length, ell_s=ell_l enforced by
using one variable for both).
Target conditions (matched via fsolve): theta_s(1)=pi, z_s(1)=0,
theta_l(1)=0, z_l(1)=0.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, least_squares
import matplotlib.pyplot as plt

q = 0.7848
D = 0.688e-3
L = 0.297
a = 0.080
La = L - a
fold_interior_deg = 140.0
turn_deg = 180.0 - fold_interior_deg
jump = np.radians(180.0 + turn_deg)   # theta_s0 - theta_l0 = jump (corrected branch: found via sign scan, 180-turn_deg had no positive-ell root)


def rhs(sp, y, ell):
    theta, v, z = y
    load = q * ell * (1 - sp)
    return [ell * v, ell * load * np.cos(theta) / D, ell * np.sin(theta)]


def integrate_arm(theta0, v0, z0, ell):
    sol = solve_ivp(rhs, [0, 1], [theta0, v0, z0], args=(ell,), method="RK45",
                     rtol=1e-10, atol=1e-12, dense_output=True)
    return sol


def residual(p):
    theta_l0, z0, v0, ell = p
    theta_s0 = theta_l0 + jump
    sol_s = integrate_arm(theta_s0, v0, z0, ell)
    sol_l = integrate_arm(theta_l0, v0, z0, ell)
    ys = sol_s.y[:, -1]
    yl = sol_l.y[:, -1]
    return [ys[0] - np.pi, ys[2], yl[0], yl[2]]


guess = [np.radians(-15), 0.02, 0.0, 0.09]
sol_p, info, ier, msg = fsolve(residual, guess, full_output=True, xtol=1e-12, maxfev=5000)
res = residual(sol_p)
print(f"ier={ier}, msg={msg}, residual={res}")
theta_l0, z0, v0, ell = sol_p
theta_s0 = theta_l0 + jump
print(f"theta_s0={np.degrees(theta_s0):.3f}deg, theta_l0={np.degrees(theta_l0):.3f}deg, "
      f"z0={z0*1000:.4f}mm, v0={v0:.4f}, ell={ell*1000:.4f}mm")

print(f"note: ell={ell*1000:.4f}mm vs short-arm material={a*1000:.1f}mm -- "
      f"{'short arm has enough material to reach touchdown' if ell <= a else 'short arm runs out of material before touchdown: it stays lifted its entire length'}")

sol_s = integrate_arm(theta_s0, v0, z0, ell)
sol_l = integrate_arm(theta_l0, v0, z0, ell)

# short arm: only trust the solution up to its own material length (sp_cut);
# beyond that the arm simply doesn't exist -- it's not "touching", it just ends.
sp_cut_s = min(1.0, a / ell) if ell > 0 else 1.0
sp_s = np.linspace(0, sp_cut_s, 1000)
sp_l = np.linspace(0, min(1.0, La / ell) if ell > 0 else 1.0, 1000)
theta_s, v_s, z_s = sol_s.sol(sp_s)
theta_l, v_l, z_l = sol_l.sol(sp_l)
x_s = np.concatenate([[a], a + np.cumsum((np.cos(theta_s[:-1]) + np.cos(theta_s[1:])) / 2 * np.diff(sp_s) * ell)])
x_l = np.concatenate([[a], a + np.cumsum((np.cos(theta_l[:-1]) + np.cos(theta_l[1:])) / 2 * np.diff(sp_l) * ell)])

print(f"z ranges: short {z_s.min()*1000:.4f}..{z_s.max()*1000:.4f}mm, "
      f"long {z_l.min()*1000:.4f}..{z_l.max()*1000:.4f}mm")
print(f"x ranges: short {x_s.min()*1000:.4f}..{x_s.max()*1000:.4f}mm, "
      f"long {x_l.min()*1000:.4f}..{x_l.max()*1000:.4f}mm")

X_short, Z_short = x_s, z_s
X_long, Z_long = x_l, z_l
if sp_cut_s >= 1.0 - 1e-9 and ell < a - 1e-6:
    X_short = np.concatenate([X_short, [0.0]])
    Z_short = np.concatenate([Z_short, [0.0]])
if ell < La - 1e-6:
    X_long = np.concatenate([X_long, [L]])
    Z_long = np.concatenate([Z_long, [0.0]])

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.axhline(0, color="#718096", lw=1, label="table (z=0)")
ax.plot(X_short * 1000, Z_short * 1000, color="#805ad5", lw=2.2, label="short arm")
ax.plot(X_long * 1000, Z_long * 1000, color="#c53030", lw=2.2, label="long arm")
ax.scatter([a * 1000], [z0 * 1000], color="#2b6cb0", zorder=5, s=70,
           label=f"crease (free, z={z0*1000:.2f}mm)")
ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)")
ax.set_title(f"Nonlinear coupled shooting, asymmetric crease at {a*1000:.0f}mm, {fold_interior_deg:.0f}deg fold")
ax.set_aspect("equal", adjustable="datalim")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout()
out = "/run/media/me/D/BUISNESS/pages_generation/scripts/paper_bend/nonlinear_asymmetric_shoot_80mm.png"
fig.savefig(out, dpi=160)
print("saved", out)
