from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import brentq, fsolve, least_squares

from elastica_continuation import _dv0_dq_fixedstep, _rk4_fixed_9

__version__ = "1.0.0"

Q = 0.7848      # distributed weight parameter [N/m per unit width]
D = 0.688e-3    # bending stiffness parameter [N*m per unit width]
L = 0.297       # A4 long side, bending direction x [m]
W = 0.210       # A4 short side, transverse direction y [m]

NX, NY = 240, 81
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "paper_bend", "a4_page_bending_out")
RTOL, ATOL = 1e-11, 1e-13
TOL = 1e-6

# solve_free_arm continuation constants (see elastica_continuation.py, tuned
FREE_ARM_N_OUTER = 3
FREE_ARM_N_INNER = 8
FREE_ARM_N_POLISH = 1
FREE_ARM_N_POLISH_STEPS = 20


def rhs(sp: float, state: np.ndarray, ell: float, q: float, d: float,
        r_tip: float, h_force: float) -> list[float]:
    theta, v, _ = state
    shear = q * ell * (1.0 - sp) + r_tip
    return [ell * v,
            ell * (shear * np.cos(theta) - h_force * np.sin(theta)) / d,
            ell * np.sin(theta)]


def integrate(theta0: float, v0: float, z0: float, ell: float, q: float = Q,
              d: float = D, r_tip: float = 0.0, h_force: float = 0.0,
              rtol: float = RTOL, atol: float = ATOL):
    return solve_ivp(rhs, (0.0, 1.0), [theta0, v0, z0],
                     args=(ell, q, d, r_tip, h_force), method="RK45",
                     rtol=rtol, atol=atol, dense_output=True)


def _crosses(sol, plane: float, n: int = 4000) -> float | None:
    sp = np.linspace(0.0, 1.0, n)
    diff = sol.sol(sp)[2] - plane
    nonzero = np.flatnonzero(diff != 0.0)
    if nonzero.size == 0:
        return None
    sign0 = np.sign(diff[nonzero[0]])
    flips = np.flatnonzero((np.sign(diff) != sign0) & (np.sign(diff) != 0.0))
    flips = flips[flips > nonzero[0]]
    return None if flips.size == 0 else sp[max(flips[0] - 1, 0)]


def solve_free_arm(theta0: float, z0: float, ell_max: float, q: float = Q,
                   d: float = D, v0_guess: float = 0.0,
                   rtol: float = RTOL, atol: float = ATOL) -> dict:
    """Working solution. Never change it!
    Finds v0 with M(ell_max)=0 by continuation in q from q=0 (where v0=0 is
    the exact trivial root, for any theta0/ell_max), tracking the one branch
    continuously connected to that anchor."""
    q_steps = np.linspace(0.0, q, FREE_ARM_N_OUTER + 1)
    v0 = 0.0
    h_q = q / FREE_ARM_N_OUTER
    for i in range(FREE_ARM_N_OUTER):
        qq = q_steps[i]
        k1 = _dv0_dq_fixedstep(qq, v0, ell_max, d, theta0, FREE_ARM_N_INNER)[0]
        k2 = _dv0_dq_fixedstep(qq + 0.5 * h_q, v0 + 0.5 * h_q * k1, ell_max, d, theta0, FREE_ARM_N_INNER)[0]
        k3 = _dv0_dq_fixedstep(qq + 0.5 * h_q, v0 + 0.5 * h_q * k2, ell_max, d, theta0, FREE_ARM_N_INNER)[0]
        k4 = _dv0_dq_fixedstep(qq + h_q, v0 + h_q * k3, ell_max, d, theta0, FREE_ARM_N_INNER)[0]
        v0 = v0 + (h_q / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    for _ in range(FREE_ARM_N_POLISH):
        y = _rk4_fixed_9(theta0, v0, ell_max, q, d, n_steps=FREE_ARM_N_POLISH_STEPS)
        v0 = v0 - y[1] / y[4]
    return {"branch": "free", "theta0": theta0, "v0": v0, "ell": ell_max,
            "r_tip": 0.0, "sol": integrate(theta0, v0, z0, ell_max, q, d, rtol=rtol, atol=atol)}


def _tip_res(theta0, z0, ell_max, plane, qq, d, rtol, atol):
    def res(p):
        sol = integrate(theta0, p[0], z0, ell_max, qq, d, r_tip=p[1], rtol=rtol, atol=atol)
        return [sol.y[2, -1] - plane, sol.y[1, -1]]
    return res


def _flat_res(theta0, z0, ell_max, plane, flat_theta, qq, d, rtol, atol):
    def res(p):
        sol = integrate(theta0, p[0], z0, p[1] * ell_max, qq, d, rtol=rtol, atol=atol)
        return [sol.y[2, -1] - plane, sol.y[0, -1] - flat_theta]
    return res


def _argmin_z(sol, n: int = 4000) -> float:
    sp = np.linspace(0.0, 1.0, n)
    return float(sp[np.argmin(sol.sol(sp)[2])])


def _flat_fit(theta0, z0, ell_max, plane, flat_theta, qq, d, rtol, atol, v0g, frac_g):
    res = _flat_res(theta0, z0, ell_max, plane, flat_theta, qq, d, rtol, atol)
    return least_squares(res, [v0g, max(frac_g, 1e-6)],
                         bounds=([-np.inf, 1e-6], [np.inf, 1.0]), xtol=_XTOL, ftol=_XTOL)


_XTOL = 1e-9
_ATOL_ODE = 1e-8
_RTOL_ODE = 1e-6


def solve_arm(theta0: float, z0: float, ell_max: float, plane: float = 0.0,
              flat_theta: float = 0.0, q: float = Q, d: float = D,
              v0_guess: float = 0.0, allow_contact: bool = True,
              rtol: float = _RTOL_ODE, atol: float = _ATOL_ODE,
              n_coarse: int = 9, n_bisect: int = 24) -> dict:
    min_dev = np.radians(1.0)
    dev = theta0 - flat_theta
    if abs(dev) < min_dev:
        theta0 = flat_theta + (min_dev if dev >= 0 else -min_dev)

    sp_dense = np.linspace(0.0, 1.0, 500)

    v0g = v0_guess
    q_lo, v0_lo = 0.0, 0.0
    q_hi = None
    for qq in np.linspace(0.0, q, n_coarse)[1:]:
        arm = solve_free_arm(theta0, z0, ell_max, qq, d, v0g, rtol=rtol, atol=atol)
        v0g = arm["v0"]
        if allow_contact and arm["sol"].sol(sp_dense)[2].min() < plane - 1e-9:
            q_hi = qq
            break
        q_lo, v0_lo = qq, v0g
    else:
        return arm

    for _ in range(n_bisect):
        mid = 0.5 * (q_lo + q_hi)
        arm_mid = solve_free_arm(theta0, z0, ell_max, mid, d, v0_lo, rtol=rtol, atol=atol)
        if arm_mid["sol"].sol(sp_dense)[2].min() < plane - 1e-9:
            q_hi = mid
        else:
            q_lo, v0_lo = mid, arm_mid["v0"]

    v0g, r_tip_g = v0_lo, 0.0
    res = _tip_res(theta0, z0, ell_max, plane, q_hi, d, rtol, atol)
    fit = least_squares(res, [v0g, r_tip_g], xtol=_XTOL, ftol=_XTOL)
    v0g, r_tip_g = fit.x
    sol = integrate(theta0, v0g, z0, ell_max, q_hi, d, r_tip=r_tip_g, rtol=rtol, atol=atol)
    tip_ok = fit.status > 0 and np.max(np.abs(fit.fun)) <= TOL and sol.sol(sp_dense)[2].min() >= plane - 1e-9

    q_from = q_hi
    frac_g = None if tip_ok else _argmin_z(sol)
    phase = "tip" if tip_ok else "flat"
    last = ({"branch": "tip", "theta0": theta0, "v0": v0g, "ell": ell_max,
             "r_tip": r_tip_g, "sol": sol} if tip_ok else None)

    for qq in np.linspace(q_from, q, 5)[1:]:
        if phase == "tip":
            res = _tip_res(theta0, z0, ell_max, plane, qq, d, rtol, atol)
            fit = least_squares(res, [v0g, r_tip_g], xtol=_XTOL, ftol=_XTOL)
            if fit.status > 0 and np.max(np.abs(fit.fun)) <= TOL:
                v0g, r_tip_g = fit.x
                sol = integrate(theta0, v0g, z0, ell_max, qq, d, r_tip=r_tip_g, rtol=rtol, atol=atol)
                if sol.sol(sp_dense)[2].min() >= plane - 1e-9:
                    last = {"branch": "tip", "theta0": theta0, "v0": v0g, "ell": ell_max,
                           "r_tip": r_tip_g, "sol": sol}
                    continue
                frac_g = _argmin_z(sol)
            phase = "flat"

        fit = _flat_fit(theta0, z0, ell_max, plane, flat_theta, qq, d, rtol, atol, v0g, frac_g)
        if fit.status <= 0 or np.max(np.abs(fit.fun)) > TOL:
            raise RuntimeError(
                f"solve_arm: flat continuation failed at q={qq:.4f} "
                f"(target {q:.4f}), theta0={np.degrees(theta0):.4f}deg, "
                f"ell_max={ell_max*1e3:.2f}mm")
        v0g, frac_g = fit.x
        ell = frac_g * ell_max
        last = {"branch": "flat", "theta0": theta0, "v0": v0g, "ell": ell,
               "r_tip": 0.0, "sol": integrate(theta0, v0g, z0, ell, qq, d, rtol=rtol, atol=atol)}
    return last


def solve_edge_cantilever(theta0_of_x, ell: float, xs: np.ndarray,
                          q: float = Q, d: float = D,
                          v0_guess: float = 0.0) -> list[dict]:
    arms = []
    v0g = v0_guess
    for x in xs:
        arm = solve_arm(theta0_of_x(x), 0.0, ell, q=q, d=d, v0_guess=v0g)
        v0g = arm["v0"]
        arms.append(arm)
    return arms


def calibrate_load_for_sag(ell: float, target_sag: float, q_lo: float = 1e-4,
                           q_hi: float = Q, d: float = D) -> float:
    def tip(qt):
        return solve_free_arm(0.0, 0.0, ell, q=qt, d=d)["sol"].y[2, -1]

    return brentq(lambda qt: tip(qt) + target_sag, q_lo, q_hi, xtol=1e-13)


def arm_xz(arm: dict, x0: float, n: int = 600,
           material_len: float | None = None):
    sp = np.linspace(0.0, 1.0, n)
    theta, _, z = arm["sol"].sol(sp)
    ds = np.diff(sp) * arm["ell"]
    dx = 0.5 * (np.cos(theta[:-1]) + np.cos(theta[1:])) * ds
    x = x0 + np.concatenate([[0.0], np.cumsum(dx)])
    if (material_len is not None and arm["branch"] == "flat"
            and arm["ell"] < material_len - 1e-9):
        x = np.append(x, x[-1] + np.cos(theta[-1]) * (material_len - arm["ell"]))
        z = np.append(z, arm["sol"].y[2, -1])
    return x, z


def mirror_join(x_right, z_right, x_left, z_left):
    x = np.concatenate([x_left[::-1], x_right])
    z = np.concatenate([z_left[::-1], z_right])
    order = np.argsort(x)
    return x[order], z[order]


def tile_y(x, z, ny: int = NY):
    y = np.linspace(0.0, W, ny)
    return (np.tile(x, (ny, 1)), np.tile(y[:, None], (1, len(x))),
            np.tile(z, (ny, 1)))


def tile_x(y, z, nx: int = NX):
    x = np.linspace(0.0, L, nx)
    return (np.tile(x, (len(y), 1)), np.tile(np.asarray(y)[:, None], (1, nx)),
            np.tile(np.asarray(z)[:, None], (1, nx)))


def render(title, profiles, labels, X, Y, Z, fname, support=None,
           profile_xlabel="x [mm]"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for (x, z), lab in zip(profiles, labels):
        ax1.plot(np.asarray(x) * 1e3, np.asarray(z) * 1e3, lw=2, label=lab)
    ax1.axhline(0.0, color="#777", lw=1)
    if support is not None:
        ax1.plot([support[0] * 1e3] * 2, [0.0, support[1] * 1e3],
                 color="#2b6cb0", lw=2.5, alpha=0.8)
        ax1.scatter([support[0] * 1e3], [support[1] * 1e3], s=45,
                    color="#2b6cb0", zorder=5)
    ax1.set_xlabel(profile_xlabel)
    ax1.set_ylabel("z [mm]")
    ax1.set_title("profile")
    ax1.grid(alpha=0.3)
    ax1.set_aspect("equal", adjustable="datalim")
    if len(labels) > 1:
        ax1.legend(fontsize=8)

    pc = ax2.pcolormesh(X * 1e3, Y * 1e3, Z * 1e3, shading="gouraud",
                        cmap="viridis")
    fig.colorbar(pc, ax=ax2, label="z [mm]")
    cs = ax2.contour(X * 1e3, Y * 1e3, Z * 1e3, levels=10, colors="k",
                     linewidths=0.4, alpha=0.5)
    ax2.clabel(cs, fontsize=6, fmt="%.0f")
    ax2.set_xlabel("x [mm]")
    ax2.set_ylabel("y [mm]")
    ax2.set_title("Z heatmap")
    ax2.set_aspect("equal")

    fig.suptitle(title)
    fig.tight_layout()
    path = os.path.join(OUTDIR, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _center_support_profile(h, allow_contact=True, v0_guess=-8.0):
    half = L / 2
    right = solve_arm(0.0, h, half, allow_contact=allow_contact,
                      v0_guess=v0_guess)
    left = solve_arm(np.pi, h, half, flat_theta=np.pi,
                     allow_contact=allow_contact, v0_guess=right["v0"])
    xr, zr = arm_xz(right, half, material_len=half)
    xl, zl = arm_xz(left, half, material_len=half)
    return mirror_join(xr, zr, xl, zl), right, left


def case1():
    h = 0.040
    (x, z), right, _ = _center_support_profile(h, allow_contact=False)
    X, Y, Z = tile_y(x, z)
    print(f"case1  branch={right['branch']:5s} ell={right['ell']*1e3:7.2f}mm "
          f"v0={right['v0']:8.4f}  z=[{z.min()*1e3:7.3f},{z.max()*1e3:7.3f}]mm")
    return render("Case 1 — line support x=L/2, h=40 mm, free ends (contact disabled)",
                  [(x, z)], ["profile"], X, Y, Z, "case1_center_40mm_free.png",
                  support=(L / 2, h))


def case2():
    h = 0.020
    (x, z), right, _ = _center_support_profile(h)
    X, Y, Z = tile_y(x, z)
    print(f"case2  branch={right['branch']:5s} ell={right['ell']*1e3:7.2f}mm "
          f"v0={right['v0']:8.4f}  flat_len={(L/2-right['ell'])*1e3:6.2f}mm")
    return render("Case 2 — line support x=L/2, h=20 mm, ends touch down on z=0",
                  [(x, z)], ["profile"], X, Y, Z, "case2_center_20mm_contact.png",
                  support=(L / 2, h))


def case3(h_top=0.040, h_bot=0.020):
    ys = np.linspace(0.0, W, NY)
    profiles, branches = [], []
    v0r = v0l = -8.0
    xmin, xmax = np.inf, -np.inf
    for y in ys:
        h = h_top + (h_bot - h_top) * (y / W)
        (x, z), right, left = _center_support_profile(h, v0_guess=v0r)
        v0r, v0l = right["v0"], left["v0"]
        profiles.append((x, z))
        branches.append(right["branch"])
        xmin, xmax = min(xmin, x.min()), max(xmax, x.max())
    xc = np.linspace(xmin, xmax, NX)
    Z = np.array([np.interp(xc, x, z, left=np.nan, right=np.nan)
                  for x, z in profiles])
    X, Y = np.meshgrid(xc, ys)
    switch = next((ys[i] for i in range(1, NY) if branches[i] != branches[i - 1]),
                  None)
    print(f"case3  branches={sorted(set(branches))} "
          f"switch_at_y={'n/a' if switch is None else f'{switch*1e3:.1f}mm'}")
    return render("Case 3 — angled support, h=40 mm at y=0 to h=20 mm at y=W",
                  [profiles[0], profiles[-1]],
                  [f"y=0 (h={h_top*1e3:.0f} mm)", f"y=W (h={h_bot*1e3:.0f} mm)"],
                  X, Y, Z, "case3_angled_support.png")


def case4(a=0.070, h=0.020):
    la = L - a

    def res(p):
        theta_l0, v0, ell_l = p
        s_left = integrate(theta_l0 + np.pi, v0, h, a)
        s_right = integrate(theta_l0, v0, h, ell_l)
        return [s_left.y[1, -1], s_right.y[2, -1], s_right.y[0, -1]]

    theta_l0, v0, ell_l = fsolve(res, [0.0, -7.8, 0.14], xtol=1e-13)
    assert np.allclose(res([theta_l0, v0, ell_l]), 0.0, atol=1e-10)
    left = {"branch": "free", "theta0": theta_l0 + np.pi, "v0": v0, "ell": a,
            "r_tip": 0.0, "sol": integrate(theta_l0 + np.pi, v0, h, a)}
    right = {"branch": "flat", "theta0": theta_l0, "v0": v0, "ell": ell_l,
             "r_tip": 0.0, "sol": integrate(theta_l0, v0, h, ell_l)}
    xs, zs = arm_xz(left, a, material_len=a)
    xl, zl = arm_xz(right, a, material_len=la)
    x, z = mirror_join(xl, zl, xs, zs)
    X, Y, Z = tile_y(x, z)
    print(f"case4  theta_l0={np.degrees(theta_l0):7.3f}deg v0={v0:8.4f} "
          f"ell_right={ell_l*1e3:7.2f}mm flat_len={(la-ell_l)*1e3:6.2f}mm "
          f"left_tip_z={zs[-1]*1e3:6.2f}mm")
    return render("Case 4 — support at x=70 mm, h=20 mm, right arm lies on z=0",
                  [(x, z)], ["profile"], X, Y, Z, "case4_offset_support.png",
                  support=(a, h))


def case5(theta0_deg=30.0, plane=0.010):
    arm = solve_arm(np.radians(theta0_deg), 0.0, L, plane=plane)
    x, z = arm_xz(arm, 0.0, material_len=L)
    X, Y, Z = tile_y(x, z)
    print(f"case5  branch={arm['branch']:5s} ell={arm['ell']*1e3:7.2f}mm "
          f"v0={arm['v0']:8.4f} z_max={z.max()*1e3:6.3f}mm "
          f"touchdown_x={x[len(x)-2]*1e3:7.2f}mm")
    return render(f"Case 5 — cantilever clamped at {theta0_deg:.0f}°, "
                  f"obstacle plane z={plane*1e3:.0f} mm",
                  [(x, z)], ["profile"], X, Y, Z, "case5_book_page_plane.png")


def case6(h=0.040, corner_sag=0.003):
    (x1, z1), right, _ = _center_support_profile(h, allow_contact=False)
    xc = np.linspace(x1.min(), x1.max(), NX)
    zx = np.interp(xc, x1, z1)

    q_t = calibrate_load_for_sag(W / 2, corner_sag)
    trans = solve_free_arm(0.0, 0.0, W / 2, q=q_t)
    sp = np.linspace(0.0, 1.0, NY // 2 + 1)
    zt = trans["sol"].sol(sp)[2]
    yt = np.concatenate([[0.0], np.cumsum(
        0.5 * (np.cos(trans["sol"].sol(sp)[0][:-1])
               + np.cos(trans["sol"].sol(sp)[0][1:])) * np.diff(sp) * (W / 2))])
    y_half = np.linspace(0.0, W / 2, NY // 2 + 1)
    g_half = np.interp(y_half, yt, zt)
    ys = np.linspace(0.0, W, NY)
    g = np.interp(np.abs(ys - W / 2), y_half, g_half)

    f = np.abs(zx - h) / max(abs(zx - h).max(), 1e-12)
    Z = zx[None, :] + f[None, :] * g[:, None]
    X, Y = np.meshgrid(xc, ys)
    i_end = int(np.argmin(np.abs(xc - xc.max())))
    print(f"case6  q_t={q_t:.6f} (vs q={Q}) corner_droop="
          f"{(Z[NY//2, i_end]-Z[0, i_end])*1e3:6.3f}mm "
          f"z=[{Z.min()*1e3:7.3f},{Z.max()*1e3:7.3f}]mm")
    return render("Case 6 — h=40 mm support with transverse sag "
                  f"({corner_sag*1e3:.0f} mm corner droop)",
                  [(xc, Z[NY // 2]), (xc, Z[0])],
                  ["y=W/2 (centerline)", "y=0 (long edge)"],
                  X, Y, Z, "case6_2d_sag.png")


def case7(theta_bottom_deg=30.0, theta_top_deg=45.0):
    theta0_of_x = (lambda x: np.radians(theta_bottom_deg)
                   + np.radians(theta_top_deg - theta_bottom_deg) * (x / L))
    xs = np.linspace(0.0, L, NX)
    arms = solve_edge_cantilever(theta0_of_x, W, xs,
                                 v0_guess=solve_free_arm(theta0_of_x(0.0), 0.0,
                                                         W)["v0"])
    sp_fine = np.linspace(0.0, 1.0, 2000)
    z_min = min(arm["sol"].sol(sp_fine)[2].min() for arm in arms)
    assert z_min >= -1e-6, f"case7: z went to {z_min*1e3:.4f}mm below the table"

    profiles = [arm_xz(a, 0.0, material_len=W) for a in arms]
    ymin, ymax = 0.0, min(y.max() for y, _ in profiles)
    yc = np.linspace(ymin, ymax, NY)
    Z = np.array([np.interp(yc, y, z, left=np.nan, right=np.nan)
                  for y, z in profiles]).T
    X, Y = np.meshgrid(xs, yc)
    branches = sorted({a["branch"] for a in arms})
    print(f"case7  theta(x=0)={theta_bottom_deg:.1f}deg "
          f"theta(x=L)={theta_top_deg:.1f}deg branches={branches} "
          f"y_reach=[{min(y.max() for y, _ in profiles)*1e3:.2f},"
          f"{max(y.max() for y, _ in profiles)*1e3:.2f}]mm "
          f"z_min_finegrid={z_min*1e3:.4f}mm (verified >= 0, every x-slice) "
          f"z=[{np.nanmin(Z)*1e3:7.3f},{np.nanmax(Z)*1e3:7.3f}]mm")
    return render("Case 7 — long edge y=0 clamped, angle "
                  f"{theta_bottom_deg:.0f}° (x=0) to {theta_top_deg:.0f}° "
                  "(x=L), cantilevered across W",
                  [profiles[0], profiles[-1]],
                  [f"x=0 ({theta_bottom_deg:.0f}°)", f"x=L ({theta_top_deg:.0f}°)"],
                  X, Y, Z, "case7_edge_angle_gradient.png",
                  profile_xlabel="y [mm]")


def solve_grounded_free_arm(theta0_deg: float, ell_max: float,
                            plane: float = 0.0, flat_theta: float = 0.0,
                            n_cont: int = 41) -> dict:
    theta0 = np.radians(theta0_deg)
    v0g = 0.0
    for tdeg in np.linspace(0.0, theta0_deg, n_cont):
        v0g = solve_free_arm(np.radians(tdeg), plane, ell_max, v0_guess=v0g)["v0"]
    return solve_arm(theta0, plane, ell_max, plane=plane, flat_theta=flat_theta,
                     v0_guess=v0g)


def solve_valley_case8(a: float, turn_deg: float, q: float = Q, d: float = D,
                       tol: float = 1e-6, step_deg: float = 1.0,
                       max_iter: int = 200) -> dict:
    La = L - a
    jump_deg = 180.0 - turn_deg

    def side(theta_deg, ell_max, flat_theta, vg):
        return solve_arm(np.radians(theta_deg), 0.0, ell_max, plane=0.0,
                         flat_theta=flat_theta, q=q, d=d, v0_guess=vg)

    theta_r = turn_deg / 2.0
    right = side(theta_r, La, 0.0, -1.0)
    left = side(theta_r + jump_deg, a, np.pi, 1.0)
    s = right["v0"] + left["v0"]
    if abs(s) < tol:
        return {"theta_r_deg": theta_r, "right": right, "left": left}

    probe = step_deg if abs(right["v0"]) >= abs(left["v0"]) else -step_deg
    right_try = side(theta_r + probe, La, 0.0, right["v0"])
    left_try = side(theta_r + probe + jump_deg, a, np.pi, left["v0"])
    s_try = right_try["v0"] + left_try["v0"]
    direction = probe if abs(s_try) < abs(s) else -probe

    step = step_deg
    for _ in range(max_iter):
        theta_next = theta_r + direction
        right_next = side(theta_next, La, 0.0, right["v0"])
        left_next = side(theta_next + jump_deg, a, np.pi, left["v0"])
        prev_s = s
        theta_r, right, left = theta_next, right_next, left_next
        s = right["v0"] + left["v0"]
        if abs(s) < tol:
            return {"theta_r_deg": theta_r, "right": right, "left": left}
        if prev_s * s < 0:
            step *= 0.5
            direction = -np.sign(direction) * step
        else:
            direction = np.sign(direction) * step
    raise RuntimeError(
        f"solve_valley_case8: momentum sum did not converge for "
        f"a={a*1e3:.1f}mm, turn={turn_deg:.1f}deg, last sum={s}")


def _case8_crease_bvp(sp, y, p, q, d):
    ell = p[0]
    theta_left, k_left, _z_left, theta_right, k_right, _z_right = y
    v_left = q * ell * (1.0 - sp)
    v_right = q * ell * (1.0 - sp)
    return np.vstack([ell * k_left, ell * v_left * np.cos(theta_left) / d, ell * np.sin(theta_left),
                       ell * k_right, ell * v_right * np.cos(theta_right) / d, ell * np.sin(theta_right)])


def _case8_crease_system(a, La, jump, guess, q=Q, d=D):
    theta_r0_g, _z0_g, v0_g, ell_g = guess
    n = 50
    sp = np.linspace(0.0, 1.0, n)
    y0 = np.zeros((6, n))
    y0[0] = theta_r0_g + jump
    y0[3] = theta_r0_g
    y0[1] = v0_g
    y0[4] = v0_g

    def bc(ya, yb, p):
        theta_left0, k_left0, z_left0, theta_right0, k_right0, z_right0 = ya
        theta_left1, k_left1, z_left1, theta_right1, k_right1, z_right1 = yb
        return np.array([z_left0, z_right0, k_left0 - k_right0,
                          theta_left0 - theta_right0 - jump,
                          theta_left1 - np.pi, theta_right1 - 0.0, z_left1 - z_right1])

    sol = solve_bvp(lambda s, y, p: _case8_crease_bvp(s, y, p, q, d), bc, sp, y0,
                     p=[ell_g], tol=1e-10, max_nodes=20000)
    if sol.status != 0:
        raise RuntimeError(f"case8 crease solve_bvp did not converge: {sol.message}")
    theta_r0 = sol.y[3, 0]
    v0 = sol.y[1, 0]
    ell = sol.p[0]
    z0 = -sol.y[5, -1]
    return theta_r0, z0, v0, ell


def _case8_profile(a, La, theta_r0, z0, v0, ell, jump):
    theta_left0 = theta_r0 + jump
    sol_left = integrate(theta_left0, v0, z0, ell)
    sol_right = integrate(theta_r0, v0, z0, ell)

    left_ok = ell <= a + 1e-9
    cut = min(ell, a)
    sp = np.linspace(0.0, cut / ell, 300)
    th, _, z_left = sol_left.sol(sp)
    ds = np.diff(sp) * ell
    x_left = a + np.concatenate([[0.0], np.cumsum(0.5 * (np.cos(th[:-1]) + np.cos(th[1:])) * ds)])
    if not left_ok:
        print(f"case8  INCONSISTENT: shared ell={ell*1e3:.3f}mm exceeds the left "
              f"arm's {a*1e3:.1f}mm material by {(ell-a)*1e3:.3f}mm — left arm cut "
              f"off {cut*1e3:.1f}mm in, ending at theta={np.degrees(th[-1]):.2f}deg "
              f"z={z_left[-1]*1e3:.3f}mm (not flat on the table)")

    right = {"branch": "flat" if ell < La - 1e-9 else "free", "theta0": theta_r0,
             "v0": v0, "ell": ell, "sol": sol_right}
    x_right, z_right = arm_xz(right, a, material_len=La)

    x = np.concatenate([x_left[::-1], x_right])
    z = np.concatenate([z_left[::-1], z_right])
    order = np.argsort(x)
    return x[order], z[order]


def case8(a=0.070, turn_deg=30.0):
    La = L - a
    jump_m = np.radians(180.0 + turn_deg)
    guess = [np.radians(-turn_deg / 2), 0.02, 0.0, 0.6 * min(a, La)]
    theta_r0_m, z0_m, v0_m, ell_m = _case8_crease_system(a, La, jump_m, guess)
    x, z_mountain = _case8_profile(a, La, theta_r0_m, z0_m, v0_m, ell_m, jump_m)

    print(f"case8  mountain: turn={turn_deg:.1f}deg a={a*1e3:.1f}mm "
          f"theta_r0={np.degrees(theta_r0_m):.4f}deg v0={v0_m:.3e} "
          f"ell={ell_m*1e3:.4f}mm z0={z0_m*1e3:.4f}mm")

    valley = solve_valley_case8(a, turn_deg)
    right_v, left_v = valley["right"], valley["left"]
    xr_v, zr_v = arm_xz(right_v, a, material_len=La)
    xl_v, zl_v = arm_xz(left_v, a, material_len=a)
    xv, z_valley = mirror_join(xr_v, zr_v, xl_v, zl_v)

    right_flat_mm = (La - right_v["ell"]) * 1e3 if right_v["branch"] == "flat" else 0.0
    z_min, z_max = z_valley.min(), z_valley.max()
    print(f"case8  valley: right branch={right_v['branch']:5s} v0={right_v['v0']:8.4f} "
          f"ell={right_v['ell']*1e3:.4f}mm; left branch={left_v['branch']:5s} "
          f"v0={left_v['v0']:8.4f} ell={left_v['ell']*1e3:.4f}mm. right flat="
          f"{right_flat_mm:.1f}mm of {La*1e3:.0f}mm ({right_flat_mm/(La*1e3)*100:.1f}%) "
          f"z=[{z_min*1e3:.4f},{z_max*1e3:.3f}]mm")
    assert z_min >= -1e-6, f"case8 valley: z went to {z_min*1e3:.4f}mm below the table"

    Xm, Ym, Zm = tile_y(x, z_mountain)
    Xv, Yv, Zv = tile_y(xv, z_valley)

    p1 = render(f"Case 8 mountain (/\\) — crease at x={a*1e3:.0f}mm, {turn_deg:.0f}° fold, "
                f"solved theta_r0={np.degrees(theta_r0_m):.1f}°",
               [(x, z_mountain)], ["profile"], Xm, Ym, Zm,
               "case8_mountain_70mm_30deg.png", support=(a, z0_m))
    p2 = render(f"Case 8 valley (V) — crease grounded at z=0, {turn_deg:.0f}° fold, "
                f"right side {right_flat_mm/(La*1e3)*100:.0f}% flat",
               [(xv, z_valley)], ["profile"], Xv, Yv, Zv,
               "case8_valley_70mm_30deg.png", support=(a, 0.0))
    return p1, p2


CASES = (case1, case2, case3, case4, case5, case6, case7, case8)


def solve_two_point_support(p1: float, h1: float, p2: float, h2: float,
                            q: float = Q, d: float = D, span: float = W,
                            theta0_guess: float = 0.0,
                            r1_guess: float | None = None) -> dict:
    if r1_guess is None:
        r1_guess = q * span / 2
    la, lb = p2 - p1, span - p2

    def residual(params):
        theta0, r1 = params
        r_tip1 = -q * p1
        r_tip2 = r1 - q * p2
        s1 = integrate(theta0, 0.0, 0.0, p1, q, d, r_tip=r_tip1)
        th1, v1, z1 = s1.y[:, -1]
        s2 = integrate(th1, v1, z1, la, q, d, r_tip=r_tip2)
        th2, v2, z2 = s2.y[:, -1]
        s3 = integrate(th2, v2, z2, lb, q, d, r_tip=0.0)
        return [(z2 - z1) - (h2 - h1), s3.y[1, -1]]

    theta0, r1 = fsolve(residual, [theta0_guess, r1_guess], xtol=1e-14, maxfev=20000)
    resid = residual([theta0, r1])
    if max(abs(r) for r in resid) > 1e-6:
        raise RuntimeError(f"case9 two-support shooting did not converge: {resid}")
    r2 = q * span - r1
    sol1 = integrate(theta0, 0.0, 0.0, p1, q, d, r_tip=-q * p1)
    th1, v1, z1 = sol1.y[:, -1]
    sol2 = integrate(th1, v1, z1, la, q, d, r_tip=r1 - q * p2)
    th2, v2, z2 = sol2.y[:, -1]
    sol3 = integrate(th2, v2, z2, lb, q, d, r_tip=0.0)
    return {"theta0": theta0, "r1": r1, "r2": r2, "z_shift": h1 - z1,
            "sol1": sol1, "sol2": sol2, "sol3": sol3,
            "p1": p1, "p2": p2, "span": span}


def _profile_xy(res: dict, n: int = 300):
    y0 = 0.0
    ys, zs = [], []
    for sol, ell in ((res["sol1"], res["p1"]),
                    (res["sol2"], res["p2"] - res["p1"]),
                    (res["sol3"], res["span"] - res["p2"])):
        sp = np.linspace(0.0, 1.0, n)
        theta, _, z = sol.sol(sp)
        ds = np.diff(sp) * ell
        y = y0 + np.concatenate([[0.0], np.cumsum(0.5 * (np.cos(theta[:-1]) + np.cos(theta[1:])) * ds)])
        ys.append(y)
        zs.append(z + res["z_shift"])
        y0 = y[-1]
    return np.concatenate(ys), np.concatenate(zs)


def case9(subcase: int = 1):
    if subcase == 1:
        p1, h1 = 0.020, 0.020
        p2, h2 = W - 0.020, 0.015
        title_bit = "20mm@20mm / 20mm-from-right@15mm"
    elif subcase == 2:
        p1, h1 = 0.090, 0.030
        p2, h2 = W - 0.080, 0.020
        title_bit = "90mm@30mm / 80mm-from-right@20mm"
    else:
        raise ValueError("subcase must be 1 or 2")

    res = solve_two_point_support(p1, h1, p2, h2)
    y, z = _profile_xy(res)
    print(f"case9.{subcase}  theta0={np.degrees(res['theta0']):8.4f}deg "
          f"R1={res['r1']:.5f} R2={res['r2']:.5f}  "
          f"z(p1)={(res['sol1'].y[2,-1]+res['z_shift'])*1e3:.4f}mm(want {h1*1e3:.1f}) "
          f"z(p2)={(res['sol2'].y[2,-1]+res['z_shift'])*1e3:.4f}mm(want {h2*1e3:.1f}) "
          f"z_min={z.min()*1e3:.3f}mm z_max={z.max()*1e3:.3f}mm")
    if z.min() < -1e-6:
        print(f"case9.{subcase}  FLAG: profile goes to {z.min()*1e3:.2f}mm, below "
              "the z=0 table, with both outer edges free — no physically valid "
              "resting branch found (flat touchdown never reaches theta=0 here, "
              "tip contact needs a negative/pulling reaction) — shown as computed.")

    X, Y, Z = tile_x(y, z)
    p_img = render(f"Case 9.{subcase} — two supports, {title_bit}",
                   [(y, z)], ["profile"], X, Y, Z, f"case9_{subcase}_two_support.png",
                   profile_xlabel="y [mm]")
    return p_img


CASES = (case1, case2, case3, case4, case5, case6, case7, case8, case9)

if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    for fn in CASES:
        print("saved", fn())
