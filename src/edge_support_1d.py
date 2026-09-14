"""1D heavy-elastica simulation of an A4 page draped over a knife-edge
ridge support running along the long side, centered on the short side.

Same ODE as crease_bend_solver.py (theta' = v, v' = load*cos(theta)/D,
z' = sin(theta), load = q * ell * (1 - sp) for a cantilever of material
length ell under uniform distributed weight q). The whole cross-section
(both sides of the ridge) is solved as ONE system: a single state
vector stacking both sides' (theta, v, z), a single rhs, and a single
fsolve call over all unknowns together -- not two separately-solved
pieces glued afterward.

`load = q*ell*(1-sp)` is the weight of material still hanging free
beyond a point, so `ell` must be that free segment's own (unknown)
length to its touchdown point, not the full material length to the
page edge -- otherwise the already-table-supported tail's weight would
wrongly enter the bending moment and the tail would lift back off the
table instead of staying flat. Material past a touchdown carries zero
net load (the table reaction cancels its own weight locally), so it
has nothing left to integrate: it is flat by construction.

Packaged as `PageBendProfile1D`, meant to be reused by the main
pipeline (src/synth/geometry.py) as a physically-solved alternative to
the analytic bend/fold height fields.
"""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

Q_DEFAULT = 0.7848        # distributed weight load (N/m^2), matches crease_bend_solver.py
D_DEFAULT = 0.2 * 0.688e-3  # bending stiffness; D/q ratio 80% below crease_bend_solver.py

PAGE_W_A4 = 0.210         # short side (m) -- the bend cross-section direction
PAGE_H_A4 = 0.297         # long side (m) -- the ridge's own direction


def _rhs_side(y, ell, q, D, sp):
    theta, v, z = y
    load = q * ell * (1 - sp)
    return [ell * v, ell * load * np.cos(theta) / D, ell * np.sin(theta)]


def _rhs_both(sp, y, ell_l, ell_r, q, D):
    dl = _rhs_side(y[0:3], ell_l, q, D, sp)
    dr = _rhs_side(y[3:6], ell_r, q, D, sp)
    return dl + dr


def _free_tip_v0(h, ell, q, D, v0_guess=0.0):
    """Pre-check only: does this side reach the table at all? Solves
    the single-unknown free-tip case (v0, natural BC v(1)=0) in
    isolation, just to seed/decide the combined solve below."""
    def rhs(sp, y):
        return _rhs_side(y, ell, q, D, sp)

    def residual(v0):
        sol = solve_ivp(rhs, [0, 1], [0.0, v0[0], h], method="RK45",
                         rtol=1e-10, atol=1e-13)
        return [sol.y[1, -1]]

    v0, _, _, _ = fsolve(residual, [v0_guess], full_output=True, xtol=1e-12, maxfev=5000)
    sol = solve_ivp(rhs, [0, 1], [0.0, v0[0], h], method="RK45",
                     rtol=1e-10, atol=1e-13, dense_output=True)
    z_min = sol.sol(np.linspace(0, 1, 200))[2].min()
    return v0[0], z_min < 0.0


@dataclass
class PageBendProfile1D:
    """Full short-side cross-section: a ridge of height `ridge_h` at
    `ridge_x`, both sides solved together as one system. `x`/`z` run
    the full page width, x=0 at one page edge to x=page_w at the
    other."""
    page_w: float
    ridge_x: float
    ridge_h: float
    q: float
    D: float
    touched_down_left: bool
    touched_down_right: bool
    ell_left: float   # touchdown length (== ridge_x if never touches)
    ell_right: float  # touchdown length (== page_w - ridge_x if never touches)
    x: np.ndarray
    z: np.ndarray

    def z_of_x(self, x_query):
        return np.interp(x_query, self.x, self.z)


def build_a4_edge_support_profile(ridge_h=0.030, page_w=PAGE_W_A4, ridge_x=None,
                                   q=Q_DEFAULT, D=D_DEFAULT, n=400):
    """Solve the whole cross-section (both sides of the ridge) as one
    combined nonlinear system. `ell_max_l`/`ell_max_r` are the material
    lengths from the ridge to each page edge; `touches_l`/`touches_r`
    (decided by a cheap single-side free-tip pre-check) select, per
    side, between the touchdown BC (theta=0, z=0 at an unknown length
    ell <= ell_max) and the free-tip BC (v=0 at ell_max)."""
    if ridge_x is None:
        ridge_x = page_w / 2
    ell_max_l = ridge_x
    ell_max_r = page_w - ridge_x

    v0_l_guess, touches_l = _free_tip_v0(ridge_h, ell_max_l, q, D)
    v0_r_guess, touches_r = _free_tip_v0(ridge_h, ell_max_r, q, D)

    ell_l_guess = 0.6 * ell_max_l if touches_l else ell_max_l
    ell_r_guess = 0.6 * ell_max_r if touches_r else ell_max_r

    def unpack(p):
        i = 0
        v0_l = p[i]; i += 1
        ell_l = p[i] if touches_l else ell_max_l
        i += 1 if touches_l else 0
        v0_r = p[i]; i += 1
        ell_r = p[i] if touches_r else ell_max_r
        i += 1 if touches_r else 0
        return v0_l, ell_l, v0_r, ell_r

    p0 = [v0_l_guess] + ([ell_l_guess] if touches_l else []) + \
         [v0_r_guess] + ([ell_r_guess] if touches_r else [])

    def residual(p):
        v0_l, ell_l, v0_r, ell_r = unpack(p)
        sol = solve_ivp(_rhs_both, [0, 1], [0.0, v0_l, ridge_h, 0.0, v0_r, ridge_h],
                         args=(ell_l, ell_r, q, D), method="RK45",
                         rtol=1e-11, atol=1e-13)
        th_l, v_l, z_l, th_r, v_r, z_r = sol.y[:, -1]
        out = []
        out.append(z_l if touches_l else v_l)
        if touches_l:
            out.append(th_l)
        out.append(z_r if touches_r else v_r)
        if touches_r:
            out.append(th_r)
        return out

    p_sol, _, ier, _ = fsolve(residual, p0, full_output=True, xtol=1e-13, maxfev=8000)
    v0_l, ell_l, v0_r, ell_r = unpack(p_sol)

    sol = solve_ivp(_rhs_both, [0, 1], [0.0, v0_l, ridge_h, 0.0, v0_r, ridge_h],
                     args=(ell_l, ell_r, q, D), method="RK45",
                     rtol=1e-11, atol=1e-13, dense_output=True)
    sp = np.linspace(0, 1, n)
    th_l, v_l, z_l, th_r, v_r, z_r = sol.sol(sp)

    def profile_side(theta, z, ell, ell_max, touched):
        x = np.concatenate([[0.0],
                             np.cumsum((np.cos(theta[:-1]) + np.cos(theta[1:])) / 2 * np.diff(sp) * ell)])
        if touched and ell < ell_max - 1e-9:
            n_tail = max(2, int(round((ell_max - ell) / ell_max * n)))
            x_tail = x[-1] + np.linspace(0, ell_max - ell, n_tail)[1:]
            x = np.concatenate([x, x_tail])
            z = np.concatenate([z, np.zeros_like(x_tail)])
        return x, z

    x_l, z_l_full = profile_side(th_l, z_l, ell_l, ell_max_l, touches_l)
    x_r, z_r_full = profile_side(th_r, z_r, ell_r, ell_max_r, touches_r)

    x = np.concatenate([ridge_x - x_l[::-1], ridge_x + x_r[1:]])
    z = np.concatenate([z_l_full[::-1], z_r_full[1:]])

    return PageBendProfile1D(page_w=page_w, ridge_x=ridge_x, ridge_h=ridge_h, q=q, D=D,
                              touched_down_left=touches_l, touched_down_right=touches_r,
                              ell_left=ell_l, ell_right=ell_r, x=x, z=z)


if __name__ == "__main__":
    profile = build_a4_edge_support_profile()
    print(f"ridge_h={profile.ridge_h*1000:.1f}mm at x={profile.ridge_x*1000:.1f}mm")
    print(f"  left:  ell={profile.ell_left*1000:.2f}mm of {profile.ridge_x*1000:.2f}mm material, "
          f"touched_down={profile.touched_down_left}")
    print(f"  right: ell={profile.ell_right*1000:.2f}mm of {(profile.page_w-profile.ridge_x)*1000:.2f}mm material, "
          f"touched_down={profile.touched_down_right}")
    print(f"z range: {profile.z.min()*1000:.3f}..{profile.z.max()*1000:.3f} mm")
