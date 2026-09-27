import math

import numpy as np

from .newton import newton_step
from .ode import D, Q, rk4_3

N_SEG_STEPS = 60
N_CHECK_STEPS = 300
N_PROFILE_STEPS = 800
N_NEWTON = 80


def _segment_rk4_sens(theta0, v0, z0, sens0, ell, q, d, r_tip, dshear_dp,
                       n_steps=N_SEG_STEPS):
    n_params = sens0.shape[1]
    y = np.concatenate([[theta0, v0, z0], sens0.T.reshape(-1)])
    dshear_dp = np.asarray(dshear_dp, dtype=float)

    def rhs(sp, y):
        theta, v = y[0], y[1]
        shear = q * ell * (1.0 - sp) + r_tip
        ct, st = math.cos(theta), math.sin(theta)
        out = np.empty_like(y)
        out[0] = ell * v
        out[1] = ell * shear * ct / d
        out[2] = ell * st
        a = y[3::3]
        b = y[4::3]
        c = y[5::3]
        out[3::3] = ell * b
        out[4::3] = ell * (dshear_dp * ct - shear * st * a) / d
        out[5::3] = ell * ct * a
        return out

    h = 1.0 / n_steps
    sp = 0.0
    for _ in range(n_steps):
        k1 = rhs(sp, y)
        k2 = rhs(sp + 0.5 * h, y + 0.5 * h * k1)
        k3 = rhs(sp + 0.5 * h, y + 0.5 * h * k2)
        k4 = rhs(sp + h, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        sp += h

    theta, v, z = y[0], y[1], y[2]
    sens = y[3:].reshape(n_params, 3).T
    return theta, v, z, sens


def _zmin_interior(theta0, v0, ell, q, d, r_tip, n_steps=N_CHECK_STEPS, ext_offset=0.0):
    traj = rk4_3(theta0, v0, ell, q, d, n_steps=n_steps, r_tip=r_tip)
    z = traj[1:, 2] + ext_offset
    idx = int(np.argmin(z))
    return z[idx], (idx + 1) / n_steps


def solve_two_support(p1, h1, p2, h2, span, q=Q, d=D,
                                  theta0_guess=0.0, r1_guess=None):
    if r1_guess is None:
        r1_guess = q * span / 2.0

    la, lb = p2 - p1, span - p2
    r_tip1 = -q * p1

    def solve_free():
        def F_of(state):
            theta0, r1 = state
            r_tip2 = r1 + r_l - q * p2
            sens0 = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
            th1, v1, z1, sens1 = _segment_rk4_sens(theta0, 0.0, 0.0, sens0, p1, q, d,
                                                    r_tip1, dshear_dp=[0.0, 0.0])
            th2, v2, z2, sens2 = _segment_rk4_sens(th1, v1, z1, sens1, la, q, d,
                                                    r_tip2, dshear_dp=[0.0, 1.0])
            _th3, v3, _z3, sens3 = _segment_rk4_sens(th2, v2, z2, sens2, lb, q, d,
                                                      0.0, dshear_dp=[0.0, 0.0])
            F = np.array([(z2 - z1) - (h2 - h1), v3])
            J = np.array([sens2[2, :] - sens1[2, :], sens3[1, :]])
            return F, J

        return newton_step(F_of, np.array([theta0_guess, r1_guess]), n_iter=N_NEWTON)

    def solve_left_tip(theta0_seed, r1_seed):
        def F_of(state):
            theta0, r_l, r1 = state
            sens0 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            th1, v1, z1, sens1 = _segment_rk4_sens(theta0, 0.0, 0.0, sens0, p1, q, d,
                                                    r_l - q * p1, dshear_dp=[0.0, 1.0, 0.0])
            th2, v2, z2, sens2 = _segment_rk4_sens(th1, v1, z1, sens1, la, q, d,
                                                    r_l + r1 - q * p2, dshear_dp=[0.0, 1.0, 1.0])
            _th3, v3, _z3, sens3 = _segment_rk4_sens(th2, v2, z2, sens2, lb, q, d,
                                                      0.0, dshear_dp=[0.0, 0.0, 0.0])
            F = np.array([z1 - h1, (z2 - z1) - (h2 - h1), v3])
            J = np.array([sens1[2, :], sens2[2, :] - sens1[2, :], sens3[1, :]])
            return F, J

        return newton_step(F_of, np.array([theta0_seed, 0.0, r1_seed]), n_iter=N_NEWTON)

    def solve_tip(theta0_seed, r1_seed):
        def F_of(state):
            theta0, r1, r_tipC = state
            r_tip2 = r1 + r_l - q * p2
            sens0 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            th1, v1, z1, sens1 = _segment_rk4_sens(theta0, 0.0, 0.0, sens0, p1, q, d,
                                                    r_tip1, dshear_dp=[0.0, 0.0, 0.0])
            th2, v2, z2, sens2 = _segment_rk4_sens(th1, v1, z1, sens1, la, q, d,
                                                    r_tip2, dshear_dp=[0.0, 1.0, 0.0])
            _th3, v3, z3, sens3 = _segment_rk4_sens(th2, v2, z2, sens2, lb, q, d,
                                                     r_tipC, dshear_dp=[0.0, 0.0, 1.0])
            F = np.array([(z2 - z1) - (h2 - h1), v3, z3 - (z1 - h1)])
            J = np.array([sens2[2, :] - sens1[2, :],
                          sens3[1, :],
                          sens3[2, :] - sens1[2, :]])
            return F, J

        return newton_step(F_of, np.array([theta0_seed, r1_seed, 0.0]), n_iter=N_NEWTON)

    def solve_flat(theta0_seed, r1_seed, r_edge_seed, frac_seed):
        def raw_end_state(theta0, r1, fracC, r_edgeC):
            r_tip2 = r1 + r_l - q * p2
            sens0 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            th1, v1, z1, sens1 = _segment_rk4_sens(theta0, 0.0, 0.0, sens0, p1, q, d,
                                                    r_tip1, dshear_dp=[0.0, 0.0, 0.0])
            th2, v2, z2, sens2 = _segment_rk4_sens(th1, v1, z1, sens1, la, q, d,
                                                    r_tip2, dshear_dp=[0.0, 1.0, 0.0])
            gC = fracC * lb
            th3, v3, z3, sens3 = _segment_rk4_sens(th2, v2, z2, sens2, gC, q, d,
                                                    r_edgeC, dshear_dp=[0.0, 0.0, 1.0])
            F = np.array([(z2 - z1) - (h2 - h1), z3 - (z1 - h1), th3, v3])
            Jsub = np.array([sens2[2, :] - sens1[2, :],
                             sens3[2, :] - sens1[2, :],
                             sens3[0, :],
                             sens3[1, :]])
            return F, Jsub

        def F_of(state):
            theta0, r1, r_edgeC, u = state
            fracC = 1.0 / (1.0 + math.exp(-u))
            F, Jsub = raw_end_state(theta0, r1, fracC, r_edgeC)
            h = 1e-6
            du = math.exp(-u) / (1.0 + math.exp(-u)) ** 2
            F_hi, _ = raw_end_state(theta0, r1, fracC + h * du, r_edgeC)
            F_lo, _ = raw_end_state(theta0, r1, fracC - h * du, r_edgeC)
            col_u = (F_hi - F_lo) / (2.0 * h)
            J = np.column_stack([Jsub, col_u])
            return F, J

        u0 = math.log(max(frac_seed, 1e-6) / (1.0 - max(frac_seed, 1e-6)))
        theta0, r1, r_edgeC, u = newton_step(
            F_of, np.array([theta0_seed, r1_seed, r_edge_seed, u0]), n_iter=N_NEWTON)
        return theta0, r1, r_edgeC, 1.0 / (1.0 + math.exp(-u))

    def solve_mid(theta0_seed, r1_seed, g_seed):
        def raw(theta0, r1, g, rc):
            sc = p1 + g
            sens0 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            th1, v1, z1, sens1 = _segment_rk4_sens(theta0, 0.0, 0.0, sens0, p1, q, d,
                                                    r_tip1, dshear_dp=[0.0, 0.0, 0.0])
            thc, vc, zc, sensc = _segment_rk4_sens(th1, v1, z1, sens1, g, q, d,
                                                    r1 + r_l - q * sc, dshear_dp=[0.0, 1.0, 0.0])
            th2, v2, z2, sens2 = _segment_rk4_sens(thc, vc, zc, sensc, p2 - sc, q, d,
                                                    r1 + r_l + rc - q * p2, dshear_dp=[0.0, 1.0, 1.0])
            _th3, v3, _z3, sens3 = _segment_rk4_sens(th2, v2, z2, sens2, lb, q, d,
                                                      0.0, dshear_dp=[0.0, 0.0, 0.0])
            F = np.array([(z2 - z1) - (h2 - h1), v3, zc - z1 + h1, thc])
            J = np.array([sens2[2, :] - sens1[2, :], sens3[1, :], sensc[2, :] - sens1[2, :], sensc[0, :]])
            return F, J

        def F_of(state):
            theta0, r1, g, rc = state
            F, Jsub = raw(theta0, r1, g, rc)
            h = 1e-7
            col_g = (raw(theta0, r1, g + h, rc)[0] - raw(theta0, r1, g - h, rc)[0]) / (2.0 * h)
            return F, np.column_stack([Jsub[:, :2], col_g, Jsub[:, 2]])

        return newton_step(F_of, np.array([theta0_seed, r1_seed, g_seed, 0.0]), n_iter=N_NEWTON)

    r_l = 0.0
    theta0, r1 = solve_free()
    v0 = 0.0
    branch_left = "free"

    sol1 = rk4_3(theta0, 0.0, p1, q, d, n_steps=N_CHECK_STEPS, r_tip=r_tip1)
    z1 = sol1[-1, 2]
    zmin_left, _ = _zmin_interior(theta0, 0.0, p1, q, d, r_tip1, ext_offset=h1 - z1)
    if zmin_left < -1e-7:
        theta0, r_l, r1 = solve_left_tip(theta0, r1)
        r_tip1 = r_l - q * p1
        branch_left = "tip"

    sol1 = rk4_3(theta0, v0, p1, q, d, n_steps=N_CHECK_STEPS, r_tip=r_tip1)
    th1, v1, z1 = sol1[-1]
    r_tip2 = r1 + r_l - q * p2
    sol2 = rk4_3(th1, v1, la, q, d, n_steps=N_CHECK_STEPS, r_tip=r_tip2)
    th2, v2, z2 = sol2[-1]

    zmin_right, _ = _zmin_interior(th2, v2, lb, q, d, 0.0, ext_offset=h1 + z2)
    branch_right = "free"
    r_tipC = 0.0
    fracC = 1.0
    if zmin_right < -1e-7:
        theta0, r1, r_tipC = solve_tip(theta0, r1)
        r_tip2 = r1 + r_l - q * p2
        sol1 = rk4_3(theta0, v0, p1, q, d, n_steps=N_CHECK_STEPS, r_tip=r_tip1)
        th1, v1, z1 = sol1[-1]
        sol2 = rk4_3(th1, v1, la, q, d, n_steps=N_CHECK_STEPS, r_tip=r_tip2)
        th2, v2, z2 = sol2[-1]
        branch_right = "tip"
        zmin_tip, sp_at_zmin_tip = _zmin_interior(th2, v2, lb, q, d, r_tipC, ext_offset=h1 + z2)
        if zmin_tip < -1e-7:
            frac_seed = min(max(sp_at_zmin_tip, 1e-3), 1.0 - 1e-3)
            theta0, r1, r_edgeC, fracC = solve_flat(theta0, r1, r_tipC, frac_seed)
            r_tip2 = r1 + r_l - q * p2
            r_tipC = r_edgeC
            sol1 = rk4_3(theta0, v0, p1, q, d, n_steps=N_CHECK_STEPS, r_tip=r_tip1)
            th1, v1, z1 = sol1[-1]
            sol2 = rk4_3(th1, v1, la, q, d, n_steps=N_CHECK_STEPS, r_tip=r_tip2)
            th2, v2, z2 = sol2[-1]
            branch_right = "flat"

    zmin_mid, sp_mid = _zmin_interior(th1, v1, la, q, d, r_tip2, ext_offset=h1)
    g_c, r_c = None, 0.0
    if zmin_mid < -1e-7 and branch_right == "free":
        theta0, r1, g_c, r_c = solve_mid(theta0, r1, sp_mid * la)
        r_tip2 = r1 + r_l - q * p2

    sol1 = rk4_3(theta0, v0, p1, q, d, n_steps=N_PROFILE_STEPS, r_tip=r_tip1)
    th1, v1, z1 = sol1[-1]
    if g_c is None:
        sol2 = rk4_3(th1, v1, la, q, d, n_steps=N_PROFILE_STEPS, r_tip=r_tip2)
    else:
        n_a = max(1, round(N_PROFILE_STEPS * g_c / la))
        sa = rk4_3(th1, v1, g_c, q, d, n_steps=n_a, r_tip=r1 + r_l - q * (p1 + g_c))
        sb = rk4_3(*sa[-1, :2], la - g_c, q, d, n_steps=max(1, N_PROFILE_STEPS - n_a), r_tip=r1 + r_l + r_c - q * p2)
        sb[:, 2] += sa[-1, 2]
        sol2 = np.vstack([sa, sb[1:]])
    th2, v2, z2 = sol2[-1]
    ell_curved_C = fracC * lb
    sol3_curved = rk4_3(th2, v2, ell_curved_C, q, d, n_steps=N_PROFILE_STEPS, r_tip=r_tipC)
    th3, v3, z3 = sol3_curved[-1]

    sol2 = sol2.copy(); sol2[:, 2] += z1
    sol3_curved = sol3_curved.copy(); sol3_curved[:, 2] += z1 + z2
    segments = [sol1, sol2, sol3_curved]
    seg_ell = [p1, la, ell_curved_C]
    if branch_right == "flat" and ell_curved_C < lb - 1e-9:
        flat_len = lb - ell_curved_C
        n_flat = max(int(N_PROFILE_STEPS * flat_len / lb), 2)
        flat_traj = np.zeros((n_flat, 3))
        flat_traj[:, 2] = z1 + z2 + z3
        segments.append(flat_traj)
        seg_ell.append(flat_len)

    r2 = q * span - r1 - r_l - r_c
    z_shift = h1 - z1
    return {"theta0": theta0, "v0": v0, "r1": r1, "r2": r2, "z_shift": z_shift,
            "p1": p1, "p2": p2, "span": span,
            "segments": tuple(segments), "seg_ell": tuple(seg_ell),
            "branch_right": branch_right, "r_tipC": r_tipC, "fracC": fracC,
            "branch_left": branch_left, "r_left": r_l, "mid_contact": g_c, "r_mid": r_c}


def profile_xy(res, n=300):
    support_y = [None, None]
    y0 = 0.0
    ys, zs = [], []
    for i, (sol, ell) in enumerate(zip(res["segments"], res["seg_ell"])):
        n_steps = sol.shape[0] - 1
        if n_steps == 0:
            ys.append(np.array([y0, y0 + ell]))
            zs.append(np.array([sol[0, 2], sol[0, 2]]))
            y0 = y0 + ell
        else:
            idx = np.linspace(0, n_steps, n).round().astype(int)
            theta = sol[idx, 0]
            z = sol[idx, 2]
            sp = idx / n_steps
            ds = np.diff(sp) * ell
            y = y0 + np.concatenate([[0.0], np.cumsum(0.5 * (np.cos(theta[:-1]) + np.cos(theta[1:])) * ds)])
            ys.append(y)
            zs.append(z + res["z_shift"])
            y0 = y[-1]
        if i == 0:
            support_y[0] = y0
        elif i == 1:
            support_y[1] = y0
    return np.concatenate(ys), np.concatenate(zs), tuple(support_y)
