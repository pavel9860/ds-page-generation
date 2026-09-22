import numpy as np
from scipy.optimize import least_squares

from elastica_continuation import D, _rk4_fixed_3, _dv0_dq_fixedstep, _rk4_fixed_9

N_OUTER, N_INNER, N_POLISH, N_POLISH_STEPS = 3, 8, 1, 20
N_STEPS_SP = 30


def _zmin_theta_end(theta0, v0, frac, ell, q, d):
    traj = _rk4_fixed_3(theta0, v0, frac * ell, q, d, n_steps=N_STEPS_SP)
    return traj, traj[:, 2].min()


def _tip_traj(theta0, v0, r_tip, ell, q, d):
    traj = _rk4_fixed_3(theta0, v0, ell, q, d, n_steps=N_STEPS_SP, r_tip=r_tip)
    return traj, traj[:, 2].min()


def solve_arm_fast(theta0, ell, plane=0.0, flat_theta=0.0, q=0.7848, d=D):
    """The same continuation as free_v0_fast (n_outer=3 RK4 steps in q, one
    Newton polish), with free/tip/flat switching happening inline inside
    that one loop -- not a separate pre-check plus a different fallback
    loop. Once a step's trajectory dips below the plane, the remaining
    outer steps track (v0, r_tip) [tip] or (v0, frac) [flat] instead of
    plain v0, each step warm-started from the previous."""
    min_dev = np.radians(1.0)
    dev = theta0 - flat_theta
    if abs(dev) < min_dev:
        theta0 = flat_theta + (min_dev if dev >= 0 else -min_dev)

    q_steps = np.linspace(0.0, q, N_OUTER + 1)
    h_q = q / N_OUTER
    v0, r_tip, frac = 0.0, 0.0, 1.0
    phase = "free"

    for i in range(N_OUTER):
        qq = q_steps[i + 1]

        if phase == "free":
            q0 = q_steps[i]
            k1 = _dv0_dq_fixedstep(q0, v0, ell, d, theta0, N_INNER)[0]
            k2 = _dv0_dq_fixedstep(q0 + 0.5 * h_q, v0 + 0.5 * h_q * k1, ell, d, theta0, N_INNER)[0]
            k3 = _dv0_dq_fixedstep(q0 + 0.5 * h_q, v0 + 0.5 * h_q * k2, ell, d, theta0, N_INNER)[0]
            k4 = _dv0_dq_fixedstep(q0 + h_q, v0 + h_q * k3, ell, d, theta0, N_INNER)[0]
            v0 = v0 + (h_q / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if i == N_OUTER - 1:
                y = _rk4_fixed_9(theta0, v0, ell, qq, d, n_steps=N_POLISH_STEPS)
                v0 = v0 - y[1] / y[4]

            traj, zmin = _zmin_theta_end(theta0, v0, 1.0, ell, qq, d)
            if zmin < plane - 1e-9:
                phase = "tip"
            else:
                continue

        if phase == "tip":
            def res_tip(p, qq_=qq):
                t, _ = _tip_traj(theta0, p[0], p[1], ell, qq_, d)
                return [t[-1, 2] - plane, t[-1, 1]]

            fit = least_squares(res_tip, [v0, r_tip], xtol=1e-11, ftol=1e-11)
            v0t, r_tipt = fit.x
            t, zmin = _tip_traj(theta0, v0t, r_tipt, ell, qq, d)
            if zmin >= plane - 1e-9:
                v0, r_tip = v0t, r_tipt
                continue
            phase = "flat"
            frac = float(np.linspace(0.0, 1.0, N_STEPS_SP + 1)[np.argmin(t[:, 2])])
            frac = min(max(frac, 1e-3), 1.0)
            v0 = v0t

        def res_flat(p, qq_=qq):
            t, _ = _zmin_theta_end(theta0, p[0], p[1], ell, qq_, d)
            return [t[-1, 2] - plane, t[-1, 0] - flat_theta]

        fit = least_squares(res_flat, [v0, frac], bounds=([-np.inf, 1e-6], [np.inf, 1.0]),
                            xtol=1e-11, ftol=1e-11)
        v0, frac = fit.x

    if phase == "flat":
        return {"branch": "flat", "v0": v0, "ell": frac * ell}
    if phase == "tip":
        return {"branch": "tip", "v0": v0, "ell": ell, "r_tip": r_tip}
    return {"branch": "free", "v0": v0, "ell": ell}
