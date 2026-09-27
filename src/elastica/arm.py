import math

import numpy as np

from .newton import newton_step
from .ode import D, Q, dv0_dq, rk4_9, rk4_9_tip, rk4_12_flat, zmin
from .seeds import flat_seed, free_touches, tip_seed

N_OUTER = 6
N_INNER = 4
N_POLISH_STEPS = 8
N_STEPS_SP = 30


def solve_arm(theta0, ell, plane=0.0, flat_theta=0.0, q=Q, d=D):
    min_dev = math.radians(1.0)
    dev = theta0 - flat_theta
    if plane >= 0.0 and abs(dev) < min_dev:
        theta0 = flat_theta + (min_dev if dev >= 0 else -min_dev)

    m = math.cos(flat_theta)
    th = (theta0 - flat_theta) * m

    def F_of_flat(state):
        vv, g, re = state
        y = rk4_12_flat(theta0, vv, g, q, d, N_STEPS_SP, r_edge=re)
        theta1, v1, z1, a1, b1, c1, a2, b2, c2, a3, b3, c3 = y
        return (np.array([z1 - plane, theta1 - flat_theta, v1]),
                np.array([[c1, c2, c3], [a1, a2, a3], [b1, b2, b3]]))

    seed = flat_seed(th, m, -plane, q, d)
    if seed is not None:
        v0, g, r_edge = newton_step(F_of_flat, seed)
        if g <= ell:
            return {"branch": "flat", "theta0": theta0, "v0": v0, "ell": g, "r_tip": r_edge}

    def F_of_tip(state):
        vv, rr = state
        y = rk4_9_tip(theta0, vv, ell, q, d, N_STEPS_SP, r_tip=rr)
        theta1, v1, z1, a1, b1, c1, a2, b2, c2 = y
        return np.array([z1 - plane, v1]), np.array([[c1, c2], [b1, b2]])

    reachable = ell > plane
    if reachable and free_touches(th, -plane, ell, q, d):
        v0, r_tip = newton_step(F_of_tip, tip_seed(th, m, -plane, ell, q, d))
        if r_tip <= 0.0:
            return {"branch": "tip", "theta0": theta0, "v0": v0, "ell": ell, "r_tip": r_tip}

    q_steps = np.linspace(0.0, q, N_OUTER + 1)
    h_q = q / N_OUTER
    v0 = 0.0
    for i in range(N_OUTER):
        q0 = q_steps[i]
        k1 = dv0_dq(q0, v0, ell, d, theta0, N_INNER)[0]
        k2 = dv0_dq(q0 + 0.5 * h_q, v0 + 0.5 * h_q * k1, ell, d, theta0, N_INNER)[0]
        k3 = dv0_dq(q0 + 0.5 * h_q, v0 + 0.5 * h_q * k2, ell, d, theta0, N_INNER)[0]
        k4 = dv0_dq(q0 + h_q, v0 + h_q * k3, ell, d, theta0, N_INNER)[0]
        v0 = v0 + (h_q / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    y = rk4_9(theta0, v0, ell, q, d, n_steps=N_POLISH_STEPS)
    v0 = v0 - y[1] / y[4]
    if not reachable or zmin(theta0, v0, ell, q, d) >= plane - 1e-9:
        return {"branch": "free", "theta0": theta0, "v0": v0, "ell": ell}

    v0, r_tip = newton_step(F_of_tip, tip_seed(th, m, -plane, ell, q, d))
    return {"branch": "tip", "theta0": theta0, "v0": v0, "ell": ell, "r_tip": r_tip}
