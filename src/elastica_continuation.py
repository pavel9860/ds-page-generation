import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

D = 0.688e-3


def _rhs3(sp, y, ell, q, d):
    theta, v, _z = y
    shear = q * ell * (1.0 - sp)
    return [ell * v, ell * shear * np.cos(theta) / d, ell * np.sin(theta)]


def _M_end(v0, ell, q, d, theta0):
    sol = solve_ivp(_rhs3, (0.0, 1.0), [theta0, v0, 0.0], args=(ell, q, d),
                    method="RK45", rtol=1e-10, atol=1e-12)
    return sol.y[1, -1]


def _dv0_dq(q, v0, ell, d, theta0, h=1e-6):
    F0 = _M_end(v0, ell, q, d, theta0)
    Fv = (_M_end(v0 + h, ell, q, d, theta0) - F0) / h
    Fq = (_M_end(v0, ell, q + h, d, theta0) - F0) / h
    return [-Fq / Fv]


def solve_free_arm_continuation(theta0: float, ell_max: float, q_target: float,
                                d: float = D):
    sol = solve_ivp(lambda q, v0: _dv0_dq(q, v0[0], ell_max, d, theta0),
                    (0.0, q_target), [0.0], method="RK45",
                    rtol=1e-8, atol=1e-10, dense_output=True)
    v0 = sol.y[0, -1]
    final = solve_ivp(_rhs3, (0.0, 1.0), [theta0, v0, 0.0],
                      args=(ell_max, q_target, d), method="RK45",
                      rtol=1e-11, atol=1e-13, dense_output=True)
    return v0, final, sol


if __name__ == "__main__":
    import sys
    v0, sol, path = solve_free_arm_continuation(np.radians(30.0), 0.297, 0.7848)
    print("v0 =", v0, " M(ell) residual =", sol.y[1, -1])
    sp = np.linspace(0.0, 1.0, 2000)
    th = sol.sol(sp)[0]
    print("theta range deg:", np.degrees(th.min()), np.degrees(th.max()))


def _rhs9(sp, y, ell, q, d):
    theta, v, z, a, b, c, p, r, s = y
    shear = q * ell * (1.0 - sp)
    dshear_dq = ell * (1.0 - sp)
    ct, st = np.cos(theta), np.sin(theta)
    dtheta = ell * v
    dv = ell * shear * ct / d
    dz = ell * st
    da = ell * b
    db = -ell * shear * st * a / d
    dc = ell * ct * a
    dp = ell * r
    dr = ell * (dshear_dq * ct - shear * st * p) / d
    ds = ell * ct * p
    return [dtheta, dv, dz, da, db, dc, dp, dr, ds]


def _dv0_dq_analytic(q, v0, ell, d, theta0):
    y0 = [theta0, v0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    sol = solve_ivp(_rhs9, (0.0, 1.0), y0, args=(ell, q, d),
                    method="RK23", rtol=1e-4, atol=1e-6)
    M_v0 = sol.y[4, -1]
    M_q = sol.y[7, -1]
    return [-M_q / M_v0]


def solve_free_arm_analytic(theta0: float, ell_max: float, q_target: float,
                            d: float = D):
    sol = solve_ivp(lambda q, v0: _dv0_dq_analytic(q, v0[0], ell_max, d, theta0),
                    (0.0, q_target), [0.0], method="RK23",
                    rtol=1e-4, atol=1e-6)
    v0_fast = sol.y[0, -1]

    def M(p):
        return [solve_ivp(_rhs3, (0.0, 1.0), [theta0, p[0], 0.0],
                          args=(ell_max, q_target, d), rtol=1e-10, atol=1e-12).y[1, -1]]

    v0 = fsolve(M, [v0_fast], xtol=1e-9)[0]
    final = solve_ivp(_rhs3, (0.0, 1.0), [theta0, v0, 0.0],
                      args=(ell_max, q_target, d), method="RK45",
                      rtol=1e-10, atol=1e-12, dense_output=True)
    return v0, final


def _rhs6(sp, y, ell, q, d):
    theta, v, z, a, b, c = y
    shear = q * ell * (1.0 - sp)
    ct, st = np.cos(theta), np.sin(theta)
    return [ell * v, ell * shear * ct / d, ell * st,
            ell * b, -ell * shear * st * a / d, ell * ct * a]


def _dv0_dq_hybrid(q, v0, ell, d, theta0, h=1e-4):
    y0 = [theta0, v0, 0.0, 0.0, 1.0, 0.0]
    sol = solve_ivp(_rhs6, (0.0, 1.0), y0, args=(ell, q, d),
                    method="RK23", rtol=1e-4, atol=1e-6)
    M0 = sol.y[1, -1]
    M_v0 = sol.y[4, -1]
    solh = solve_ivp(_rhs3, (0.0, 1.0), [theta0, v0, 0.0], args=(ell, q + h, d),
                     method="RK23", rtol=1e-4, atol=1e-6)
    M_q = (solh.y[1, -1] - M0) / h
    return [-M_q / M_v0]


def solve_free_arm_hybrid(theta0: float, ell_max: float, q_target: float,
                          d: float = D):
    sol = solve_ivp(lambda q, v0: _dv0_dq_hybrid(q, v0[0], ell_max, d, theta0),
                    (0.0, q_target), [0.0], method="RK23",
                    rtol=1e-4, atol=1e-6)
    v0_fast = sol.y[0, -1]

    def M(p):
        return [solve_ivp(_rhs3, (0.0, 1.0), [theta0, p[0], 0.0],
                          args=(ell_max, q_target, d), rtol=1e-10, atol=1e-12).y[1, -1]]

    v0 = fsolve(M, [v0_fast], xtol=1e-9)[0]
    final = solve_ivp(_rhs3, (0.0, 1.0), [theta0, v0, 0.0],
                      args=(ell_max, q_target, d), method="RK45",
                      rtol=1e-10, atol=1e-12, dense_output=True)
    return v0, final


def _rk4_fixed_9(theta0, v0, ell, q, d, n_steps=20):
    h = 1.0 / n_steps
    y = np.array([theta0, v0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    sp = 0.0
    for _ in range(n_steps):
        k1 = np.array(_rhs9(sp, y, ell, q, d))
        k2 = np.array(_rhs9(sp + 0.5 * h, y + 0.5 * h * k1, ell, q, d))
        k3 = np.array(_rhs9(sp + 0.5 * h, y + 0.5 * h * k2, ell, q, d))
        k4 = np.array(_rhs9(sp + h, y + h * k3, ell, q, d))
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        sp += h
    return y


def _dv0_dq_fixedstep(q, v0, ell, d, theta0, n_steps=20):
    y = _rk4_fixed_9(theta0, v0, ell, q, d, n_steps)
    M_v0 = y[4]
    M_q = y[7]
    return [-M_q / M_v0]


def solve_free_arm_fixedstep(theta0: float, ell_max: float, q_target: float,
                             d: float = D, n_outer: int = 6, n_inner: int = 15):
    q_steps = np.linspace(0.0, q_target, n_outer + 1)
    v0 = 0.0
    h_q = q_target / n_outer
    for i in range(n_outer):
        qq = q_steps[i]
        k1 = _dv0_dq_fixedstep(qq, v0, ell_max, d, theta0, n_inner)[0]
        k2 = _dv0_dq_fixedstep(qq + 0.5 * h_q, v0 + 0.5 * h_q * k1, ell_max, d, theta0, n_inner)[0]
        k3 = _dv0_dq_fixedstep(qq + 0.5 * h_q, v0 + 0.5 * h_q * k2, ell_max, d, theta0, n_inner)[0]
        k4 = _dv0_dq_fixedstep(qq + h_q, v0 + h_q * k3, ell_max, d, theta0, n_inner)[0]
        v0 = v0 + (h_q / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    v0_fast = v0

    def M(p):
        return [solve_ivp(_rhs3, (0.0, 1.0), [theta0, p[0], 0.0],
                          args=(ell_max, q_target, d), rtol=1e-10, atol=1e-12).y[1, -1]]

    v0 = fsolve(M, [v0_fast], xtol=1e-9)[0]
    final = solve_ivp(_rhs3, (0.0, 1.0), [theta0, v0, 0.0],
                      args=(ell_max, q_target, d), method="RK45",
                      rtol=1e-10, atol=1e-12, dense_output=True)
    return v0, final


def _rk4_fixed_3(theta0, v0, ell, q, d, n_steps=60):
    h = 1.0 / n_steps
    y = np.array([theta0, v0, 0.0])
    sp = 0.0
    traj = np.empty((n_steps + 1, 3))
    traj[0] = y
    for i in range(n_steps):
        k1 = np.array(_rhs3(sp, y, ell, q, d))
        k2 = np.array(_rhs3(sp + 0.5 * h, y + 0.5 * h * k1, ell, q, d))
        k3 = np.array(_rhs3(sp + 0.5 * h, y + 0.5 * h * k2, ell, q, d))
        k4 = np.array(_rhs3(sp + h, y + h * k3, ell, q, d))
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        sp += h
        traj[i + 1] = y
    return traj


def solve_free_arm_nolib(theta0: float, ell_max: float, q_target: float,
                         d: float = D, n_outer: int = 3, n_inner: int = 8,
                         n_polish: int = 1, n_final: int = 20):
    q_steps = np.linspace(0.0, q_target, n_outer + 1)
    v0 = 0.0
    h_q = q_target / n_outer
    for i in range(n_outer):
        qq = q_steps[i]
        k1 = _dv0_dq_fixedstep(qq, v0, ell_max, d, theta0, n_inner)[0]
        k2 = _dv0_dq_fixedstep(qq + 0.5 * h_q, v0 + 0.5 * h_q * k1, ell_max, d, theta0, n_inner)[0]
        k3 = _dv0_dq_fixedstep(qq + 0.5 * h_q, v0 + 0.5 * h_q * k2, ell_max, d, theta0, n_inner)[0]
        k4 = _dv0_dq_fixedstep(qq + h_q, v0 + h_q * k3, ell_max, d, theta0, n_inner)[0]
        v0 = v0 + (h_q / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    for _ in range(n_polish):
        y = _rk4_fixed_9(theta0, v0, ell_max, q_target, d, n_steps=n_final)
        v0 = v0 - y[1] / y[4]

    traj = _rk4_fixed_3(theta0, v0, ell_max, q_target, d, n_steps=n_final)
    return v0, traj
