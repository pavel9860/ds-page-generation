import math

import numpy as np

from elastica_continuation import D, _dv0_dq_fixedstep, _rk4_fixed_9

N_OUTER, N_INNER, N_POLISH, N_POLISH_STEPS = 3, 8, 1, 20
N_STEPS_SP = 30


def free_v0_fast(theta0, ell, q, d=D):
    q_steps = np.linspace(0.0, q, N_OUTER + 1)
    v0 = 0.0
    h_q = q / N_OUTER
    for i in range(N_OUTER):
        qq = q_steps[i]
        k1 = _dv0_dq_fixedstep(qq, v0, ell, d, theta0, N_INNER)[0]
        k2 = _dv0_dq_fixedstep(qq + 0.5 * h_q, v0 + 0.5 * h_q * k1, ell, d, theta0, N_INNER)[0]
        k3 = _dv0_dq_fixedstep(qq + 0.5 * h_q, v0 + 0.5 * h_q * k2, ell, d, theta0, N_INNER)[0]
        k4 = _dv0_dq_fixedstep(qq + h_q, v0 + h_q * k3, ell, d, theta0, N_INNER)[0]
        v0 = v0 + (h_q / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    for _ in range(N_POLISH):
        y = _rk4_fixed_9(theta0, v0, ell, q, d, n_steps=N_POLISH_STEPS)
        v0 = v0 - y[1] / y[4]
    return v0


def _zmin(theta0, v0, ell, q, d, r_tip=0.0, n_steps=30):
    zmin, _ = _zmin_loc(theta0, v0, ell, q, d, r_tip, n_steps)
    return zmin


def _zmin_loc(theta0, v0, ell, q, d, r_tip=0.0, n_steps=30):
    # zmin/sp_min are tracked only over the integrated interior points, sp
    # in (0, 1] -- the sp=0 anchor is the clamp itself (z=0 by definition,
    # not a touchdown candidate), and including it as the initial "best"
    # falsely wins the comparison whenever plane >= 0, hiding the genuine
    # interior dip that is the real touchdown location solve_arm_analytic
    # seeds `frac` from.
    h = 1.0 / n_steps
    theta, v, z = theta0, v0, 0.0
    sp = 0.0
    zmin = sp_min = None
    for _ in range(n_steps):
        def rhs(sp_, th, vv):
            shear = q * ell * (1.0 - sp_) + r_tip
            return ell * vv, ell * shear * math.cos(th) / d, ell * math.sin(th)
        k1t, k1v, k1z = rhs(sp, theta, v)
        k2t, k2v, k2z = rhs(sp + 0.5 * h, theta + 0.5 * h * k1t, v + 0.5 * h * k1v)
        k3t, k3v, k3z = rhs(sp + 0.5 * h, theta + 0.5 * h * k2t, v + 0.5 * h * k2v)
        k4t, k4v, k4z = rhs(sp + h, theta + h * k3t, v + h * k3v)
        theta += (h / 6.0) * (k1t + 2 * k2t + 2 * k3t + k4t)
        v += (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        z += (h / 6.0) * (k1z + 2 * k2z + 2 * k3z + k4z)
        sp += h
        if zmin is None or z < zmin:
            zmin, sp_min = z, sp
    return zmin, sp_min


# --------------------------------------------------------------------------- #
# Shared 9-state system: (theta,v,z) + d/dx1 (v0, always) + d/dx2 (second
# unknown: r_tip for tip, g=frac*ell for flat). q-sensitivity dropped --
# ∂F/∂q is instead taken by one cheap finite-difference re-integration.
# --------------------------------------------------------------------------- #
def _rhs9_tip(sp, y, ell, q, d, r_tip):
    theta, v, z, a1, b1, c1, a2, b2, c2 = y
    shear = q * ell * (1.0 - sp) + r_tip
    ct, st = math.cos(theta), math.sin(theta)
    da1 = ell * b1
    db1 = -ell * shear * st * a1 / d
    dc1 = ell * ct * a1
    da2 = ell * b2
    db2 = ell * (ct - shear * st * a2) / d
    dc2 = ell * ct * a2
    return (ell * v, ell * shear * ct / d, ell * st, da1, db1, dc1, da2, db2, dc2)


def _rk4_9_tip(theta0, v0, ell, q, d, n_steps, r_tip=0.0):
    h = 1.0 / n_steps
    y = [theta0, v0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    sp = 0.0
    for _ in range(n_steps):
        k1 = _rhs9_tip(sp, y, ell, q, d, r_tip)
        y2 = [y[i] + 0.5 * h * k1[i] for i in range(9)]
        k2 = _rhs9_tip(sp + 0.5 * h, y2, ell, q, d, r_tip)
        y3 = [y[i] + 0.5 * h * k2[i] for i in range(9)]
        k3 = _rhs9_tip(sp + 0.5 * h, y3, ell, q, d, r_tip)
        y4 = [y[i] + h * k3[i] for i in range(9)]
        k4 = _rhs9_tip(sp + h, y4, ell, q, d, r_tip)
        y = [y[i] + (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(9)]
        sp += h
    return y


def _rhs12_flat(sp, y, g, q, d, r_edge):
    # A touchdown point where the curved part only matches the plane's
    # position and slope (z=plane, theta=flat_theta) leaves its bending
    # moment (v) free -- generically nonzero, which physically means an
    # external point moment/force sitting exactly at the touchdown point
    # that nothing in the model supplies. The flat segment beyond it has
    # zero curvature identically, so a genuine smooth (no-external-moment)
    # contact requires v=0 there too. That's a 3rd condition, so it needs
    # a 3rd unknown: r_edge, a concentrated force AT the touchdown point
    # (exactly the same mechanism the tip branch already uses for r_tip,
    # just relocated from the true tip to the free/flat boundary).
    theta, v, z, a1, b1, c1, a2, b2, c2, a3, b3, c3 = y
    shear = q * g * (1.0 - sp) + r_edge
    dshear_dg = q * (1.0 - sp)
    ct, st = math.cos(theta), math.sin(theta)
    da1 = g * b1
    db1 = -g * shear * st * a1 / d
    dc1 = g * ct * a1
    da2 = v + g * b2
    db2 = (shear * ct + g * (dshear_dg * ct - shear * st * a2)) / d
    dc2 = st + g * ct * a2
    da3 = g * b3
    db3 = g * (ct - shear * st * a3) / d
    dc3 = g * ct * a3
    return (g * v, g * shear * ct / d, g * st,
            da1, db1, dc1, da2, db2, dc2, da3, db3, dc3)


def _rk4_12_flat(theta0, v0, g, q, d, n_steps, r_edge=0.0):
    h = 1.0 / n_steps
    y = [theta0, v0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    sp = 0.0
    for _ in range(n_steps):
        k1 = _rhs12_flat(sp, y, g, q, d, r_edge)
        y2 = [y[i] + 0.5 * h * k1[i] for i in range(12)]
        k2 = _rhs12_flat(sp + 0.5 * h, y2, g, q, d, r_edge)
        y3 = [y[i] + 0.5 * h * k2[i] for i in range(12)]
        k3 = _rhs12_flat(sp + 0.5 * h, y3, g, q, d, r_edge)
        y4 = [y[i] + h * k3[i] for i in range(12)]
        k4 = _rhs12_flat(sp + h, y4, g, q, d, r_edge)
        y = [y[i] + (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(12)]
        sp += h
    return y


def _newton_step(F_of, state, clamp=lambda s: s, tol=1e-22, n_iter=80):
    """Trust-region Gauss-Newton on F_of(state) -> (F, J), minimizing
    ||F||^2. The tip/flat Jacobians are ill-conditioned near a short
    contact segment (frac or r_tip -> 0) or a near-singular 2x2 (small
    det(J)); an LM step damped only through lam*diag(J^T J) still has no
    bound on its raw length in that regime and can jump clean out of the
    physical basin into a spurious root (e.g. theta wrapping past +-360
    deg). Bounding the step to a trust radius `delta`, adapted by the gain
    ratio (actual vs the linear model's predicted reduction), is the
    textbook fix: it keeps every step inside the region where the local
    linearization is trusted, so it can't overshoot into another root the
    way an unbounded damped step can. Iterates to convergence (cost stops
    improving) rather than a fixed count, so easy cases exit in a handful
    of steps."""
    state = clamp(state)
    F, J = F_of(state)
    cost = float(np.dot(F, F))
    delta = 1.0
    for _ in range(n_iter):
        if cost < tol:
            break
        gn_step, *_ = np.linalg.lstsq(J, F, rcond=None)
        gn_norm = np.linalg.norm(gn_step)
        step = gn_step if gn_norm <= delta else gn_step * (delta / gn_norm)

        trial = clamp(state - step)
        F_t, J_t = F_of(trial)
        cost_t = float(np.dot(F_t, F_t))

        predicted = cost - float(np.dot(F - J @ step, F - J @ step))
        actual = cost - cost_t
        rho = actual / predicted if predicted > 0 else -1.0

        if rho > 0.0:
            state, F, J, cost = trial, F_t, J_t, cost_t
        if rho > 0.75:
            delta *= 2.0
        elif rho < 0.25:
            delta *= 0.25
    return state


def solve_arm_analytic(theta0, ell, plane=0.0, flat_theta=0.0, q=0.7848, d=D):
    """Same continuation as free_v0_fast (N_OUTER RK4 steps in q, one Newton
    polish), with free/tip/flat switching happening inline inside that one
    loop -- see case5_solver.solve_arm_fast, which this mirrors exactly
    except Newton (using the analytic Jacobians from _rk4_9_tip/_rk4_12_flat)
    replaces least_squares for the tip/flat corrections."""
    min_dev = math.radians(1.0)
    dev = theta0 - flat_theta
    if abs(dev) < min_dev:
        theta0 = flat_theta + (min_dev if dev >= 0 else -min_dev)

    q_steps = np.linspace(0.0, q, N_OUTER + 1)
    h_q = q / N_OUTER
    v0, r_tip, frac, r_edge = 0.0, 0.0, 1.0, 0.0
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

            if _zmin(theta0, v0, ell, qq, d) < plane - 1e-9:
                phase = "tip"
            else:
                continue

        if phase == "tip":
            def F_of_tip(state):
                vv, rr = state
                y = _rk4_9_tip(theta0, vv, ell, qq, d, N_STEPS_SP, r_tip=rr)
                theta1, v1, z1, a1, b1, c1, a2, b2, c2 = y
                return np.array([z1 - plane, v1]), np.array([[c1, c2], [b1, b2]])

            v0, r_tip = _newton_step(F_of_tip, np.array([v0, r_tip]))

            zmin, sp_min = _zmin_loc(theta0, v0, ell, qq, d, r_tip=r_tip, n_steps=N_STEPS_SP)
            if zmin >= plane - 1e-9:
                continue
            phase = "flat"
            frac = min(max(sp_min, 1e-3), 1.0)
            r_edge = r_tip

        # frac lives in (0, 1); solving for it directly needs a hard clamp
        # at the bounds, and once a step lands exactly on a clamped bound
        # every further trial re-evaluates at that same point -- the gain
        # ratio the trust region relies on becomes ill-defined and it can
        # only shrink, stalling. Solving instead for u = logit(frac) makes
        # frac = sigmoid(u) unconstrained in u: no bound ever exists to
        # stall against, and du/dfrac's chain rule is exact, not an
        # approximation. r_edge is the touchdown-point force from
        # _rk4_12_flat's derivation (v1 is now a residual too).
        def F_of_flat_u(state):
            vv, u, re = state
            ff = 1.0 / (1.0 + math.exp(-u))
            y = _rk4_12_flat(theta0, vv, ff * ell, qq, d, N_STEPS_SP, r_edge=re)
            theta1, v1, z1, a1, b1, c1, a2, b2, c2, a3, b3, c3 = y
            dff_du = ff * (1.0 - ff)
            return (np.array([z1 - plane, theta1 - flat_theta, v1]),
                    np.array([[c1, c2 * ell * dff_du, c3],
                              [a1, a2 * ell * dff_du, a3],
                              [b1, b2 * ell * dff_du, b3]]))

        frac = min(max(frac, 1e-6), 1.0 - 1e-6)
        u0 = math.log(frac / (1.0 - frac))
        v0, u, r_edge = _newton_step(F_of_flat_u, np.array([v0, u0, r_edge]))
        frac = 1.0 / (1.0 + math.exp(-u))

    if phase == "flat":
        # r_tip (not r_edge) on purpose: any caller integrating this arm's
        # curved part (0 to "ell") consumes this exactly like the tip
        # branch's r_tip -- same shear-equation role, just relocated from
        # the true tip to the touchdown point. A branch-specific key name
        # is precisely what let a stale "r_tip"-only caller silently fall
        # back to 0.0 for flat arms and plot an unsupported, lifted curve.
        return {"branch": "flat", "v0": v0, "ell": frac * ell, "r_tip": r_edge}
    if phase == "tip":
        return {"branch": "tip", "v0": v0, "ell": ell, "r_tip": r_tip}
    return {"branch": "free", "v0": v0, "ell": ell}
