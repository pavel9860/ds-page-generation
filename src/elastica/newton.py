import numpy as np


def newton_step(F_of, state, tol=1e-22, n_iter=10):
    F, J = F_of(state)
    cost = float(np.dot(F, F))
    delta = 1.0
    for _ in range(n_iter):
        if cost < tol:
            break
        gn_step, *_ = np.linalg.lstsq(J, F, rcond=None)
        gn_norm = np.linalg.norm(gn_step)
        step = gn_step if gn_norm <= delta else gn_step * (delta / gn_norm)

        trial = state - step
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
