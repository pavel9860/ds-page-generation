import math

import numpy as np

Q = 0.7848
D = 0.688e-3


def rhs3(sp, theta, v, ell, q, d, r_tip=0.0):
    shear = q * ell * (1.0 - sp) + r_tip
    return ell * v, ell * shear * math.cos(theta) / d, ell * math.sin(theta)


def rk4_3(theta0, v0, ell, q, d, n_steps=60, r_tip=0.0):
    h = 1.0 / n_steps
    theta, v, z = theta0, v0, 0.0
    sp = 0.0
    traj = np.empty((n_steps + 1, 3))
    traj[0] = (theta, v, z)
    for i in range(n_steps):
        k1t, k1v, k1z = rhs3(sp, theta, v, ell, q, d, r_tip)
        k2t, k2v, k2z = rhs3(sp + 0.5 * h, theta + 0.5 * h * k1t, v + 0.5 * h * k1v, ell, q, d, r_tip)
        k3t, k3v, k3z = rhs3(sp + 0.5 * h, theta + 0.5 * h * k2t, v + 0.5 * h * k2v, ell, q, d, r_tip)
        k4t, k4v, k4z = rhs3(sp + h, theta + h * k3t, v + h * k3v, ell, q, d, r_tip)
        theta += (h / 6.0) * (k1t + 2 * k2t + 2 * k3t + k4t)
        v += (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        z += (h / 6.0) * (k1z + 2 * k2z + 2 * k3z + k4z)
        sp += h
        traj[i + 1] = (theta, v, z)
    return traj


def rhs9(sp, y, ell, q, d):
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


def rk4_9(theta0, v0, ell, q, d, n_steps=20):
    h = 1.0 / n_steps
    y = np.array([theta0, v0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    sp = 0.0
    for _ in range(n_steps):
        k1 = np.array(rhs9(sp, y, ell, q, d))
        k2 = np.array(rhs9(sp + 0.5 * h, y + 0.5 * h * k1, ell, q, d))
        k3 = np.array(rhs9(sp + 0.5 * h, y + 0.5 * h * k2, ell, q, d))
        k4 = np.array(rhs9(sp + h, y + h * k3, ell, q, d))
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        sp += h
    return y


def dv0_dq(q, v0, ell, d, theta0, n_steps=20):
    y = rk4_9(theta0, v0, ell, q, d, n_steps)
    M_v0 = y[4]
    M_q = y[7]
    return [-M_q / M_v0]


def zmin(theta0, v0, ell, q, d, r_tip=0.0, n_steps=30):
    h = 1.0 / n_steps
    theta, v, z = theta0, v0, 0.0
    sp = 0.0
    lowest = None
    for _ in range(n_steps):
        k1t, k1v, k1z = rhs3(sp, theta, v, ell, q, d, r_tip)
        k2t, k2v, k2z = rhs3(sp + 0.5 * h, theta + 0.5 * h * k1t, v + 0.5 * h * k1v, ell, q, d, r_tip)
        k3t, k3v, k3z = rhs3(sp + 0.5 * h, theta + 0.5 * h * k2t, v + 0.5 * h * k2v, ell, q, d, r_tip)
        k4t, k4v, k4z = rhs3(sp + h, theta + h * k3t, v + h * k3v, ell, q, d, r_tip)
        theta += (h / 6.0) * (k1t + 2 * k2t + 2 * k3t + k4t)
        v += (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        z += (h / 6.0) * (k1z + 2 * k2z + 2 * k3z + k4z)
        sp += h
        if lowest is None or z < lowest:
            lowest = z
    return lowest


def rhs9_tip(sp, y, ell, q, d, r_tip):
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


def rk4_9_tip(theta0, v0, ell, q, d, n_steps, r_tip=0.0):
    h = 1.0 / n_steps
    y = [theta0, v0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    sp = 0.0
    for _ in range(n_steps):
        k1 = rhs9_tip(sp, y, ell, q, d, r_tip)
        y2 = [y[i] + 0.5 * h * k1[i] for i in range(9)]
        k2 = rhs9_tip(sp + 0.5 * h, y2, ell, q, d, r_tip)
        y3 = [y[i] + 0.5 * h * k2[i] for i in range(9)]
        k3 = rhs9_tip(sp + 0.5 * h, y3, ell, q, d, r_tip)
        y4 = [y[i] + h * k3[i] for i in range(9)]
        k4 = rhs9_tip(sp + h, y4, ell, q, d, r_tip)
        y = [y[i] + (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(9)]
        sp += h
    return y


def rhs12_flat(sp, y, g, q, d, r_edge):
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


def rk4_12_flat(theta0, v0, g, q, d, n_steps, r_edge=0.0):
    h = 1.0 / n_steps
    y = [theta0, v0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    sp = 0.0
    for _ in range(n_steps):
        k1 = rhs12_flat(sp, y, g, q, d, r_edge)
        y2 = [y[i] + 0.5 * h * k1[i] for i in range(12)]
        k2 = rhs12_flat(sp + 0.5 * h, y2, g, q, d, r_edge)
        y3 = [y[i] + 0.5 * h * k2[i] for i in range(12)]
        k3 = rhs12_flat(sp + 0.5 * h, y3, g, q, d, r_edge)
        y4 = [y[i] + h * k3[i] for i in range(12)]
        k4 = rhs12_flat(sp + h, y4, g, q, d, r_edge)
        y = [y[i] + (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(12)]
        sp += h
    return y
