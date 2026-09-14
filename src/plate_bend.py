"""
2D page deformation built from the 1D heavy-elastica bend (crease_bend_
solver.py): a page draped over a rigid ridge, solved cross-section by
cross-section, no sag coupling between cross-sections.

The ridge runs along the page's long axis (y) at x=ridge_x, with
prescribed height h(y). Each cross-section (fixed y, scanning x) is an
independent inextensible elastica under distributed weight q with
stiffness D: symmetric about the ridge (theta=0, z=h there), each half
running out to its own page edge, which carries no support. Two BVPs,
tried in order, per half:
  1. Free-tip: fixed material length (ridge to page edge), natural BC
     v=0 (zero moment/shear) at the tip -- shooting unknown is v0 alone.
  2. If that solution dips below the table (z<0 anywhere), the edge
     must actually be touching: re-solve with touchdown BCs theta=0,
     z=0 at an unknown contact length ell < the page edge -- shooting
     unknowns are (v0, ell) -- and hold flat (z=0) from there to the edge.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

Q = 0.7848        # distributed weight load (matches crease_bend_solver.py)
D_STIFF = 0.688e-3  # bending stiffness (matches crease_bend_solver.py)


def _rhs(sp, y, ell, q, D):
    theta, v, z = y
    load = q * ell * (1 - sp)
    return [ell * v, ell * load * np.cos(theta) / D, ell * np.sin(theta)]


def solve_half_section(h, guess, q=Q, D=D_STIFF):
    """One symmetric half cross-section: ridge (sp=0, theta=0, z=h, v=v0
    unknown) arcing to touchdown (sp=1, theta=0, z=0). Solves for
    (v0, ell). Returns (v0, ell, sol) with sol a dense_output solve_ivp
    result over sp in [0,1]."""
    def residual(p):
        v0, ell = p
        sol = solve_ivp(_rhs, [0, 1], [0.0, v0, h], args=(ell, q, D),
                         method="RK45", rtol=1e-10, atol=1e-13)
        th1, v1, z1 = sol.y[:, -1]
        return [th1, z1]

    sol_p = fsolve(residual, guess, xtol=1e-12, maxfev=2000)
    v0, ell = sol_p
    ell = abs(ell)
    sol = solve_ivp(_rhs, [0, 1], [0.0, v0, h], args=(ell, q, D),
                     method="RK45", rtol=1e-10, atol=1e-13, dense_output=True)
    return v0, ell, sol


def solve_free_hang(h, ell_fixed, v0_guess, q=Q, D=D_STIFF):
    """Cantilever of FIXED material length ell_fixed hanging from the ridge
    (sp=0, theta=0, z=h) with a genuinely free tip at sp=1 (no support
    there at all -- natural BC v(1)=0, i.e. zero moment/shear at the tip,
    not a forced touchdown). Used for the page edges, where there is no
    support and the page may simply end lifted above the table, or may
    dip down and touch it before the tip -- either is a valid outcome
    the physics should decide, not something we force."""
    def residual(v0):
        sol = solve_ivp(_rhs, [0, 1], [0.0, v0[0], h], args=(ell_fixed, q, D),
                         method="RK45", rtol=1e-10, atol=1e-13)
        return [sol.y[1, -1]]

    v0, _, ier, _ = fsolve(residual, [v0_guess], xtol=1e-12, maxfev=2000, full_output=True)
    if ier != 1:
        v0, _, ier, _ = fsolve(residual, [0.0], xtol=1e-12, maxfev=2000, full_output=True)
    v0 = v0[0]
    sol = solve_ivp(_rhs, [0, 1], [0.0, v0, h], args=(ell_fixed, q, D),
                     method="RK45", rtol=1e-10, atol=1e-13, dense_output=True)
    return v0, sol


def _cumulative_x(theta, ell, n):
    ds = ell / (n - 1)
    return np.concatenate([[0.0], np.cumsum((np.cos(theta[:-1]) + np.cos(theta[1:])) / 2 * ds)])


def cross_section_profile(h, x_center, x_half_extent, n=400, guess=(0.0, 0.05),
                           q=Q, D=D_STIFF):
    """Full x-profile z(x) for one cross-section: page of half-width
    x_half_extent on each side of x_center, lifted to height h at
    x_center. There is no support at x_center +- x_half_extent (the page
    edges), so first solve the free-tip BVP (natural BC v=0, no
    touchdown assumed) over the full fixed material length x_half_extent.
    Only if that solution dips to z<0 does the page actually reach the
    table -- in that case re-solve as the touchdown BVP (unknown contact
    length ell, BCs theta=0 and z=0 there), flat beyond. Returns (x, z,
    v0, ell)."""
    if h <= 1e-9:
        x = np.linspace(x_center - x_half_extent, x_center + x_half_extent, n)
        return x, np.zeros_like(x), 0.0, 0.0

    sp = np.linspace(0, 1, n // 2)
    v0, sol = solve_free_hang(h, x_half_extent, guess[0], q, D)
    theta, z = sol.sol(sp)[[0, 2]]

    if z.min() >= -1e-9:
        # genuinely free edge: never reaches the table
        dx = _cumulative_x(theta, x_half_extent, len(sp))
        x_half = x_center + dx
        z_half = z
        ell = x_half_extent
    else:
        # the free-tip solution penetrates the table: the page actually
        # touches down at an unknown length ell < x_half_extent -- solve
        # that BVP instead (theta=0, z=0 at the touchdown), flat after it
        v0, ell, sol2 = solve_half_section(h, guess, q, D)
        theta, z = sol2.sol(sp)[[0, 2]]
        dx = _cumulative_x(theta, ell, len(sp))
        x_arc = x_center + dx
        x_flat = np.linspace(x_arc[-1], x_center + x_half_extent, 2)[1:]
        x_half = np.concatenate([x_arc, x_flat])
        z_half = np.concatenate([z, np.zeros_like(x_flat)])

    # mirror about x_center for the other side
    x_left = 2 * x_center - x_half[::-1]
    z_left = z_half[::-1]
    x_full = np.concatenate([x_left, x_half[1:]])
    z_full = np.concatenate([z_left, z_half[1:]])
    return x_full, z_full, v0, ell


def edge_support_page(page_w, page_h, ridge_x, peak_height, ny=61, nx=200,
                       q=Q, D=D_STIFF):
    """Page of size (page_w x page_h) lying on z=0, lifted by a ridge that
    runs along y at x=ridge_x, with height profile
        h(y) = peak_height * sin(pi * y / page_h)
    (zero at y=0 and y=page_h, peak at the mid-length). Returns
    (X, Y, Z) grids of shape (ny, nx)."""
    x_half_extent = max(ridge_x, page_w - ridge_x)
    y = np.linspace(0, page_h, ny)
    h_of_y = peak_height * np.sin(np.pi * y / page_h)

    x_common = np.linspace(0, page_w, nx)
    Z = np.zeros((ny, nx))
    guess = (0.0, 0.3 * peak_height + 0.02)
    # continue from the peak (max h, most nonlinear) outward to both ends
    center_i = ny // 2
    order = list(range(center_i, ny)) + list(range(center_i - 1, -1, -1))
    for i in order:
        h = h_of_y[i]
        x_prof, z_prof, v0, ell = cross_section_profile(
            h, ridge_x, x_half_extent, n=400, guess=guess, q=q, D=D)
        if h > 1e-9:
            guess = (v0, ell)
        x_prof = np.clip(x_prof, 0, page_w)
        Z[i] = np.interp(x_common, x_prof, z_prof)
    X, Y = np.meshgrid(x_common, y)
    return X, Y, Z
