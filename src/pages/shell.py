import numpy as np
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from skfem import (Basis, BilinearForm, ElementTriMorley, ElementTriP1, ElementTriP2, ElementVector,
                   LinearForm, MeshQuad, bmat, condense, solve)
from skfem.helpers import dd, ddot, grad, sym_grad, trace


def _outer_sym(a, b):
    return 0.5 * (np.einsum("i...,j...->ij...", a, b) + np.einsum("i...,j...->ij...", b, a))


def solve_w(u, v, z0, lines, clamp, paper, nodes, spine=True):
    """Marguerre shallow shell linearized about z0(u, v) under gravity.
    w = 0 on the table (z0 = 0), on support segments (u, v_lo, v_hi), on the clamp
    edge u = 0 over clamp = (v_lo, v_hi) (with zero slope), and, with spine, along
    the solved profile at the middle generator."""
    m = MeshQuad.init_tensor(np.linspace(u[0], u[-1], nodes[0]),
                             np.linspace(v[0], v[-1], nodes[1])).to_meshtri(style="x")
    ba = Basis(m, ElementVector(ElementTriP2()))
    bw = Basis(m, ElementTriMorley(), quadrature=ba.quadrature)
    bz = Basis(m, ElementTriP1(), quadrature=ba.quadrature)
    x, y = m.p
    z_nodes = RegularGridInterpolator((v, u), z0)(np.column_stack([y, x]))
    d, nu = paper.bending_stiffness, paper.poisson
    c = paper.membrane_stiffness / (1.0 - nu ** 2)

    def membrane(e1, e2):
        return c * ((1.0 - nu) * ddot(e1, e2) + nu * trace(e1) * trace(e2))

    @BilinearForm
    def k_ww(w1, w2, p):
        bend = d * ((1.0 - nu) * ddot(dd(w1), dd(w2)) + nu * trace(dd(w1)) * trace(dd(w2)))
        return bend + membrane(_outer_sym(grad(p["z"]), grad(w1)), _outer_sym(grad(p["z"]), grad(w2)))

    @BilinearForm
    def k_aa(a1, a2, _):
        return membrane(sym_grad(a1), sym_grad(a2))

    @BilinearForm
    def k_wa(w1, a2, p):
        return membrane(_outer_sym(grad(p["z"]), grad(w1)), sym_grad(a2))

    @LinearForm
    def load(w2, _):
        return -paper.q * w2

    z = bz.interpolate(z_nodes)
    kaw = k_wa.assemble(bw, ba, z=z)
    K = bmat([[k_ww.assemble(bw, z=z), kaw.T], [kaw, k_aa.assemble(ba)]], "csr")
    f = np.concatenate([load.assemble(bw), np.zeros(ba.N)])

    hx = (u[-1] - u[0]) / (nodes[0] - 1)
    hy = (v[-1] - v[0]) / (nodes[1] - 1)
    held = z_nodes < 1e-6
    for pu, v_lo, v_hi in lines:
        held |= (np.abs(x - pu) <= 0.5 * hx) & (y >= v_lo - 0.5 * hy) & (y <= v_hi + 0.5 * hy)
    if spine:
        held |= np.abs(y - 0.5 * (v[0] + v[-1])) <= 0.25 * hy
    fixed = [bw.get_dofs(nodes=np.flatnonzero(held)).all("u")]
    if clamp is not None:
        facets = m.facets_satisfying(lambda s: (s[0] < u[0] + 1e-12) & (s[1] >= clamp[0] - 0.5 * hy)
                                     & (s[1] <= clamp[1] + 0.5 * hy))
        fixed += [bw.get_dofs(nodes=np.unique(m.facets[:, facets])).all("u"),
                  bw.get_dofs(facets=facets).all("u_n")]
    a0 = np.argmin(np.hypot(x - u[0], y - 0.5 * (v[0] + v[-1])))
    a1 = np.argmin(np.hypot(x - u[-1], y - 0.5 * (v[0] + v[-1])))
    fixed += [ba.get_dofs(nodes=np.array([a0])).all() + bw.N, ba.get_dofs(nodes=np.array([a1])).all("u^2") + bw.N]
    s = 1.0 / np.sqrt(np.abs(K.diagonal()))
    Ks = K.multiply(s[:, None]).multiply(s[None, :]).tocsr()
    sol = s * solve(*condense(Ks, f * s, D=np.unique(np.concatenate(fixed))))
    U, V = np.meshgrid(u, v)
    return LinearNDInterpolator(m.p.T, sol[:bw.N][bw.nodal_dofs[0]])(U, V)
