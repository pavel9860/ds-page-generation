import numpy as np
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from skfem import (Basis, BilinearForm, ElementTriMorley, ElementTriP1, ElementTriP2, ElementVector,
                   LinearForm, MeshQuad, bmat, condense, solve)
from skfem.helpers import dd, ddot, grad, sym_grad, trace


def _outer_sym(a, b):
    return 0.5 * (np.einsum("i...,j...->ij...", a, b) + np.einsum("i...,j...->ij...", b, a))


def solve_sag(u, v, z0, span, paper, nodes):
    """Sag w(u, v): Marguerre shallow shell about z0 under gravity, paper = config.paper_props. The solved
    profile is held (w = 0, zero slope) over the support strip v in span and where it lies on the table."""
    m = MeshQuad.init_tensor(np.linspace(u[0], u[-1], nodes[0]),
                             np.linspace(v[0], v[-1], nodes[1])).to_meshtri(style="x")
    ba = Basis(m, ElementVector(ElementTriP2()))
    bw = Basis(m, ElementTriMorley(), quadrature=ba.quadrature)
    bz = Basis(m, ElementTriP1(), quadrature=ba.quadrature)
    x, y = m.p
    z_nodes = RegularGridInterpolator((v, u), z0)(np.column_stack([y, x]))
    d, nu = paper["bending"], paper["poisson"]
    c = paper["membrane"] / (1.0 - nu ** 2)

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
        return -paper["q"] * w2

    z = bz.interpolate(z_nodes)
    kaw = k_wa.assemble(bw, ba, z=z)
    K = bmat([[k_ww.assemble(bw, z=z), kaw.T], [kaw, k_aa.assemble(ba)]], "csr")
    f = np.concatenate([load.assemble(bw), np.zeros(ba.N)])

    tol = 0.25 * (v[-1] - v[0]) / (nodes[1] - 1)
    in_span = (y >= span[0] - tol) & (y <= span[1] + tol)
    held = in_span | (z_nodes < 1e-6)
    strip = np.flatnonzero(in_span[m.facets].all(axis=0))
    fixed = [bw.get_dofs(nodes=np.flatnonzero(held)).all("u"), bw.get_dofs(facets=strip).all("u_n")]
    a0 = np.argmin(np.hypot(x - u[0], y - 0.5 * (v[0] + v[-1])))
    a1 = np.argmin(np.hypot(x - u[-1], y - 0.5 * (v[0] + v[-1])))
    fixed += [ba.get_dofs(nodes=np.array([a0])).all() + bw.N, ba.get_dofs(nodes=np.array([a1])).all("u^2") + bw.N]
    s = 1.0 / np.sqrt(np.abs(K.diagonal()))
    Ks = K.multiply(s[:, None]).multiply(s[None, :]).tocsr()
    sol = s * solve(*condense(Ks, f * s, D=np.unique(np.concatenate(fixed))))
    U, V = np.meshgrid(u, v)
    return LinearNDInterpolator(m.p.T, sol[:bw.N][bw.nodal_dofs[0]])(U, V)
