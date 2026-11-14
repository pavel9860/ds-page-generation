"""Cockling of water-damaged paper: multi-scale random waviness added along the surface normal of a page grid.

Target = the band table of one real water-damaged page (cockle_targets.json: rms and 1/e correlation lengths
across / along of Z per octave band Z_a - Z_b, Gaussian sigma (a, b) nodes). The page keeps its own shape; the added
field brings each band up to the target energy:

    h = crop(irfft2(N * sqrt(sum_k c_k phi_k))),   phi_k = exp(-ln(rho / f_k)^2 / (2 W^2)),   rho = sqrt(fx^2 + (e fy)^2)

N = spectrum of white noise on the grid padded by PAD nodes (no wrap-around correlation in the crop); phi_k = NBUMP
log-spaced radial bumps of the power spectrum, e = elongation along the page height (target Ly / Lx over that of
isotropic noise, interpolated from the band centres to f_k, snapped to E_GRID). The band variance of the cropped
field is linear in c: var_j = sum_k gain[e_k, j, k] c_k, gain = expected band variance of the cropped field of bump
k alone (Monte Carlo once per grid shape, cached). c >= 0 is one NNLS fit (relative error) of
var_j = target_j^2 - present_j^2 (present = band energy the page already has; cross terms with independent noise
are zero in expectation). One rfft2 / irfft2 pair per page. Grids are (rows, cols, 3) [mm], rows down the height."""
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import nnls

from ..noise import smooth_noise

BANDS = np.array(((1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64)))   # Gaussian sigma, nodes
SIGMAS = np.unique(BANDS)
PAD = 64                                                                   # nodes: ~ the coarsest sigma
NBUMP, W_BUMP = 16, 0.45                                                   # power-spectrum bumps, log width
F_BUMP = np.geomspace(1 / (2 * np.pi * 96), 0.35, NBUMP)                   # cycles / node, below .. above the bands
F_BAND = 1 / (2 * np.pi * np.sqrt(BANDS.prod(1)))                          # band centres, cycles / node
E_GRID = np.geomspace(1.0, 4.0, 7)                                         # elongations with a gain table
MC = 16                                                                    # noise realisations per gain entry


@lru_cache(1)
def targets():
    """Band tables of the real water-damaged pages: dicts of arrays rms, lx, ly (per band, mm)."""
    t = json.loads(Path(__file__).with_name("cockle_targets.json").read_text())
    assert np.array_equal(t["bands"], BANDS)
    return [{k: np.asarray(p[k], float) for k in ("rms", "lx", "ly")} for p in t["pages"]]


@lru_cache(1)
def pitch():
    """(mm across, mm down) per node of the grid the band tables were measured on."""
    return tuple(json.loads(Path(__file__).with_name("cockle_targets.json").read_text())["step_mm"])


LEVELS = ("low", "medium", "high")                                         # tertiles of the real pages' cockle rms


def level_of(t):
    """Level of a band table: its cockle rms (bands 2..16 nodes) against the tertiles of the real pages."""
    r = lambda x: float(np.sqrt((x["rms"][1:4] ** 2).sum()))
    q = np.quantile([r(x) for x in targets()], (1 / 3, 2 / 3))
    return LEVELS[int(np.searchsorted(q, r(t)))]


def sample(rng, level=None):
    """One real page's band table (all pages of the level equally likely; level None: all pages)."""
    p = [t for t in targets() if level is None or level_of(t) == level]
    return p[rng.integers(len(p))]


def mask(shape, rng, cover, region, edge=0.15):
    """Wet part of the page in [0, 1]: a smooth random field (correlation length region nodes) above its (1 - cover)
    quantile, soft edge of `edge` field std; cover >= 1: the whole page."""
    if cover >= 1:
        return np.ones(shape, np.float32)
    f = smooth_noise(rng, shape, region)
    return (1 / (1 + np.exp(-(f - np.quantile(f, 1 - cover)) / edge))).astype(np.float32)


def bands(Z):
    """(len(BANDS), rows, cols) band-passed copies of Z: Gaussian blurs (edge-replicated = scipy mode "nearest",
    truncate 4), one per distinct sigma."""
    Z = np.ascontiguousarray(Z, np.float64)
    g = np.stack([cv2.GaussianBlur(Z, (2 * int(4 * s + 0.5) + 1,) * 2, float(s), borderType=cv2.BORDER_REPLICATE) for s in SIGMAS])
    i = np.searchsorted(SIGMAS, BANDS)
    return g[i[:, 0]] - g[i[:, 1]]


def corr_len(x, ax):
    """1/e correlation lengths along axis ax (1 or 2) of a stack x (n, rows, cols), nodes."""
    n = x.shape[ax]
    f = np.fft.rfft(x - x.mean(ax, keepdims=True), n=2 * n, axis=ax)
    c = np.fft.irfft(np.abs(f) ** 2, axis=ax).mean(3 - ax)[:, : n // 2]
    return np.argmax(c / c[:, :1] < np.exp(-1), axis=1).astype(float)


def measure(Z, step):
    """Band table of a height map Z (rows, cols) [mm]; step = (mm across, mm down) per node."""
    b = bands(Z)
    return dict(rms=b.std((1, 2)), lx=corr_len(b, 2) * step[0], ly=corr_len(b, 1) * step[1])


@lru_cache(4)
def _freq(shape):
    return np.fft.fftfreq(shape[0])[:, None] ** 2, np.fft.rfftfreq(shape[1])[None] ** 2


def _phi(shape, e):
    """(NBUMP, rows, cols // 2 + 1) power bumps on a grid of this shape; e = per-bump elongation."""
    fy, fx = _freq(shape)
    rho = np.sqrt(fx[None] + (e[:, None, None] ** 2) * fy[None]) + 1e-12
    return np.exp(-np.log(rho / F_BUMP[:, None, None]) ** 2 / (2 * W_BUMP ** 2))


def _pad(shape):
    return shape[0] + 2 * PAD, shape[1] + 2 * PAD


def _crop(h, shape):
    return h[..., PAD:PAD + shape[0], PAD:PAD + shape[1]]


@lru_cache(4)
def _iso_ratio(shape):
    """Ly / Lx per band of isotropic white noise on this grid (the estimator's own anisotropy)."""
    m = measure(np.random.default_rng(0).standard_normal(shape), (1.0, 1.0))
    return m["ly"] / np.maximum(m["lx"], 1e-9)


GAIN_FILE = Path(__file__).with_name("cockle_gain.npz")                     # precomputed tables, key "rows x cols"


@lru_cache(4)
def _gain(shape):
    """(len(E_GRID), len(BANDS), NBUMP) expected band variance of the cropped field of each unit bump: from
    GAIN_FILE when it holds this shape, else computed (~40 s)."""
    key = f"{shape[0]}x{shape[1]}"
    if GAIN_FILE.exists():
        with np.load(GAIN_FILE) as f:
            if key in f.files:
                return f[key]
    return _gain_mc(shape)


def _gain_mc(shape):
    P = _pad(shape)
    N = np.fft.rfft2(np.random.default_rng(0).standard_normal((MC, 1) + P))         # (MC, 1, *spectrum)
    out = np.empty((len(E_GRID), len(BANDS), NBUMP))
    for i, e in enumerate(E_GRID):
        h = _crop(np.fft.irfft2(N * np.sqrt(_phi(P, np.full(NBUMP, e))), s=P), shape)  # (MC, NBUMP, rows, cols)
        out[i] = np.stack([bands(x).var((1, 2)) for x in h.reshape(-1, *shape)]).reshape(MC, NBUMP, -1).mean(0).T
    return out


def field(shape, rng, target, present=None, scale=1.0):
    """Height field [mm] of shape (rows, cols) whose band rms is sqrt(target^2 - present^2) in expectation.
    target / present: band tables (present None = flat page); scale multiplies the target rms (amount of water)."""
    shape = tuple(shape)
    P = _pad(shape)
    e_band = np.maximum(1.0, (target["ly"] / np.maximum(target["lx"], 1e-9)) / _iso_ratio(shape))
    e = np.interp(np.log(F_BUMP), np.log(F_BAND[::-1]), e_band[::-1])
    ie = np.abs(np.log(e)[:, None] - np.log(E_GRID)[None]).argmin(1)                # nearest gain-table elongation
    M = _gain(shape)[ie, :, np.arange(NBUMP)].T                                     # (bands, bumps)
    d = np.maximum((scale * target["rms"]) ** 2 - (0.0 if present is None else present["rms"] ** 2), 0.0)
    r = np.maximum(d, 1e-6 * d.max() + 1e-12)                                      # relative error; empty bands stay ~0
    A = M / r[:, None]
    col = np.maximum(np.linalg.norm(A, axis=0), 1e-30)                             # bump energies span decades
    c = nnls(A / col, d / r, maxiter=50 * NBUMP)[0] / col
    S = np.einsum("k,kyx->yx", c, _phi(P, E_GRID[ie]))
    return _crop(np.fft.irfft2(np.fft.rfft2(rng.standard_normal(P)) * np.sqrt(S), s=P), shape)


def local(h, rng, cover, region, dry=0.0):
    """h confined to a wet part (mask; the dry part keeps `dry` of the amplitude), its cockle band energy (bands
    2..16 nodes) kept: the wet part carries the whole page's energy."""
    if cover >= 1:
        return h
    w = dry + (1 - dry) * mask(h.shape, rng, cover, region)
    hw = h * w
    e = lambda x: float(np.sqrt((bands(x)[1:4] ** 2).sum(0).mean()))
    return hw * e(h) / max(e(hw), 1e-12)


def apply(G, rng, target=None, scale=1.0, level=None, cover=1.0, region=200, dry=0.0):
    """G (rows, cols, 3) [mm] displaced along its unit normals so its Z bands reach `target` (a real page of `level`
    sampled if None), the field confined to a wet part covering `cover` of the page (local); lowest node back at Z = 0."""
    target = sample(rng, level) if target is None else target
    present = dict(rms=bands(G[..., 2]).std((1, 2)))
    n = np.cross(np.gradient(G, axis=1), np.gradient(G, axis=0))
    n *= np.sign(n[..., 2].mean()) / np.linalg.norm(n, axis=-1, keepdims=True)
    out = G + local(field(G.shape[:2], rng, target, present, scale), rng, cover, region, dry)[..., None] * n
    out[..., 2] -= out[..., 2].min()
    return out
