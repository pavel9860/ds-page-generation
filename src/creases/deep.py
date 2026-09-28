"""Deep creases: 3D crumple ridges (1-2 mm) for synth.geometry.make_surface.

A crease is two sharp bends joined by a ramp; make_ridge_network places a crumple-then-flatten ridge network
(Blair & Kudrolli statistics). Lengths/radii in mm on the synth patch (ppmm converts pixel ranges)."""
import numpy as np

CREASE_R_MM = (1, 2)                  # bend radius at each end
CREASE_ANGLE_DEG = (15, 45)           # ramp angle between bends
CREASE_HEIGHT_MM = (1, 2)             # peak amplitude
CREASE_LENGTH_PX = (50, 150)          # along-mark length


def make_crease(rng, span, ppmm, severity=1.0):
    """One crease: two sharp bends (radius, primary) connected by a ramp at
    `angle` (primary), peak amplitude `height` (primary). `width` is
    SECONDARY -- derived as 2*height/tan(angle) so the ramp's own plateau
    equals `height` without normalizing it away."""
    r_mm = rng.uniform(*CREASE_R_MM) / ppmm
    angle = np.deg2rad(rng.uniform(*CREASE_ANGLE_DEG))
    height_mm = rng.uniform(*CREASE_HEIGHT_MM) * severity
    sign = rng.choice([-1, 1])
    width_mm = 2 * height_mm / np.tan(angle)   # secondary: makes plateau == height
    length_mm = rng.uniform(*CREASE_LENGTH_PX) / ppmm
    th = rng.uniform(0, np.pi)
    cx, cy = rng.uniform(0.0, 1.0, 2) * span
    return dict(th=th, cx=cx, cy=cy, r=r_mm, width=width_mm, angle=angle,
               height=height_mm * sign, length=length_mm)


def _lognormal_clip(rng, median, sigma, lo, hi):
    v = rng.lognormal(mean=np.log(median), sigma=sigma)
    return float(np.clip(v, lo, hi))


def make_ridge_network(rng, span, ppmm, severity=1.0, n_seed_nodes=(2, 5),
                       n_grow_attempts=(40, 90), max_degree=4, bifurcate_prob=0.65):
    """Crumple-then-flatten ridge network, per the measured statistics in
    Blair & Kudrolli, "Geometry of Crumpled Paper" (PRL 94, 166107, 2005)
    and "Ridge Network in Crumpled Paper" (arXiv:cond-mat/0703020):
      - ridge lengths are log-normally distributed (a few long dominant
        ridges + many short ones), not a uniform range
      - vertex degree has an exponentially decaying tail -- `max_degree`
        is a hard cap that approximates this (almost all nodes stay
        low-degree; very few reach the cap)
      - a large fraction of ridges terminate WITHOUT bifurcating --
        `bifurcate_prob` gates most growth attempts into dangling ends
        rather than forced branches
      - curvature (`r`, the bend radius) is exponentially distributed,
        concentrated/sharp -- unlike the old uniform CREASE_R_MM draw
    Height/curvature-radius/angle ranges reuse the same config knobs as the
    old single-crease model (CREASE_HEIGHT_MM, CREASE_R_MM, CREASE_ANGLE_DEG)
    -- only the population/graph structure is new."""
    nodes, degree, ridges = [], [], []

    n_seed = rng.integers(*n_seed_nodes)
    n_side = max(1, int(np.ceil(np.sqrt(n_seed))))
    cell = span / n_side
    cells = [(i, j) for i in range(n_side) for j in range(n_side)]
    rng.shuffle(cells)
    for k in range(n_seed):
        gi, gj = cells[k % len(cells)]
        x = (gi + rng.uniform(0.2, 0.8)) * cell
        y = (gj + rng.uniform(0.2, 0.8)) * cell
        nodes.append((x, y))
        degree.append(0)

    for _ in range(rng.integers(*n_grow_attempts)):
        if rng.random() >= bifurcate_prob:
            continue   # dangling attempt -- most ridges don't bifurcate
        eligible = [i for i, d in enumerate(degree) if d < max_degree]
        if not eligible:
            break
        i = eligible[rng.integers(len(eligible))]
        th = rng.uniform(0, np.pi)
        length_mm = _lognormal_clip(rng, span * 0.12, 0.5, span * 0.024, span * 0.36)
        dx, dy = length_mm / 2 * np.cos(th), length_mm / 2 * np.sin(th)
        cx, cy = nodes[i][0] + dx, nodes[i][1] + dy
        r_min_mm = CREASE_R_MM[0] / ppmm
        r_mm = rng.exponential(r_min_mm) + r_min_mm * 0.15
        angle = np.deg2rad(rng.uniform(*CREASE_ANGLE_DEG))
        height_mm = rng.uniform(*CREASE_HEIGHT_MM) * severity
        sign = rng.choice([-1, 1])
        width_mm = 2 * height_mm / np.tan(angle)
        ridges.append(dict(th=th, cx=cx, cy=cy, r=r_mm, width=width_mm, angle=angle,
                           length=length_mm, height=height_mm * sign))
        nodes.append((cx + dx, cy + dy))
        degree.append(0)
        degree[i] += 1

    return ridges


def crease_field(c, U, V):
    """Two sharp bends at +-width/2, a ramp between them, an envelope
    relaxing to flat, an along-taper for a short mark not an infinite line."""
    du, dv = U - c['cx'], V - c['cy']
    t_perp = du * np.cos(c['th']) + dv * np.sin(c['th'])
    t_along = -du * np.sin(c['th']) + dv * np.cos(c['th'])
    w2 = c['width'] / 2
    # radius of curvature at each bend = 2*r_param/tan(angle); solve for
    # r_param so that equals c['r'] (plugging c['r'] straight in is ~4x off)
    r_param = c['r'] * np.tan(c['angle']) / 2
    ridge = 0.5 * np.tan(c['angle']) * (
        np.sqrt((t_perp + w2) ** 2 + r_param ** 2) - np.sqrt((t_perp - w2) ** 2 + r_param ** 2))
    envelope = np.exp(-(t_perp / (3 * c['width'])) ** 2)
    along = np.exp(-(t_along / (c['length'] / 2)) ** 2)
    # ridge's plateau already equals |height| (width was derived for that) -- only sign needed
    return np.sign(c['height']) * ridge * envelope * along
