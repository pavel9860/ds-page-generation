"""Page choice inside a document: cheap layout vectors from the text blocks and a farthest-first pick (the most different pages first)."""

import numpy as np

from .config import GRID  # noqa: F401

LITE = 20 + GRID[0] * GRID[1]  # block-level layout features of a text page


def budget(n):
    """Pages to take of a document of n pages."""
    return int(max(10 + np.ceil(0.10 * n), 50))


def lite_feats(pg):
    """Cheap layout vector of a PDF page from its text blocks only (get_text("blocks"): 3-5 ms, no spans, no drawings): sizes and alignment
    of the blocks, characters per line, digits / capitals, a density grid of the text; -> (vector, number of characters)."""
    W, H = pg.rect.width, pg.rect.height
    bl = pg.get_text("blocks")
    f = np.zeros(LITE)
    txt = [b for b in bl if b[6] == 0 and b[4].strip()]
    img = [b for b in bl if b[6] == 1]
    f[2] = len(img)
    f[3] = min(sum(max(0, b[2] - b[0]) * max(0, b[3] - b[1]) for b in img) / (W * H), 1)
    f[18] = np.log1p(W / H * 10)
    chars = 0
    if txt:
        a = np.array([[b[0] / W, b[1] / H, b[2] / W, b[3] / H] for b in txt])
        n = np.array([len(b[4].strip()) for b in txt], float)
        nl = np.array([b[4].count("\n") + 1 for b in txt], float)
        chars = int(n.sum())
        w, h = a[:, 2] - a[:, 0], a[:, 3] - a[:, 1]
        x0 = np.round(a[:, 0] * 20).astype(int)
        s = "".join(b[4] for b in txt)
        f[0], f[1] = np.log1p(chars), np.log1p(len(txt))
        f[4], f[5], f[6], f[7] = min((w * h).sum(), 1), w.mean(), w.std(), h.mean()
        f[8], f[9] = np.log1p(n.mean()), np.bincount(x0).max() / len(x0)
        f[10], f[11] = len(set(x0)) / 20, np.log1p((nl).mean())
        f[12], f[13] = sum(c.isdigit() for c in s) / max(len(s), 1), sum(c.isupper() for c in s) / max(len(s), 1)
        f[14], f[15] = a[:, 1].min(), a[:, 3].max()
        f[16], f[17] = (np.abs((a[:, 0] + a[:, 2]) / 2 - 0.5) < 0.05).mean(), (a[:, 2] > 0.9).mean()
        f[19] = np.log1p((n / nl).mean())
        gx0, gx1 = (
            np.clip(a[:, 0] * GRID[1], 0, GRID[1] - 1).astype(int),
            np.clip(a[:, 2] * GRID[1], 0, GRID[1] - 1).astype(int),
        )
        gy0, gy1 = (
            np.clip(a[:, 1] * GRID[0], 0, GRID[0] - 1).astype(int),
            np.clip(a[:, 3] * GRID[0], 0, GRID[0] - 1).astype(int),
        )
        cells = np.zeros((GRID[0], GRID[1]))
        for i in range(len(a)):
            cells[gy0[i] : gy1[i] + 1, gx0[i] : gx1[i] + 1] += n[i] / ((gy1[i] - gy0[i] + 1) * (gx1[i] - gx0[i] + 1))
        f[20:] = (cells / max(cells.sum(), 1)).ravel()
    return f, chars


def farthest_first(V, seeds, k):
    """Indices of k rows of V picked greedily, each as far as possible (Euclidean) from the seeds and the rows picked before."""
    d = np.full(len(V), np.inf, np.float32)
    for s in seeds:
        d = np.minimum(d, np.linalg.norm(V - s, axis=1))
    out = []
    for _ in range(min(k, len(V))):
        i = int(d.argmax())
        out.append(i)
        d = np.minimum(d, np.linalg.norm(V - V[i], axis=1))
        d[i] = -1
    return out


def pick(doc):
    """Page numbers of an open pymupdf document: budget(n) pages, the most different first."""
    n = doc.page_count
    if n <= budget(n):
        return list(range(n))
    V = np.stack([lite_feats(p)[0] for p in doc])
    return sorted(int(i) for i in farthest_first(V, [], budget(n)))
