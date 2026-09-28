from collections import Counter

import numpy as np

from dspages.conditions import cells, factors, plan


def test_plan_marginals_exact(full_small):
    n = 1000
    specs = plan(full_small, n, 0, 50)
    f = factors(full_small)
    gsm = Counter(s["gsm"] for s in specs)
    assert set(gsm.values()) == {n // len(f["gsm"])}
    combos = Counter(s["combo"] for s in specs)
    w = np.array([p for _, p in f["combo"]])
    for (c, p), m in zip(f["combo"], w / w.sum()):
        assert abs(combos[c] - n * p) < 1
    for s in specs:
        assert s["combo"][3] or (not s["deep"])
        assert not s["combo"][3] or (not s["two"] and not s["flat"] and s["deep"])
        assert -full_small.geometry.bend_dir_deg <= s["bend_dir"] <= full_small.geometry.bend_dir_deg


def test_cells_are_a_distribution(full_small):
    for group in ("geometry", "view", "layout"):
        cs = cells(full_small, group)
        assert abs(sum(p for _, p in cs) - 1) < 1e-9 and len(cs) > 0
