from collections import Counter

import numpy as np

from conftest import fake_manifest
from dspages.conditions import cells, entry_pools, factors, plan, script


def test_plan_marginals_exact(full_small):
    n = 1000
    man = fake_manifest()
    specs = plan(full_small, n, 0, man)
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


def test_script_mix_and_eligibility(full_small):
    man = fake_manifest()
    pools = entry_pools(full_small, man)
    assert all((i // 6) % 2 == 0 for p in pools.values() for i in p)
    n = 1000
    specs = plan(full_small, n, 1, man)
    got = Counter(s["script"] for s in specs)
    for g, w in full_small.layout.script_mix:
        assert abs(got[g] - n * w) <= 1
    assert all(script(man[s["entry"]], full_small.layout.scripts) == s["script"] for s in specs)
