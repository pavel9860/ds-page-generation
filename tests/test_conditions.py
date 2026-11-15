from collections import Counter

import numpy as np
from conftest import fake_manifest

from dspages.conditions import cells, entry_pools, factors, plan, script


def test_plan_marginals_exact(full_small):
    n = 1000
    man = [dict(e, source_path=f"{e['source_path']}.{k}") for k in range(40) for e in fake_manifest()]
    specs = plan(full_small, n, 0, man)
    assert len(specs) == n
    f = factors(full_small)
    gsm = Counter(s["gsm"] for s in specs if s["combo"][0] != "open")
    assert max(gsm.values()) - min(gsm.values()) <= 0.1 * n / len(f["gsm"])
    assert all(
        full_small.paper.open_gsm[0] <= s["gsm"] <= full_small.paper.open_gsm[1]
        for s in specs
        if s["combo"][0] == "open"
    )
    combos = Counter(s["combo"] for s in specs)
    w = np.array([p for _, p in f["combo"]])
    for (c, p), m in zip(f["combo"], w / w.sum()):
        assert abs(combos[c] - n * m) < 1
    assert all(man[s["entry"]]["kind"] == "refont" for s in specs if s["combo"][0] == "open")
    for s in specs:
        assert s["combo"][3] or s["deep"] in ("", "light")
        assert not s["combo"][3] or (not s["two"] and s["deep"])
        assert -full_small.geometry.bend_dir_deg <= s["bend_dir"] <= full_small.geometry.bend_dir_deg


def test_cells_are_a_distribution(full_small):
    for group in ("geometry", "view", "layout"):
        cs = cells(full_small, group)
        assert abs(sum(p for _, p in cs) - 1) < 1e-9 and len(cs) > 0


def test_pages_are_never_repeated_and_stay_in_their_script_group(full_small):
    man = fake_manifest()
    specs = plan(full_small, 1000, 1, man)
    assert all(script(man[s["entry"]], full_small.layout.scripts) == s["script"] for s in specs)
    entries = [s["entry"] for s in specs]
    assert len(entries) == len(set(entries))


def test_restyled_sizes_fill_the_ranges_in_source_order(full_small):
    man = fake_manifest()
    specs = plan(full_small, 60, 2, man)
    c = full_small.layout
    for opened, (lo, hi) in ((True, c.open_font_pt), (False, c.font_pt)):
        got = sorted(
            (man[s["entry"]]["body_pt"], s["font_pt"])
            for s in specs
            if man[s["entry"]]["kind"] == "refont" and (s["combo"][0] == "open") == opened
        )
        pts = np.array([f for _, f in got])
        assert len(pts) and lo <= pts.min() and pts.max() <= hi
        assert all(b0 == b1 or f0 <= f1 for (b0, f0), (b1, f1) in zip(got, got[1:]))
    assert all("font_pt" not in s for s in specs if man[s["entry"]]["kind"] != "refont")


def test_unknown_uses_all(full_small):
    man = fake_manifest()
    specs = plan(full_small, 300, 3, man)
    assert sum(s["script"] == "unknown" for s in specs) == len(entry_pools(full_small, man)["unknown"])


def test_water_leaves_other_factors_unchanged(full_small):
    from dataclasses import replace

    n = 400
    man = [dict(e, source_path=f"{e['source_path']}.{k}") for k in range(20) for e in fake_manifest()]
    g = full_small.geometry
    dry = replace(full_small, geometry=replace(g, water=replace(g.water, prob=0.0)))
    wet = replace(full_small, geometry=replace(g, water=replace(g.water, prob=0.1, scale=(0.5, 1.0))))
    a, b = plan(dry, n, 0, man), plan(wet, n, 0, man)
    assert all("water" not in s for s in a)
    assert [{k: v for k, v in s.items() if k != "water"} for s in b] == a
    w = [s for s in b if "water" in s]
    assert len(w) == round(0.1 * len(b))
    assert not any(s["combo"][3] or s["shallow"] or s["deep"] for s in w) and all(0.5 <= s["water"] <= 1.0 for s in w)
