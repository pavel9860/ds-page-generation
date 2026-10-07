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
    gsm = Counter(s["gsm"] for s in specs)
    assert set(gsm.values()) == {n // len(f["gsm"])}
    combos = Counter(s["combo"] for s in specs)
    w = np.array([p for _, p in f["combo"]])
    for (c, p), m in zip(f["combo"], w / w.sum()):
        assert abs(combos[c] - n * m) < 1
    assert all(man[s["entry"]]["kind"] == "refont" for s in specs if s["combo"][0] == "open")
    for s in specs:
        assert s["combo"][3] or s["deep"] in ("", "light")
        assert not s["combo"][3] or (not s["two"] and not s["flat"] and s["deep"])
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


def test_layout_bins_cover_the_config_ranges(full_small):
    specs = plan(full_small, 60, 2, fake_manifest())
    c = full_small.layout
    shift = c.open_font_pt[0] - c.font_pt[0]
    fonts = [tuple(v - shift for v in s["font_bin"]) if s["combo"][0] == "open" else s["font_bin"] for s in specs]
    assert all(c.open_font_pt[0] <= s["font_bin"][0] for s in specs if s["combo"][0] == "open")
    for got, (lo, hi) in ((Counter(fonts), c.font_pt), (Counter(s["scale_bin"] for s in specs), c.role_scale)):
        assert min(a for a, _ in got) == lo and max(b for _, b in got) == hi and max(got.values()) - min(got.values()) <= 1


def test_unknown_uses_all(full_small):
    man = fake_manifest()
    specs = plan(full_small, 300, 3, man)
    assert sum(s["script"] == "unknown" for s in specs) == len(entry_pools(full_small, man)["unknown"])
