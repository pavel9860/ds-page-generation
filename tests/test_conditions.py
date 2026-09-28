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
    assert all((i // 6) % 2 == 0 for raster, _ in pools.values() for i in raster)
    n = 1000
    specs = plan(full_small, n, 1, man)
    got = Counter(s["script"] for s in specs)
    for g, w in full_small.layout.script_mix:
        assert got[g] == min(round(n * w), len(pools[g][0])) or abs(got[g] - n * w) <= 1
    assert all(script(man[s["entry"]], full_small.layout.scripts) == s["script"] for s in specs)
    raster = [s["entry"] for s in specs if man[s["entry"]]["kind"] == "raster"]
    assert len(raster) == len(set(raster))


def test_raster_pages_before_books(full_small):
    man = fake_manifest() + [dict(source_path=f"b{i}.txt", page_index=0, kind="book_text", language="en",
                                  category="book_filler") for i in range(500)]
    specs = plan(full_small, 400, 2, man)
    latin = [s["entry"] for s in specs if s["script"] == "latin"]
    raster = {i for i in latin if man[i]["kind"] == "raster"}
    assert len(raster) == len(entry_pools(full_small, man)["latin"][0])


def test_books_fill_quota_at_distinct_offsets_and_unknown_uses_all(full_small):
    man = fake_manifest()[::12] + [dict(source_path=f"b{i}.txt", page_index=0, kind="book_text", language="en",
                                        category="book", chars=50000 * (i + 1)) for i in range(2)]
    pools = entry_pools(full_small, man)
    specs = plan(full_small, 300, 3, man)
    unknown = [s for s in specs if s["script"] == "unknown"]
    assert len(unknown) == len(pools["unknown"][0])
    books = [(s["entry"], s["offset"]) for s in specs if man[s["entry"]]["kind"] == "book_text"]
    assert len(books) == len(set(books)) > 0
    k = Counter(e for e, _ in books)
    assert abs(k[len(man) - 1] - 2 * k[len(man) - 2]) <= 2
