"""Coverage selection (dspages.curate): water filling, two-level quotas, scans needed, forms plan, keep rates."""
import numpy as np
import pandas as pd

from dspages import curate as C


def test_water_caps_and_redistributes():
    sup = pd.Series([10.0, 1000.0, 1000.0], index=list("abc"))
    a = C.water(sup, 300, sup ** 0.5)
    assert a["a"] == 10 and abs(a.sum() - 300) < 1e-6 and abs(a["b"] - a["c"]) < 1e-6
    assert abs(C.water(sup, 1e6, sup).sum() - sup.sum()) < 1e-6          # budget over the supply: everything


def pool():
    rows = []
    for c1, c2, n, scan in ((0, 0, 40, False), (1, 0, 10, True), (2, 1, 30, False)):
        rows += [dict(doc=f"{c1}_{i}", c1=c1, c2=c2, scan=scan, pages=1, label="plain", folder="text") for i in range(n)]
    return pd.DataFrame(rows)


def test_allocate_and_scans_needed():
    P = pool()
    sup = C.supply(P)
    q1 = C.allocate(sup, 60)
    assert q1.sum() == 60 and (q1 <= sup.n + sup.deep).all()
    need = C.scans_needed(sup, q1)
    assert need[0] == 0 and need[2] == 0 and need[1] == q1[1] > 0       # only the scan cluster needs scans
    d = C.demand(sup, 60)
    assert abs(d.sum() - 60) < 1e-6


def test_deep_pages_of_digital_documents():
    P = pool()
    P.loc[P.c1 == 0, "pages"] = 10                                       # 2 embedded rows per document carry half each
    sup = C.supply(P)
    assert sup.deep[0] == 40 * 8 / 2 and sup.deep[1] == 0


def test_forms_plan_order():
    P = pool()
    P["p_form"] = 0.9
    P["p_fill"] = np.where(P.index % 2 == 0, 0.1, 0.9)
    P["reviewed"] = False
    P["real_filled"] = P.index < 2
    sup = C.supply(P)
    f = C.forms_plan(P, sup, C.allocate(sup, 80))
    assert (f.real + f.fill + f.cand_dig_use + f.cand_scan_use + f.download == f.need).all()
    assert f.loc[1, "fill"] == 0                                         # the scan cluster has no digital blank to fill


def test_keep_rates_and_inflate():
    docs = pd.DataFrame({"top": ["maps"] * 4 + ["tables"] * 4, "scan": [False, False, True, False] + [False] * 4,
                         "label": ["maps", "maps", "maps", "flyers", "tables", "tables", "plain", "plain"]})
    k = C.keep_rates(docs, dup=1.0)
    assert abs(k["maps"] - 0.75 * 2 / 3) < 1e-9 and abs(k["tables"] - 0.5) < 1e-9
    assert list(C.inflate(pd.Series([10, 10]), pd.Series([0.5, 0.01]))) == [20, 67]


def test_crop_box_and_page_ok():
    assert C.crop_box(1000, 1414) is None
    assert C.crop_box(1000, 1300) == (40, 0, 959, 1300)                    # US letter: the sides cut to A4
    assert C.crop_box(1414, 1000) is False                                  # landscape: out
    ok = C.page_ok([1.414, 1.3, 0.75, 1.414], [np.inf, 2000, 3000, 600])
    assert list(ok) == [True, True, False, False]


def test_near_duplicates_keeps_the_preferred_page():
    X = np.array([[1, 0], [0.999, 0.0447], [0, 1]], float)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    assert list(C.near_duplicates(X, [0, 0, 0], prefer=[1, 0, 2])) == [True, False, False]
