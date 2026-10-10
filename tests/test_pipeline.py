import json
from collections import Counter
from dataclasses import replace

import cv2
import numpy as np
import pytest

from conftest import fake_manifest
from dspages import cli
from dspages.conditions import factors, geometry_cond, plan, sample_preset
from dspages.geometry.surface import make_surface, page_points
from dspages.layout import paper
from dspages.layout.generate import draw_layout, make_layout
from dspages.render.camera import _frame_region
from dspages.render.generate import _fit_aspect, make_sample


def test_frame_marginals_exact_and_cut_frames_get_room(full_small):
    n = 1000
    man = [dict(e, source_path=f"{e['source_path']}.{k}") for k in range(40) for e in fake_manifest()]
    specs = plan(full_small, n, 0, man)
    got = Counter(s["frame"] for s in specs)
    for frame, w in factors(full_small)["frame"]:
        assert abs(got[frame] - n * w) < 1
    for s in specs[:50]:
        P = sample_preset(full_small, s)
        assert P.render.camera.frame == s["frame"]
        assert P.layout.margin_frac == (full_small.layout.cut_margin_frac if s["frame"] != "page" else s["margin_bin"])


@pytest.mark.parametrize("frame", ["sides", "inside"])
def test_frame_region_keeps_the_content_and_has_the_sheet_aspect(full_small, frame):
    c = replace(full_small.render.camera, frame=frame)
    rng = np.random.default_rng(0)
    for _ in range(200):
        content = np.sort(rng.uniform(0.05, 0.95, (2, 2)), axis=0).T.ravel()[[0, 2, 1, 3]]
        region, cut = _frame_region(rng, content, c)
        assert cut.any() and (frame != "inside" or cut.all())
        size = np.tile(content[2:] - content[:2], 2)
        k = int(content[2] - content[0] > content[3] - content[1])
        longer = np.array([k == 1, k == 0] * 2)
        assert (region[:2] <= content[:2] + 1e-12).all() and (region[2:] >= content[2:] - 1e-12).all()
        assert (region >= 0).all() and (region <= 1).all()
        assert abs((region[2] - region[0]) - (region[3] - region[1])) < 1e-9
        if frame == "inside":
            assert (np.abs(region - content)[longer] <= c.crop_margin[1] * size[longer] + 1e-12).all()


def test_flat_part_has_the_sheet_aspect_and_keeps_the_content():
    rng = np.random.default_rng(1)
    for _ in range(500):
        content = np.sort(rng.uniform(0.1, 0.9, (2, 2)), axis=0).T.ravel()[[0, 2, 1, 3]]
        tight = np.r_[rng.uniform(0, content[:2]), rng.uniform(content[2:], 1)]
        loose = np.r_[rng.uniform(0, tight[:2]), rng.uniform(tight[2:], 1)]
        a = _fit_aspect(tight, loose, content)
        assert (a[:2] <= content[:2] + 1e-12).all() and (a[2:] >= content[2:] - 1e-12).all()
        assert (a >= -1e-12).all() and (a <= 1 + 1e-12).all()
        assert abs((a[2] - a[0]) - (a[3] - a[1])) < 1e-9


@pytest.mark.parametrize("frame", ["page", "sides", "inside"])
def test_flat_is_a_slice_of_the_drawn_sheet_and_the_content_is_in_the_photo(full_small, frame):
    P = replace(full_small, layout=replace(full_small.layout, grid_prob=1.0))
    man = fake_manifest()
    for k, spec in enumerate(plan(P, 4, 3, man)):
        Ps = sample_preset(P, dict(spec, frame=frame))
        surf = make_surface(np.random.default_rng(k), Ps.geometry, Ps.paper, geometry_cond(spec))
        desc = make_layout(np.random.default_rng(k), man[spec["entry"]], Ps, None)["desc"]
        o = make_sample(np.random.default_rng(k), desc, surf, Ps)
        m = o["meta"]
        page = draw_layout(desc, m["flat_scale"], Ps)[0]
        u0, v0, u1, v1 = m["flat_part"]
        ph, pw = page.shape
        x0, y0 = round(u0 * (pw - 1)), round(v0 * (ph - 1))
        flat = o["flat"] if Ps.layout.page_px[0] <= Ps.layout.page_px[1] else np.rot90(o["flat"], -1)
        assert (flat == page[y0:y0 + flat.shape[0], x0:x0 + flat.shape[1]]).all()
        ink = np.array(desc["ink"])
        assert u0 <= ink[0] + 1e-3 and v0 <= ink[1] + 1e-3 and u1 >= ink[2] - 1e-3 and v1 >= ink[3] - 1e-3
        assert m["view"]["cut"] == (frame != "page")
        g = o["grid2d"]
        assert np.isfinite(g).all() and np.abs(g).max() < 1 + 2 / (min(o["warped"].shape) - 1)


def test_paper_effects_do_not_depend_on_the_resolution(full_small):
    c = full_small.layout
    ppm = c.page_px[0] / c.sheet_mm[0]
    for seed in range(5):
        a, fa = paper.apply(np.random.default_rng(seed), np.full((360, 256), 255, np.uint8), ppm / 4, c, full_small.geometry.shallow)
        b, fb = paper.apply(np.random.default_rng(seed), np.full((720, 512), 255, np.uint8), ppm / 2, c, full_small.geometry.shallow)
        assert fa == fb
        small = cv2.resize(b, a.shape[::-1], interpolation=cv2.INTER_AREA).astype(float)
        assert abs(small.mean() - a.mean()) < 2.0


def test_page_points_are_the_grid_nodes_at_the_nodes(full_small):
    for k, spec in enumerate(plan(full_small, 6, 2, fake_manifest())):
        s = make_surface(np.random.default_rng(k), full_small.geometry, full_small.paper, geometry_cond(spec))
        X, Y, Z = page_points(s, *np.meshgrid(s["u"], s["v"]))
        assert np.allclose(X, s["X"]) and np.allclose(Y, s["Y"]) and np.allclose(Z, s["Z"])


def _raster_manifest(tmp_path, n, missing):
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        p = tmp_path / f"page{i}.png"
        if i not in missing:
            img = np.full((420, 300), 250, np.uint8)
            for y in range(40, 380, 14):
                img[y:y + 6, 30:30 + int(rng.integers(150, 240))] = 20
            cv2.imwrite(str(p), img)
        out.append(dict(source_path=str(p), page_index=0, kind="raster", language="en", category="test", rotate90=0,
                        deskew_deg=0.0, used_h=420, used_w=300, window=[0.0, 0.0, 300.0, 420.0], margin=0.05, fill=0.8,
                        picture_share=0.0))
    path = tmp_path / "manifest_pages.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in out))
    return path


def test_a_failing_sample_is_logged_and_the_run_goes_on_and_resumes(tmp_path):
    man = _raster_manifest(tmp_path, 4, missing={2})
    out = tmp_path / "run"
    cli.main(["all", "--preset", "demo", "--out", str(out), "--n", "4", "--workers", "1", "--manifest", str(man)])
    plan_ = [json.loads(l) for l in open(out / "plan.jsonl")]
    bad = {f"{i:07d}" for i, s in enumerate(plan_) if s["entry"] == 2}
    done = {json.loads(l)["sample"] for l in open(out / "meta.jsonl")}
    failed = [json.loads(l) for l in open(out / "failed.jsonl")]
    assert bad and {f["sample"] for f in failed} == bad and all(f["error"] for f in failed)
    assert done == {f"{i:07d}" for i in range(len(plan_))} - bad
    assert all((out / "samples" / f"{s}.npz").exists() for s in done)
    cli.main(["all", "--preset", "demo", "--out", str(out), "--n", "4", "--workers", "1", "--manifest", str(man)])
    assert sum(1 for _ in open(out / "meta.jsonl")) == len(done)


def test_a_manifest_file_that_raises_is_rejected_with_its_error(tmp_path):
    from dspages.prep import manifest
    p = tmp_path / "broken.pdf"
    p.write_bytes(b"%PDF-1.4 not a pdf")
    recs = manifest._run(dict(path=str(p), language="en", category="test", digital=True, cap=2, skip=set()))
    assert len(recs) == 1 and recs[0]["kind"] == "rejected"
