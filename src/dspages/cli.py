"""python -m dspages {plan,layout,geometry,render,all} --preset NAME --out DIR [--n N] [--seed S] [--workers W] [--preview]

plan      DIR/plan.jsonl: n stratified sample specs (written by the other commands too when missing)
layout    DIR/layouts/<i>.npz   page (sheet luminance), gray (clean sheet)
geometry  DIR/geometry/<i>.npz  X, Y, Z [mm] page grid
render    DIR/samples/<i>.npz   flat, warped, uv, xyz, mask (no metadata); reads layouts/ and geometry/ of the same index
          when present, else makes them in memory; DIR/meta.jsonl: one row per sample, the render metadata and its
          layout's (source page, window, fill, fonts, effects)
all       the samples only: layout and geometry of each made in memory, not saved (the dataset run)
--preview writes previews (sheet, height map, flat | warped) to DIR_preview/<part>/<i>.jpg, outside the dataset. The
manifest and its rejected pages are copied to DIR.
Each part of sample i draws from its own stream (seed, i, part): the parts are reproducible independently.
"""
import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from . import io
from .conditions import geometry_cond, layout_preset, plan, view_preset
from .config import get_preset
from .geometry.surface import make_surface
from .layout.generate import make_layout
from .layout.sources import load_manifest
from .layout import refont
from .render.generate import make_sample

PARTS = dict(layout=0, geometry=1, render=2)
_W = {}


def _tuples(o):
    return tuple(_tuples(x) for x in o) if isinstance(o, list) else o


def load_plan(path):
    with open(path) as f:
        return [{k: _tuples(v) for k, v in json.loads(line).items()} for line in f]


def _init(preset, out, seed, manifest=None, preview=False):
    import numba
    from threadpoolctl import threadpool_limits
    cv2.setNumThreads(1)
    numba.set_num_threads(1)
    threadpool_limits(1)
    P = get_preset(preset)
    _W.update(P=P, out=Path(out), seed=seed, preview=preview, plan=load_plan(Path(out) / "plan.jsonl"),
              manifest=load_manifest(manifest or P.paths.manifest), fonts=refont.font_set(P.paths.font_dirs))


def _rng(i, part):
    return np.random.default_rng([_W["seed"], i, PARTS[part]])


def _path(part, i):
    return _W["out"] / {"layout": "layouts", "geometry": "geometry", "render": "samples"}[part] / f"{i:07d}.npz"


def _preview(part, i):
    """Previews live outside the dataset: DIR_preview/<part>/<i>.jpg."""
    return _W["out"].with_name(_W["out"].name + "_preview") / _path(part, i).parent.name / f"{i:07d}.jpg"


def _shown(part, i, what):
    """(preview image, its path) of a part when previews are asked for, else (None, None)."""
    if not _W["preview"]:
        return None, None
    img = {"layout": lambda: what, "geometry": lambda: io.height_preview(what),
           "render": lambda: np.hstack([cv2.resize(what["flat"], what["warped"].shape[1::-1]), what["warped"]])}[part]()
    return img, _preview(part, i)


def _layout(i, save):
    spec = _W["plan"][i]
    L = make_layout(_rng(i, "layout"), _W["manifest"][spec["entry"]],
                    layout_preset(_W["P"], spec), _W["fonts"])
    if save:
        io.save(_path("layout", i), dict(page=L["page"], gray=L["gray"]), L["meta"], *_shown("layout", i, L["page"]))
    return L["page"], L["meta"]


def _geometry(i, save):
    P = _W["P"]
    s = make_surface(_rng(i, "geometry"), P.geometry, P.paper, geometry_cond(_W["plan"][i]), P.render.camera)
    surf = {k: s[k].astype(np.float32) for k in ("X", "Y", "Z")}
    if save:
        meta = dict(label=s["label"], cond=s["cond"], paper=s["paper"], scene_a=s["scene_a"], scene_b=s["scene_b"],
                    n_creases=len(s["creases"]), box_strain=s["box_strain"])
        io.save(_path("geometry", i), surf, meta, *_shown("geometry", i, surf["Z"]))
    return surf


def _render(i, saved, dt):
    """A sample: its layout and geometry read from DIR when `saved` and present, else made in memory (not saved)."""
    lp, gp = _path("layout", i), _path("geometry", i)
    t = time.time()
    if saved and lp.exists():
        arrays, lmeta = io.load(lp)
        page = arrays["page"]
    else:
        page, lmeta = _layout(i, False)
    dt["layout"], t = time.time() - t, time.time()
    surf = io.load(gp)[0] if saved and gp.exists() else _geometry(i, False)
    dt["geometry"], t = time.time() - t, time.time()
    o = make_sample(_rng(i, "render"), page, surf, view_preset(_W["P"], _W["plan"][i]))
    io.save(_path("render", i), {k: o[k] for k in ("flat", "warped", "uv", "xyz", "mask")}, None, *_shown("render", i, o))
    dt["render"] = time.time() - t
    return dict(sample=f"{i:07d}", **o["meta"], layout=lmeta)


def _job(args):
    """-> (i, seconds per step, metadata row of a rendered sample or None)."""
    cmd, i = args
    t, dt = time.time(), {}
    if _path("render" if cmd in ("render", "all") else cmd, i).exists():
        return i, dt, None
    if cmd == "layout":
        _layout(i, True)
        dt["layout"] = time.time() - t
    elif cmd == "geometry":
        _geometry(i, True)
        dt["geometry"] = time.time() - t
    else:
        return i, dt, _render(i, cmd == "render", dt)
    return i, dt, None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dspages", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("plan", "layout", "geometry", "render", "all"))
    ap.add_argument("--preset", default="demo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--manifest", default=None, help="manifest path (default: the preset's)")
    ap.add_argument("--preview", action="store_true", help="write previews to DIR_preview")
    a = ap.parse_args(argv)
    out = Path(a.out)
    plan_path = out / "plan.jsonl"
    if a.cmd == "plan" or not plan_path.exists():
        P = get_preset(a.preset)
        out.mkdir(parents=True, exist_ok=True)
        with open(plan_path, "w") as f:
            specs = plan(P, a.n, a.seed, load_manifest(a.manifest or P.paths.manifest))
            for s in specs:
                f.write(json.dumps(s) + "\n")
        per = {g: sum(s["script"] == g for s in specs) for g, _ in P.layout.script_mix}
        print(f"plan: {len(specs)} of {a.n} specs -> {plan_path}; per script: {per}", flush=True)
    if a.cmd == "plan":
        return
    n = sum(1 for _ in open(plan_path))
    idx = range(a.start, min(n, a.start + (a.count or n)))
    man = Path(a.manifest or get_preset(a.preset).paths.manifest)
    for src in (man, man.with_suffix(".rejected.jsonl")):
        if src.exists() and not (out / src.name).exists():
            shutil.copy2(src, out / src.name)
    t0, times = time.time(), {}
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.preset, str(out), a.seed, a.manifest, a.preview)) as pool, \
            open(out / "meta.jsonl", "a") as meta_f:
        for k, (i, dt, row) in enumerate(pool.map(_job, [(a.cmd, i) for i in idx], chunksize=1), 1):
            if row is not None:
                meta_f.write(json.dumps(row, default=io._json) + "\n")
            for step, v in dt.items():
                times.setdefault(step, []).append(v)
            if k % 50 == 0 or k == len(idx):
                el = time.time() - t0
                steps = "  ".join(f"{s_} {np.mean(v):.2f}" for s_, v in times.items())
                print(f"{a.cmd}: {k}/{len(idx)}  s/sample/worker: {steps}  elapsed {el:.0f} s  "
                      f"eta {el / k * (len(idx) - k):.0f} s", flush=True)
