"""python -m dspages {plan,layout,geometry,render,all} --preset NAME --out DIR [--n N] [--seed S] [--workers W]

plan      DIR/plan.jsonl: n stratified sample specs (written by the other commands too when missing)
layout    DIR/layouts/<i>.npz   page (sheet luminance), gray (clean sheet) + .jpg
geometry  DIR/geometry/<i>.npz  X, Y, Z [mm] page grid + .jpg height map
render    DIR/samples/<i>.npz   flat, warped, uv, map3d, mask + .jpg flat | warped; reads layouts/ and geometry/
          of the same index when present, else makes them in memory
all       layout, geometry and render of each sample, all saved
Each part of sample i draws from its own stream (seed, i, part): the parts are reproducible independently.
"""
import argparse
import json
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
from .layout.text import find_fonts
from .render.generate import make_sample

PARTS = dict(layout=0, geometry=1, render=2)
_W = {}


def _tuples(o):
    return tuple(_tuples(x) for x in o) if isinstance(o, list) else o


def load_plan(path):
    with open(path) as f:
        return [{k: _tuples(v) for k, v in json.loads(line).items()} for line in f]


def _init(preset, out, seed):
    import numba
    from threadpoolctl import threadpool_limits
    cv2.setNumThreads(1)
    numba.set_num_threads(1)
    threadpool_limits(1)
    P = get_preset(preset)
    _W.update(P=P, out=Path(out), seed=seed, plan=load_plan(Path(out) / "plan.jsonl"),
              manifest=load_manifest(P.paths.manifest), fonts=find_fonts(P.paths.font_dirs))


def _rng(i, part):
    return np.random.default_rng([_W["seed"], i, PARTS[part]])


def _path(part, i):
    return _W["out"] / {"layout": "layouts", "geometry": "geometry", "render": "samples"}[part] / f"{i:07d}.npz"


def _layout(i, save):
    spec = _W["plan"][i]
    L = make_layout(_rng(i, "layout"), dict(_W["manifest"][spec["entry"]], offset=spec.get("offset", 0)),
                    layout_preset(_W["P"], spec), _W["fonts"])
    if save:
        io.save(_path("layout", i), dict(page=L["page"], gray=L["gray"]), L["meta"], L["page"])
    return L["page"]


def _geometry(i, save):
    P = _W["P"]
    s = make_surface(_rng(i, "geometry"), P.geometry, P.paper, geometry_cond(_W["plan"][i]))
    surf = {k: s[k].astype(np.float32) for k in ("X", "Y", "Z")}
    if save:
        meta = dict(label=s["label"], cond=s["cond"], paper=s["paper"], scene_a=s["scene_a"], scene_b=s["scene_b"],
                    n_creases=len(s["creases"]), box_strain=s["box_strain"])
        io.save(_path("geometry", i), surf, meta, io.height_preview(surf["Z"]))
    return surf


def _render(i, save_parts):
    lp, gp = _path("layout", i), _path("geometry", i)
    page = io.load(lp)[0]["page"] if lp.exists() and not save_parts else _layout(i, save_parts)
    surf = io.load(gp)[0] if gp.exists() and not save_parts else _geometry(i, save_parts)
    o = make_sample(_rng(i, "render"), page, surf, view_preset(_W["P"], _W["plan"][i]))
    preview = np.hstack([cv2.resize(o["flat"], o["warped"].shape[1::-1]), o["warped"]])
    io.save(_path("render", i), {k: o[k] for k in ("flat", "warped", "uv", "map3d", "mask")}, o["meta"], preview)


def _job(args):
    cmd, i = args
    t = time.time()
    if cmd == "layout":
        _layout(i, True)
    elif cmd == "geometry":
        _geometry(i, True)
    else:
        _render(i, cmd == "all")
    return i, time.time() - t


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
    a = ap.parse_args(argv)
    out = Path(a.out)
    plan_path = out / "plan.jsonl"
    if a.cmd == "plan" or not plan_path.exists():
        P = get_preset(a.preset)
        out.mkdir(parents=True, exist_ok=True)
        with open(plan_path, "w") as f:
            specs = plan(P, a.n, a.seed, load_manifest(P.paths.manifest))
            for s in specs:
                f.write(json.dumps(s) + "\n")
        short = {g: round(a.n * w) - sum(s["script"] == g for s in specs) for g, w in P.layout.script_mix}
        print(f"plan: {len(specs)} of {a.n} specs -> {plan_path}; short per script: {short}")
    if a.cmd == "plan":
        return
    n = sum(1 for _ in open(plan_path))
    idx = range(a.start, min(n, a.start + (a.count or n)))
    t0, times = time.time(), []
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.preset, str(out), a.seed)) as pool:
        for k, (i, dt) in enumerate(pool.map(_job, [(a.cmd, i) for i in idx], chunksize=4), 1):
            times.append(dt)
            if k % 50 == 0 or k == len(idx):
                print(f"{a.cmd}: {k}/{len(idx)}  {np.mean(times):.2f} s/sample/worker  {time.time() - t0:.0f} s",
                      flush=True)
