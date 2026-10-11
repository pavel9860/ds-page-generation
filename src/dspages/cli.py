"""python -m dspages {plan,layout,geometry,render,all} --preset NAME --out DIR [--n N] [--seed S] [--workers W] [--preview]

plan      DIR/plan.jsonl: n stratified sample specs (written by the other commands too when missing)
layout    DIR/layouts/<i>.npz   page: the sheet luminance at half size (a preview: the render lays the page out again and
          draws it at the scale its flat needs)
geometry  DIR/geometry/<i>.npz  X, Y, Z [mm] page grid
render    DIR/samples/<i>.npz   flat, warped, grid2d, grid3d, mask (no metadata); reads geometry/ of the same index when
          present, else makes it in memory; the layout is always made in memory; DIR/meta.jsonl: one row per sample, the render metadata and its
          layout's (source page, window, fill, fonts, effects)
all       the samples only: layout and geometry of each made in memory, not saved (the dataset run)
A sample that fails (a missing or broken source file) is written to DIR/failed.jsonl with its error and the run goes on;
render and all skip the samples meta.jsonl already has (a stopped run goes on where it was). DIR/run.json holds what the
plan was made from (preset, n, seed, manifest and its sha256): a run with other ones in that DIR stops with an error.
--preview writes previews (sheet, height map, flat | warped) to DIR_preview/<part>/<i>.jpg, outside the dataset. The
manifest and its rejected pages are copied to DIR.
Each part of sample i draws from its own stream (seed, i, part): the parts are reproducible independently.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from . import io
from .conditions import geometry_cond, plan, sample_preset
from .config import get_preset
from .geometry.surface import make_surface
from .layout.generate import draw_layout, make_layout
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
           "render": lambda: cv2.resize(np.hstack([what["flat"], what["warped"]]), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)}[part]()
    return img, _preview(part, i)


def _layout(i, save):
    spec = _W["plan"][i]
    L = make_layout(_rng(i, "layout"), _W["manifest"][spec["entry"]],
                    sample_preset(_W["P"], spec), _W["fonts"], spec["combo"][0] == "open", spec.get("font_pt"))
    if save:
        page = draw_layout(L["desc"], 0.5, sample_preset(_W["P"], spec))[0]
        io.save(_path("layout", i), dict(page=page), L["meta"], *_shown("layout", i, page))
    return L["desc"], L["meta"]


def _geometry(i, save):
    P = sample_preset(_W["P"], _W["plan"][i])
    s = make_surface(_rng(i, "geometry"), P.geometry, P.paper, geometry_cond(_W["plan"][i]), P.render.camera)
    surf = {k: s[k].astype(np.float32) for k in ("X", "Y", "Z")} | {k: s[k] for k in ("u", "v", "W", "Wu", "Wv")}
    if save:
        meta = dict(label=s["label"], cond=s["cond"], paper=s["paper"], scene_a=s["scene_a"], scene_b=s["scene_b"],
                    n_creases=len(s["creases"]), box_strain=s["box_strain"])
        io.save(_path("geometry", i), surf, meta, *_shown("geometry", i, surf["Z"]))
    return surf


def _render(i, saved, dt):
    """A sample: its geometry read from DIR when `saved` and present, else made in memory (not saved); its layout made
    in memory (drawn by the render at the flat's scale)."""
    gp = _path("geometry", i)
    t = time.time()
    desc, lmeta = _layout(i, False)
    dt["layout"], t = time.time() - t, time.time()
    surf = io.load(gp)[0] if saved and gp.exists() else _geometry(i, False)
    dt["geometry"], t = time.time() - t, time.time()
    o = make_sample(_rng(i, "render"), desc, surf, sample_preset(_W["P"], _W["plan"][i]))
    io.save(_path("render", i), {k: o[k] for k in ("flat", "warped", "grid2d", "grid3d", "mask")}, None, *_shown("render", i, o))
    dt["render"] = time.time() - t
    return dict(sample=f"{i:07d}", **o["meta"], layout=lmeta)


def _job(args):
    """-> (i, seconds per step, metadata row of a rendered sample or None, error or None)."""
    cmd, i = args
    try:
        return (*_run(cmd, i), None)
    except Exception as e:
        return i, {}, None, f"{type(e).__name__}: {e}"


def _run(cmd, i):
    t, dt = time.time(), {}
    if cmd in ("layout", "geometry") and _path(cmd, i).exists():
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


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_lines(path, rows):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    os.replace(tmp, path)


def _read_rows(path):
    """The rows of a JSONL file; a last line cut short (the run was stopped while writing it) is dropped from it."""
    if not path.exists():
        return []
    rows, good = [], 0
    with open(path, "rb") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:
                break
            good += len(line)
    if good < path.stat().st_size:
        os.truncate(path, good)
    return rows


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
    plan_path, run_path = out / "plan.jsonl", out / "run.json"
    P = get_preset(a.preset)
    man = Path(a.manifest or P.paths.manifest)
    run = dict(preset=a.preset, n=a.n, seed=a.seed, manifest=str(man), manifest_sha256=_sha256(man))
    if plan_path.exists() and a.cmd != "plan":
        made = json.loads(run_path.read_text()) if run_path.exists() else None
        if made != run:
            sys.exit(f"{out} was planned for {made}, not {run}: use another --out or delete it")
    else:
        out.mkdir(parents=True, exist_ok=True)
        specs = plan(P, a.n, a.seed, load_manifest(man))
        _write_lines(plan_path, specs)
        for src in (man, man.with_suffix(".rejected.jsonl")):
            if src.exists():
                shutil.copy2(src, out / src.name)
        run_path.write_text(json.dumps(run))
        for stale in ("meta.jsonl", "failed.jsonl"):
            (out / stale).unlink(missing_ok=True)
        per = {g: sum(s["script"] == g for s in specs) for g, _ in P.layout.script_mix}
        print(f"plan: {len(specs)} of {a.n} specs -> {plan_path}; per script: {per}", flush=True)
    if a.cmd == "plan":
        return
    n = sum(1 for _ in open(plan_path))
    done = {r["sample"] for r in _read_rows(out / "meta.jsonl")} if a.cmd in ("render", "all") else set()
    idx = [i for i in range(a.start, min(n, a.start + (a.count or n))) if f"{i:07d}" not in done]
    t0, times = time.time(), {}
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.preset, str(out), a.seed, a.manifest, a.preview)) as pool, \
            open(out / "meta.jsonl", "a") as meta_f, open(out / "failed.jsonl", "a") as fail_f:
        for k, (i, dt, row, err) in enumerate(pool.map(_job, [(a.cmd, i) for i in idx], chunksize=1), 1):
            if row is not None:
                meta_f.write(json.dumps(row, default=io._json) + "\n")
                meta_f.flush()
            if err is not None:
                fail_f.write(json.dumps(dict(sample=f"{i:07d}", cmd=a.cmd, error=err)) + "\n")
                fail_f.flush()
            for step, v in dt.items():
                times.setdefault(step, []).append(v)
            if k % 50 == 0 or k == len(idx):
                el = time.time() - t0
                steps = "  ".join(f"{s_} {np.mean(v):.2f}" for s_, v in times.items())
                print(f"{a.cmd}: {k}/{len(idx)}  s/sample/worker: {steps}  elapsed {el:.0f} s  "
                      f"eta {el / k * (len(idx) - k):.0f} s", flush=True)
