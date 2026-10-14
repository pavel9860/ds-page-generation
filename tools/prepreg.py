"""Prepregs of built datasets for on-the-fly samples (dspages.onfly): per sample the flat sheet, the plan spec and what
replays its geometry without the solvers (rng state, solved strips, sag at the shell nodes), checked against the
sample's grid3d; and a bank of table textures.

OUT/{train,val,test}/<k>.npz   k numbered over the parts in order, the part's split kept
OUT/index.jsonl                k, part, sample, split, max |replayed grid3d - grid3d| [mm]
OUT/tables.npy                 (n, TABLE_PX, TABLE_PX) uint8 table textures over TABLE_MM around the page centre

python tools/prepreg.py OUT --parts DIR1:SEED1 DIR2:SEED2 --n 100000 [--tables 1000] [--workers 12]
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from dspages.conditions import geometry_cond, sample_preset
from dspages.config import get_preset
from dspages.geometry.surface import make_surface, replay_surface
from dspages.onfly import TABLE_MM, TABLE_PX, grid3d, table_bank_texture

P0 = get_preset("full")
GEOMETRY = 1                    # cli.PARTS["geometry"]


def _job(a):
    k, src, out, seed, spec = a
    f = np.load(src)
    P = sample_preset(P0, spec)
    cond = geometry_cond(spec)
    s = make_surface(np.random.default_rng([seed, int(Path(src).stem.split("_")[-1]) if False else a[5], GEOMETRY]),
                     P.geometry, P.paper, cond, P.render.camera)
    strips = np.array([np.stack(q) for q in s["strips"]], np.float32)
    sag = np.zeros(0, np.float32) if s["sag"] is None else s["sag"].astype(np.float32)
    state = json.dumps(s["rng_state"])
    r = replay_surface(json.loads(state), P.geometry, P.paper, cond, list(strips), None if not sag.size else sag)
    err = float(np.abs(grid3d(r, spec) - f["grid3d"]).max())
    np.savez_compressed(out, flat=f["flat"], strips=strips, sag=sag, rng_state=state, spec=json.dumps(spec))
    return k, err


def _table(a):
    i, seed = a
    return i, table_bank_texture(np.random.default_rng([seed, i]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--parts", nargs="+", required=True, help="DIR:SEED (the seed the part was built with)")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--tables", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    out = Path(a.out)
    jobs = []
    for part in a.parts:
        d, seed = part.rsplit(":", 1)
        d = Path(d)
        plan = [json.loads(line) for line in open(d / "plan.jsonl")]
        for line in open(d / "meta.jsonl"):
            if len(jobs) == a.n:
                break
            m = json.loads(line)
            i = int(m["sample"])
            k = len(jobs)
            jobs.append((k, str(d / m["split"] / m["file"]), str(out / m["split"] / f"{k:07d}.npz"), int(seed),
                         plan[i], i, d.name, m["split"]))
    for split in ("train", "val", "test"):
        (out / split).mkdir(parents=True, exist_ok=True)
    done = set()
    if (out / "index.jsonl").exists():
        done = {json.loads(line)["k"] for line in open(out / "index.jsonl")}
    todo = [j for j in jobs if j[0] not in done]
    with ProcessPoolExecutor(a.workers) as pool, open(out / "index.jsonl", "a") as idx:
        for n, (k, err) in enumerate(pool.map(_job, todo, chunksize=4), 1):
            j = jobs[k]
            idx.write(json.dumps(dict(k=k, part=j[6], sample=f"{j[5]:07d}", split=j[7], err_mm=err)) + "\n")
            if n % 1000 == 0:
                idx.flush()
                print(f"prepregs {n}/{len(todo)}", flush=True)
        if a.tables and not (out / "tables.npy").exists():
            bank = np.lib.format.open_memmap(out / "tables.tmp.npy", "w+", np.uint8, (a.tables, TABLE_PX, TABLE_PX))
            for i, t in pool.map(_table, [(i, 0) for i in range(a.tables)]):
                bank[i] = t
            bank.flush()
            del bank
            (out / "tables.tmp.npy").rename(out / "tables.npy")
    print(f"done: {len(jobs)} prepregs, table bank {TABLE_PX}px over {TABLE_MM} mm", flush=True)


if __name__ == "__main__":
    main()
