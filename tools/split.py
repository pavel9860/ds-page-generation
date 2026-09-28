"""Split a created dataset into DIR/{train,test,val}/ by moving each sample's files (samples/<i>.npz, .jpg and
the layouts/ and geometry/ files of the same index when present), by a seeded shuffle of the sample indices.

python tools/split.py DIR [--frac 0.96 0.02 0.02] [--seed 0]
"""
import argparse
from pathlib import Path

import numpy as np

PARTS = ("samples", "layouts", "geometry")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--frac", type=float, nargs=3, default=(0.96, 0.02, 0.02))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    idx = sorted(int(p.stem) for p in (out / "samples").glob("*.npz"))
    perm = np.random.default_rng(a.seed).permutation(idx)
    n_test, n_val = round(len(idx) * a.frac[1]), round(len(idx) * a.frac[2])
    names = dict(zip(perm, ["test"] * n_test + ["val"] * n_val + ["train"] * (len(idx) - n_test - n_val)))
    for i, split in names.items():
        for part in PARTS:
            for src in (out / part).glob(f"{i:07d}.*"):
                dst = out / split / part / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                src.rename(dst)
    with open(out / "splits.txt", "w") as f:
        for i in sorted(names):
            f.write(f"{i:07d} {names[i]}\n")
    print({s: sum(v == s for v in names.values()) for s in ("train", "test", "val")})


if __name__ == "__main__":
    main()
