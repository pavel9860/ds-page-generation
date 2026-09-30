"""Split a created dataset into DIR/{train,test,val}/ by moving (renaming) each sample's files, by a seeded
shuffle of the sample indices: samples/<i>.npz to DIR/<split>/<i>.npz, layouts/ and geometry/ files of the same
index to DIR/<split>/layouts/ and DIR/<split>/geometry/ when present.

python tools/split.py DIR [--frac 0.96 0.02 0.02] [--seed 0]
"""
import argparse
import os
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
    files = {part: {} for part in PARTS}                  # one listing per part, not one glob per sample
    for part in PARTS:
        if (out / part).is_dir():
            for e in os.scandir(out / part):
                files[part].setdefault(int(e.name.split(".")[0]), []).append(e.name)
    idx = sorted(files["samples"])
    perm = np.random.default_rng(a.seed).permutation(idx)
    n_test, n_val = round(len(idx) * a.frac[1]), round(len(idx) * a.frac[2])
    names = dict(zip(perm.tolist(), ["test"] * n_test + ["val"] * n_val + ["train"] * (len(idx) - n_test - n_val)))
    for split in ("train", "test", "val"):
        for part in PARTS:
            (out / split / ("" if part == "samples" else part)).mkdir(parents=True, exist_ok=True)
    for i, split in names.items():
        for part in PARTS:
            for name in files[part].get(i, ()):
                os.rename(out / part / name, out / split / ("" if part == "samples" else part) / name)
    for part in PARTS:
        for split in ("train", "test", "val"):
            d = out / split / part
            if part != "samples" and d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        if (out / part).is_dir() and not any((out / part).iterdir()):
            (out / part).rmdir()
    with open(out / "splits.txt", "w") as f:
        for i in sorted(names):
            f.write(f"{i:07d} {names[i]}\n")
    print({s: sum(v == s for v in names.values()) for s in ("train", "test", "val")})


if __name__ == "__main__":
    main()
