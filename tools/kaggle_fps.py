"""The 100K page set of tools/ds100k.py on a Kaggle GPU: batched farthest-point sampling on the full 1536-d unit page vectors of the pool.
Every round takes up to B pages, the farthest from the set, each also farther from the pages taken in the round than from the set
(B = 100: on 50K pages the covering radius is that of one-by-one sampling, 0.6299 vs 0.6301, 40 x faster). Data: ds100k/kaggle.

  python tools/kaggle_fps.py upload        # pool vectors (fp16, 4 parts) -> dataset pavel9860/ds100k-pool
  python tools/kaggle_fps.py push [--n 100000]
  python tools/kaggle_fps.py fetch         # -> ds100k/select.parquet (order = pick order, r = distance when picked), cover.npz"""

import argparse
import json
import subprocess
import sys

import diverse_ds as DD
import ds100k as D
import numpy as np
import pandas as pd

WORK = D.OUT / "kaggle"
DS = "pavel9860/ds100k-pool"
KID = "pavel9860/ds100k-fps"
PARTS, CHUNK = 4, 100_000
KAGGLE = [sys.executable, "-W", "ignore", "-m", "kaggle"]

KERNEL = r"""
import glob, sys, time, numpy as np, torch
N_PICK, B = int(sys.argv[1]) if len(sys.argv) > 1 else N, 100
X = torch.from_numpy(np.concatenate([np.load(f) for f in sorted(glob.glob("/kaggle/input/**/pool_*.npy", recursive=True))])).cuda()
n = len(X)
d = torch.full((n,), float("inf"), device="cuda")
cell = torch.zeros(n, dtype=torch.int64, device="cuda")              # nearest set page (its pick order) of every pool page
def upd(P):                                     # unit rows: |x - p|^2 = 2 - 2 x.p
    m, i = (2 - 2 * (X @ X[P].T).float()).clamp_(min=0).min(1)
    closer = m < d
    d[closer], cell[closer] = m[closer], i[closer] + len(pick) - len(P)
acc, pick, rad = [0], [], []
r_acc = [float("inf")]
t0 = time.time()
while True:
    pick += acc; rad += r_acc
    upd(torch.tensor(acc, device="cuda"))
    if len(pick) >= N_PICK:
        break
    v, c = torch.topk(d, 4 * B)
    v, c = v.tolist(), c.tolist()
    acc, r_acc = [], []
    for dv, j in zip(v, c):
        if acc and float((2 - 2 * (X[acc] @ X[j]).float()).min()) < dv:
            continue
        acc.append(j); r_acc.append(dv ** 0.5)
        if len(acc) == B or len(pick) + len(acc) == N_PICK:
            break
    if len(pick) // 10000 != (len(pick) + len(acc)) // 10000:
        print(f"{len(pick) + len(acc)} picked, radius {r_acc[-1]:.4f}, {int(time.time() - t0)} s", flush=True)
r = d.sqrt().cpu().numpy()
np.savez("/kaggle/working/fps.npz", pick=np.array(pick, np.int64), rad=np.array(rad, np.float32), cover=r.astype(np.float32), cell=cell.cpu().numpy())
print("DONE", len(pick), "picks; covering radius", float(r.max()), int(time.time() - t0), "s", flush=True)
"""


def kaggle(*args, timeout=3600):
    r = subprocess.run(KAGGLE + list(args), capture_output=True, text=True, timeout=timeout)
    return "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "outdated" not in l)


def cmd_upload(_):
    P = pd.read_parquet(D.OUT / "pool.parquet", columns=["row"])
    E = np.load(D.EMB / "embeddings.npy", mmap_mode="r")
    d = WORK / "data"
    d.mkdir(parents=True, exist_ok=True)
    rows = P.row.to_numpy()
    for k, part in enumerate(np.array_split(rows, PARTS)):
        np.save(
            d / f"pool_{k}.npy",
            np.concatenate(
                [
                    DD.unit_rows(np.asarray(E[part[a : a + CHUNK]], np.float32)).astype(np.float16)
                    for a in range(0, len(part), CHUNK)
                ]
            ),
        )
    (d / "dataset-metadata.json").write_text(
        json.dumps({"title": DS.split("/")[1], "id": DS, "licenses": [{"name": "CC-BY-NC-SA-4.0"}]})
    )
    print(len(rows), "pool vectors;", kaggle("datasets", "create", "-p", str(d), timeout=7200)[-300:])


def cmd_push(a):
    d = WORK / "kernel"
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.py").write_text(f"N = {a.n}\n" + KERNEL)
    (d / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": KID,
                "title": KID.split("/")[1],
                "code_file": "run.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_tpu": False,
                "enable_internet": False,
                "dataset_sources": [DS],
                "kernel_sources": [],
                "competition_sources": [],
                "model_sources": [],
                "machine_shape": "NvidiaTeslaT4",
            }
        )
    )
    print(kaggle("kernels", "push", "-p", str(d)))


def cmd_fetch(_):
    st = kaggle("kernels", "status", KID)
    print(st)
    if "COMPLETE" not in st.upper():
        return
    out = WORK / "out"
    out.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", KID, "-p", str(out))[-300:])
    z = np.load(out / "fps.npz")
    P = pd.read_parquet(D.OUT / "pool.parquet")
    S = P.iloc[z["pick"]].assign(order=np.arange(len(z["pick"])), r=z["rad"]).reset_index(names="pool_i")
    S.to_parquet(D.OUT / "select.parquet")
    np.savez(
        D.OUT / "cover.npz", d=z["cover"], cell=z["cell"]
    )  # every pool page: distance to and pick order of its nearest set page
    print(
        len(S),
        "pages of",
        S.doc.nunique(),
        "documents; covering radius",
        round(float(z["cover"].max()), 4),
        "->",
        D.OUT / "select.parquet",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("upload")
    sp.add_parser("push").add_argument("--n", type=int, default=100_000)
    sp.add_parser("fetch")
    a = ap.parse_args()
    {"upload": cmd_upload, "push": cmd_push, "fetch": cmd_fetch}[a.cmd](a)
