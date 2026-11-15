"""Embedding and text / handwriting boxes of the top-up shards (tools/pages_topup.py -> pages_1m.py build: datasets pages-768x1024-18..21)
in two Kaggle runs of 2 x T4 each, every run on its own datasets: first pagetype/embed_kaggle.py as it is (both GPUs), then the detector
(handwriting_detection models/yolo26s-ds3k3250v3.pt, fp16, the 768 x 1024 page as is at imgsz 1024, conf floor 0.25), one process per GPU,
one output per input shard in the format of UVTM/Labels/hw_detection (ids, offsets, cls 0 hw / 1 print, conf uint8, failed).
Data: Layouts/archive/info/pages_topup/kaggle/<run>.

  python tools/kaggle_topup.py weights           # the weights as the private dataset pavel9860/hwdet-yolo26s-weights (once)
  python tools/kaggle_topup.py push A|B          # waits for the run's datasets, pushes its kernel
  python tools/kaggle_topup.py fetch A|B         # status; when complete: output -> <run>/out (emb/embeddings.npy, ids.npy; hw/*.npz)"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dspages.hand.config import PATHS

WORK = Path(PATHS.layouts) / "archive" / "info" / "pages_topup" / "kaggle"
EMBED = Path("/run/media/p/D/BUISNESS/corpus_collecting/pagetype/embed_kaggle.py")
WEIGHTS = Path("/run/media/p/D/BUISNESS/handwriting_detection/models/hwdet-s-1.0.0.pt")
WEIGHTS_DS = "pavel9860/hwdet-yolo26s-weights"
RUNS = {"A": (18, 19), "B": (20, 21)}
KAGGLE = [sys.executable, "-W", "ignore", "-m", "kaggle"]

KERNEL = r'''
import glob, io, json, os, re, subprocess, sys
import numpy as np
W = "/kaggle/working"
open(f"{W}/embed_kaggle.py", "w").write(EMBED_SRC)
subprocess.run([sys.executable, "-u", f"{W}/embed_kaggle.py", "/kaggle/input", f"{W}/emb"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.175"], check=True)
DET = r"""
import glob, io, re, sys, numpy as np
from PIL import Image
from ultralytics import YOLO
gpu, n = int(sys.argv[1]), int(sys.argv[2])
w = glob.glob("/kaggle/input/**/yolo26s-ds3k3250v3.pt", recursive=True)[0]
m = YOLO(w)
shards = sorted(glob.glob("/kaggle/input/**/pages_*.npz", recursive=True))[gpu::n]
for s in shards:
    ds = re.search(r"pages-768x1024-(\d+)", s).group(1)
    z = np.load(s)
    ids, off, jpg = z["ids"], z["offsets"], z["jpeg"]
    out_ids, offs, cls, conf, failed = [], [0], [], [], []
    for a in range(0, len(ids), 32):
        imgs, keep = [], []
        for i in range(a, min(a + 32, len(ids))):
            try:
                imgs.append(np.asarray(Image.open(io.BytesIO(jpg[off[i]:off[i + 1]].tobytes())).convert("RGB"))[:, :, ::-1])
                keep.append(i)
            except Exception:
                failed.append(ids[i])
        for i, r in zip(keep, m.predict(imgs, imgsz=1024, conf=0.25, quantize=16, device=gpu, verbose=False, batch=32)):
            out_ids.append(ids[i])
            cls += r.boxes.cls.int().tolist()
            conf += (r.boxes.conf * 255).round().int().tolist()
            offs.append(len(cls))
    name = s.rsplit("/", 1)[-1]
    np.savez(f"/kaggle/working/hw/d{ds}__{name}", ids=np.array(out_ids, np.int32).reshape(-1, 2), offsets=np.array(offs, np.int64),
             cls=np.array(cls, np.uint8), conf=np.array(conf, np.uint8), failed=np.array(failed, np.int32).reshape(-1, 2))
    print(f"gpu{gpu} d{ds}__{name}: {len(out_ids)} pages, {len(cls)} boxes, {len(failed)} failed", flush=True)
"""
os.makedirs(f"{W}/hw", exist_ok=True)
open(f"{W}/det.py", "w").write(DET)
ps = [subprocess.Popen([sys.executable, "-u", f"{W}/det.py", str(g), "2"]) for g in range(2)]
assert all(p.wait() == 0 for p in ps)
print("DONE", flush=True)
'''


def kaggle(*args, timeout=600):
    r = subprocess.run(KAGGLE + list(args), capture_output=True, text=True, timeout=timeout)
    return "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "outdated" not in l and "frozen" not in l)


def cmd_weights(_):
    d = WORK / "weights"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(WEIGHTS, d / WEIGHTS.name)
    (d / "dataset-metadata.json").write_text(json.dumps({"title": WEIGHTS_DS.split("/")[1], "id": WEIGHTS_DS, "licenses": [{"name": "CC-BY-NC-SA-4.0"}]}))
    print(kaggle("datasets", "create", "-p", str(d), timeout=1800))


def cmd_push(a):
    sets = [f"pavel9860/pages-768x1024-{i}" for i in RUNS[a.run]]
    t0 = time.time()
    while not all("pages_" in kaggle("datasets", "files", s) for s in sets):
        if time.time() - t0 > 5 * 3600:
            raise SystemExit(f"datasets {sets} not on Kaggle after 5 h")
        time.sleep(300)
    d = WORK / a.run / "kernel"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    (d / "run.py").write_text(f"EMBED_SRC = {EMBED.read_text()!r}\n" + KERNEL)
    kid = f"pavel9860/topup-emb-hw-{a.run.lower()}"
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": kid, "title": kid.split("/")[1], "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "keywords": ["gpu"], "dataset_sources": sets + [WEIGHTS_DS],
        "kernel_sources": [], "competition_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4"}))
    print(kaggle("kernels", "push", "-p", str(d)))


def cmd_fetch(a):
    kid = f"pavel9860/topup-emb-hw-{a.run.lower()}"
    st = kaggle("kernels", "status", kid)
    print(st)
    if "COMPLETE" in st.upper():
        out = WORK / a.run / "out"
        out.mkdir(parents=True, exist_ok=True)
        print(kaggle("kernels", "output", kid, "-p", str(out), timeout=3600)[-500:])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("weights")
    for n in ("push", "fetch"):
        p = sp.add_parser(n)
        p.add_argument("run", choices=sorted(RUNS))
    a = ap.parse_args()
    {"weights": cmd_weights, "push": cmd_push, "fetch": cmd_fetch}[a.cmd](a)
