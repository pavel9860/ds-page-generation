"""The Kaggle script of one ingest run (2 x T4, internet on): two GPU stages over the shards pages_*.npz of the attached datasets, outputs in /kaggle/working:
  1. SigLIP 2 embedding (corpus_collecting pagetype/embed_kaggle.py, both GPUs)      -> embeddings.npy, ids.npy
  2. handwriting / print boxes (yolo26s, fp16, imgsz 1024, conf floor 0.25), one process per GPU -> hw/<shard>.npz (ids, offsets, cls 0 hw / 1 print, conf, failed)
Fill / pic is CPU work: it runs as its own CPU kernel (dspages.fillpic kernel.py, `kaggle.push`), never on the GPU session.
`kaggle.push` fills EMBED_SRC (the source of the embedding script) in front of this file."""

import os
import subprocess
import sys

W = "/kaggle/working"
open(f"{W}/embed_kaggle.py", "w").write(EMBED_SRC)  # noqa: F821
subprocess.run([sys.executable, "-u", f"{W}/embed_kaggle.py", "/kaggle/input", W], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.175"], check=True)

DET = r"""
import glob, io, os, sys, numpy as np
from PIL import Image
from ultralytics import YOLO
gpu, n = int(sys.argv[1]), int(sys.argv[2])
m = YOLO(glob.glob("/kaggle/input/**/yolo26s-ds3k3250v3.pt", recursive=True)[0])
for s in sorted(glob.glob("/kaggle/input/**/pages_*.npz", recursive=True))[gpu::n]:
    z = np.load(s)
    ids, off, jpg = z["ids"], z["offsets"], z["jpeg"]
    if int(os.environ.get("LIMIT", 0)):
        ids = ids[:int(os.environ["LIMIT"])]
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
    np.savez("/kaggle/working/hw/" + name, ids=np.array(out_ids, np.int32).reshape(-1, 2), offsets=np.array(offs, np.int64),
             cls=np.array(cls, np.uint8), conf=np.array(conf, np.uint8), failed=np.array(failed, np.int32).reshape(-1, 2))
    print(f"gpu{gpu} {name}: {len(out_ids)} pages, {len(cls)} boxes, {len(failed)} failed", flush=True)
"""
os.makedirs(f"{W}/hw", exist_ok=True)
open(f"{W}/det.py", "w").write(DET)
ps = [subprocess.Popen([sys.executable, "-u", f"{W}/det.py", str(g), "2"]) for g in range(2)]
assert all(p.wait() == 0 for p in ps)
print("DONE", flush=True)
