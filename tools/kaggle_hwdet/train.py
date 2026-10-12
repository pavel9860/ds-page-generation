"""Kaggle kernel: fine-tunes YOLO26 s (GPU 0) and m (GPU 1) on the hwdet dataset, one process per GPU (no DDP), each
for HOURS of wall time (ultralytics sizes the epochs and the cosine schedule from it), then validates and exports ONNX.

Micro-batch from the memory bench (T4, 1024 px, densest mosaic batches): s 16 (peak 13.2 GB; 24 and 32 ran out of
memory), m 8 (12.7 GB; 16 and 24 ran out); nbs=32 makes the optimizer step a batch of 32 by accumulation.
Outputs (/kaggle/working): hwdet-yolo26{s,m}.{pt,onnx}, train_meta_{s,m}.json, train_{s,m}.log, runs/."""
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

W = "/kaggle/working"
ULTRALYTICS = "8.4.175"
HOURS = 1.4
LANES = {0: ("s", 16), 1: ("m", 8)}
PARAMS = dict(
    imgsz=1024, epochs=40, time=HOURS, nbs=32, workers=2, cache=False, seed=0, deterministic=False,
    optimizer="AdamW", lr0=0.001, lrf=0.01, cos_lr=True, warmup_epochs=1, weight_decay=0.0005, patience=4,
    close_mosaic=1, mosaic=0.5, scale=0.5, translate=0.05, degrees=2.0, shear=0.0, fliplr=0.0, hsv_h=0.01,
    hsv_s=0.5, hsv_v=0.3, plots=False, device=0,
)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input", f"ultralytics=={ULTRALYTICS}", "onnx", "onnxslim"],
               check=True, timeout=600)
src = os.path.dirname(glob.glob("/kaggle/input/**/data.yaml", recursive=True)[0])
data = f"{W}/data.yaml"
open(data, "w").write(f"path: {src}\ntrain: train/images\nval: val/images\nnames:\n  0: handwriting\n  1: print\n")
manifest = json.load(open(f"{src}/manifest.json")) if os.path.exists(f"{src}/manifest.json") else {}

CHILD = '''
import json, shutil, time
import torch
from ultralytics import YOLO
size, batch, W, data = {size!r}, {batch}, {W!r}, {data!r}
params = {params!r}
YOLO(f"yolo26{{size}}.pt").train(data=data, project=W + "/runs", name=f"hw_{{size}}", exist_ok=True, batch=batch, **params)
best = f"{{W}}/runs/hw_{{size}}/weights/best.pt"
model = YOLO(best)
m = model.val(data=data, imgsz=params["imgsz"], batch=8, device=0, plots=False)
onnx = model.export(format="onnx", imgsz=params["imgsz"], opset=17, dynamic=True, simplify=True)
shutil.copy(best, f"{{W}}/hwdet-yolo26{{size}}.pt")
shutil.copy(onnx, f"{{W}}/hwdet-yolo26{{size}}.onnx")
json.dump(dict(size=size, batch=batch, params=params, torch=torch.__version__,
               val=dict(zip(("precision", "recall", "mAP50", "mAP50-95"), map(float, m.box.mean_results()))),
               val_ap50=dict(zip(m.names.values(), map(float, m.box.ap50))), val_speed_ms=m.speed,
               finished=time.strftime("%Y-%m-%dT%H:%M:%S")), open(f"{{W}}/train_meta_{{size}}.json", "w"), indent=1)
'''

def pump(size, p, log):
    """Child output to the kernel's stdout (visible while it runs) and to the log file; progress-bar updates are
    printed at most every 30 s, every other line at once."""
    last = 0.0
    for raw in p.stdout:
        log.write(raw)
        log.flush()
        line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", raw).strip()
        if not line:
            continue
        bar = "it/s" in line or "s/it" in line or "━" in line
        if bar and time.time() - last < 30:
            continue
        last = time.time() if bar else last
        print(f"[{size} {time.time() - T0:5.0f}s] {line[:200]}", flush=True)


T0 = time.time()
procs = {}
for gpu, (size, batch) in LANES.items():
    code = CHILD.format(size=size, batch=batch, W=W, data=data, params=PARAMS)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    procs[size] = subprocess.Popen([sys.executable, "-u", "-c", code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True, text=True, bufsize=1)
    threading.Thread(target=pump, args=(size, procs[size], open(f"{W}/train_{size}.log", "w")), daemon=True).start()
deadline = time.time() + HOURS * 3600 + 25 * 60
for size, p in procs.items():
    try:
        p.wait(timeout=max(1, deadline - time.time()))
    except subprocess.TimeoutExpired:
        print(f"{size}: TIMEOUT", flush=True)
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
    print(f"{size}: exit {p.returncode}", flush=True)
for size in LANES.values():
    f = f"{W}/train_meta_{size[0]}.json"
    if os.path.exists(f):
        print(open(f).read(), flush=True)
shutil.copy(data, f"{W}/data_used.yaml")
