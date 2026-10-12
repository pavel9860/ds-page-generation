"""Kaggle kernel: fine-tunes YOLO26 s (GPU 0) and m (GPU 1) on the hwdet dataset, one process per GPU, then validates
and exports ONNX. Micro-batch from the memory bench (T4, 1024 px): s 16, m 8; nbs=32 makes an optimizer step 32 pages."""
import glob
import os
import signal
import subprocess
import sys

W = "/kaggle/working"
HOURS = 0.06       # training wall time; the real run is 1.4
FRACTION = 0.1     # share of the train pages; the real run is 1.0
LANES = {0: ("s", 16), 1: ("m", 8)}

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.175", "onnx", "onnxslim"], check=True)
src = os.path.dirname(glob.glob("/kaggle/input/**/data.yaml", recursive=True)[0])
data = f"{W}/data.yaml"
open(data, "w").write(f"path: {src}\ntrain: train/images\nval: val/images\nnames:\n  0: handwriting\n  1: print\n")

CODE = """
import shutil
from ultralytics import YOLO
YOLO("yolo26{size}.pt").train(data="{data}", imgsz=1024, epochs=40, time={hours}, fraction={fraction}, batch={batch}, nbs=32,
    device=0, workers=2, cache=False, optimizer="AdamW", lr0=0.001, cos_lr=True, warmup_epochs=1, patience=4, close_mosaic=1,
    mosaic=0.5, fliplr=0.0, degrees=2.0, scale=0.5, translate=0.05, hsv_h=0.01, project="{W}/runs", name="hw_{size}",
    exist_ok=True, plots=False)
best = YOLO("{W}/runs/hw_{size}/weights/best.pt")
best.val(data="{data}", imgsz=1024, batch=8, device=0, plots=False)
shutil.copy("{W}/runs/hw_{size}/weights/best.pt", "{W}/hwdet-yolo26{size}.pt")
shutil.copy(best.export(format="onnx", imgsz=1024, opset=17, dynamic=True, simplify=True), "{W}/hwdet-yolo26{size}.onnx")
"""

procs = []
for gpu, (size, batch) in LANES.items():
    os.makedirs(f"{W}/lane_{size}", exist_ok=True)       # own cwd: ultralytics downloads (weights, AMP check) must not race
    code = CODE.format(size=size, batch=batch, data=data, hours=HOURS, fraction=FRACTION, W=W)
    procs.append(subprocess.Popen([sys.executable, "-u", "-c", code], cwd=f"{W}/lane_{size}", start_new_session=True,
                                  env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))))
for p in procs:
    try:
        p.wait(timeout=HOURS * 3600 + 600)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
