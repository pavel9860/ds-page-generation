import glob
import os
import subprocess
import sys

W = "/kaggle/working"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.175", "onnx"], check=True)
src = os.path.dirname(glob.glob("/kaggle/input/**/data.yaml", recursive=True)[0])
data = f"{W}/data.yaml"
open(data, "w").write(f"path: {src}\ntrain: train/images\nval: val/images\nnames:\n  0: handwriting\n  1: print\n")
w = glob.glob("/kaggle/input/**/yolo26n-doclaynet.pt", recursive=True)[0]
train = f"""
from ultralytics import YOLO
YOLO("{w}").train(data="{data}", imgsz=1024, epochs=40, time=0.05, fraction=0.1, batch=16, device={{gpu}}, workers=2,
    cache=False, optimizer="AdamW", lr0=0.001, cos_lr=True, warmup_epochs=0, mosaic=0.5, close_mosaic=3, fliplr=0.0,
    degrees=0.0, scale=0.35, translate=0.05, hsv_h=0.01, project="{W}/runs", name="hw{{gpu}}", exist_ok=True, plots=False)
"""
procs = [subprocess.Popen([sys.executable, "-u", "-c", train.format(gpu=g)]) for g in (0, 1)]
for p in procs:
    p.wait(timeout=10 * 60)
