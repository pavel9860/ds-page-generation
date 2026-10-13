import os
import subprocess
import sys

W = "/kaggle/working"
LANES = {0: ("s", 16), 1: ("m", 8)}
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.175", "onnx"], check=True)
src = "/kaggle/input/datasets/pavel9860/hw-ds-0-5-0"
data = f"{W}/data.yaml"
open(data, "w").write(f"path: {src}\ntrain: train/images\nval: val/images\nnames:\n  0: handwriting\n  1: print\n")
train = f"""
from ultralytics import YOLO
YOLO("yolo26{{size}}.pt").train(data="{data}", imgsz=1024, epochs=40, time=1.4, batch={{batch}}, nbs=32, device={{gpu}},
    workers=2, cache=False, optimizer="AdamW", lr0=0.001, cos_lr=True, warmup_epochs=1, patience=4, close_mosaic=1,
    mosaic=0.5, fliplr=0.0, degrees=0.0, scale=0.35, translate=0.05, hsv_h=0.01, project="{W}/runs", name="hw_{{size}}",
    exist_ok=True, plots=False)
best = YOLO("{W}/runs/hw_{{size}}/weights/best.pt")
best.val(data="{data}", imgsz=1024, batch=8, device={{gpu}}, plots=False)
best.export(format="onnx", imgsz=1024, opset=17, dynamic=True, simplify=True)
"""
procs = [subprocess.Popen([sys.executable, "-u", "-c", train.format(size=s, batch=b, gpu=g)]) for g, (s, b) in LANES.items()]
for p in procs:
    p.wait()
