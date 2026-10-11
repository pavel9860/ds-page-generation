import glob
import os
import shutil
import subprocess
import sys

W = "/kaggle/working"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.175", "onnx", "onnxslim"], check=True)
src = os.path.dirname(glob.glob("/kaggle/input/**/data.yaml", recursive=True)[0])
data = f"{W}/data.yaml"
open(data, "w").write(f"path: {src}\ntrain: train/images\nval: val/images\nnames:\n  0: handwriting\n  1: print\n")
w = (glob.glob("/kaggle/input/**/hwdet-init.pt", recursive=True) or glob.glob("/kaggle/input/**/yolo26n-doclaynet.pt", recursive=True))[0]
train = f"""
from ultralytics import YOLO
YOLO("{w}").train(data="{data}", imgsz=1024, epochs=40, time=0.35, batch=64, device=[0, 1], workers=4, cache=False,
    optimizer="AdamW", lr0=0.001, cos_lr=True, warmup_epochs=0, mosaic=0.5, close_mosaic=3, fliplr=0.0, degrees=0.0,
    scale=0.35, translate=0.05, hsv_h=0.01, project="{W}/runs", name="hw", exist_ok=True, plots=False)
"""
with open(f"{W}/train.log", "w") as log:
    try:
        subprocess.run([sys.executable, "-u", "-c", train], stdout=log, stderr=subprocess.STDOUT, timeout=50 * 60)
    except subprocess.TimeoutExpired:
        log.write("\nTIMEOUT\n")
best = f"{W}/runs/hw/weights/best.pt"
if not os.path.exists(best):
    best = f"{W}/runs/hw/weights/last.pt"
from ultralytics import YOLO

b = YOLO(best)
b.val(data=data, imgsz=1024, batch=32, device=0)
p = b.export(format="onnx", imgsz=1024, opset=17, dynamic=True, simplify=True, end2end=False)
shutil.copy(best, f"{W}/hwdet-yolo26n.pt")
shutil.copy(p, f"{W}/hwdet-yolo26n.onnx")
