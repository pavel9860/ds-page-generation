import glob
import os
import subprocess
import sys

W = "/kaggle/working"
print("before pip", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.175"], check=True)
print("after pip", flush=True)
src = os.path.dirname(glob.glob("/kaggle/input/**/det_s.pt", recursive=True)[0])
print("after data", src, flush=True)
RUNS = {0: ("cls_pretrained", 'YOLO("yolo26s-cls.pt")'), 1: ("det_init", f'YOLO("yolo26s-cls.yaml").load("{src}/det_s.pt")')}
code = """
from ultralytics import YOLO
m = {model}
m.train(data="{src}", imgsz=224, epochs=15, batch=128, device={gpu}, workers=2, optimizer="AdamW", lr0=0.001,
        cos_lr=True, patience=5, project="{W}/runs", name="{name}", exist_ok=True, plots=False)
"""
procs = [subprocess.Popen([sys.executable, "-u", "-c", code.format(model=m, src=src, gpu=g, W=W, name=n)])
         for g, (n, m) in RUNS.items()]
print("training started", flush=True)
for p in procs:
    p.wait()
print("done", flush=True)
