"""Kaggle kernel: fine-tunes YOLO26 on the hwdet dataset, validates, exports ONNX and writes train_meta.json.

Inputs (/kaggle/input): the dataset (data.yaml, train/, val/, manifest.json) and an init checkpoint: `hwdet-init.pt`
(continue a round), else `yolo26n-doclaynet.pt` (size n) or the Ultralytics pretrained `yolo26<SIZE>.pt` (downloaded).
Outputs (/kaggle/working): hwdet-yolo26<SIZE>.{pt,onnx}, train_meta.json, train.log, runs/.
"""
import glob
import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time

W = "/kaggle/working"
ULTRALYTICS = "8.4.175"
SIZE = "s"
HOURS = 2.0
PARAMS = dict(
    imgsz=1024, epochs=25, time=HOURS, batch=64, device=[0, 1], workers=4, cache=False, seed=0, deterministic=False,
    optimizer="AdamW", lr0=0.001, lrf=0.01, cos_lr=True, warmup_epochs=1, weight_decay=0.0005, patience=8,
    close_mosaic=5, mosaic=0.5, scale=0.5, translate=0.05, degrees=2.0, shear=0.0, fliplr=0.0, hsv_h=0.01,
    hsv_s=0.5, hsv_v=0.3, plots=False, save_period=5,
)

logging.basicConfig(filename=f"{W}/train.log", level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("train")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"ultralytics=={ULTRALYTICS}", "onnx", "onnxslim"], check=True)

yaml = glob.glob("/kaggle/input/**/data.yaml", recursive=True)[0]
src = os.path.dirname(yaml)
data = f"{W}/data.yaml"
open(data, "w").write(f"path: {src}\ntrain: train/images\nval: val/images\nnames:\n  0: handwriting\n  1: print\n")
init = (glob.glob("/kaggle/input/**/hwdet-init.pt", recursive=True)
        or glob.glob(f"/kaggle/input/**/yolo26{SIZE}-doclaynet.pt", recursive=True) or [f"yolo26{SIZE}.pt"])[0]
manifest = os.path.join(src, "manifest.json")
meta = dict(size=SIZE, init=os.path.basename(init), params=PARAMS, ultralytics=ULTRALYTICS,
            manifest_sha256=hashlib.sha256(open(manifest, "rb").read()).hexdigest() if os.path.exists(manifest) else None,
            started=time.strftime("%Y-%m-%dT%H:%M:%S"))
log.info("init %s, params %s", init, PARAMS)

code = f"""
from ultralytics import YOLO
YOLO({init!r}).train(data={data!r}, project={W + '/runs'!r}, name="hw", exist_ok=True, **{PARAMS!r})
"""
with open(f"{W}/train.log", "a") as out:
    proc = subprocess.Popen([sys.executable, "-u", "-c", code], stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        proc.wait(timeout=int(HOURS * 3600 + 15 * 60))
    except subprocess.TimeoutExpired:
        log.warning("TIMEOUT")
        os.killpg(proc.pid, signal.SIGKILL)      # the DDP launcher and both ranks, not only the parent
        proc.wait()

best = f"{W}/runs/hw/weights/best.pt"
if not os.path.exists(best):
    best = f"{W}/runs/hw/weights/last.pt"
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

model = YOLO(best)
m = model.val(data=data, imgsz=PARAMS["imgsz"], batch=32, device=0, plots=False)
meta.update(torch=torch.__version__, weights=os.path.basename(best), finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
            val=dict(zip(("precision", "recall", "mAP50", "mAP50-95"), map(float, m.box.mean_results()))),
            val_ap50=dict(zip(m.names.values(), map(float, m.box.ap50))), val_speed_ms=m.speed)
onnx = model.export(format="onnx", imgsz=PARAMS["imgsz"], opset=17, dynamic=True, simplify=True)
shutil.copy(best, f"{W}/hwdet-yolo26{SIZE}.pt")
shutil.copy(onnx, f"{W}/hwdet-yolo26{SIZE}.onnx")
json.dump(meta, open(f"{W}/train_meta.json", "w"), indent=1)
log.info("done %s", meta["val"])
