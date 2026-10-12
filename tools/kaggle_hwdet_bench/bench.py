"""Kaggle bench of YOLO26n training steps on 2xT4, one process per GPU, no DDP. Lane 0 (GPU 0): batch 8, 16, 32 eager.
Lane 1 (GPU 1): batch 16 eager, then batch 16 torch.compile. Every config is its own process with a timeout, 100 steps,
each step printed (ms of that step, not cumulative) with peak memory; the whole kernel is bounded so that it finishes and
Kaggle uploads its log (a cancelled script kernel has an empty log)."""
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time

W = "/kaggle/working"
T0 = time.time()
STEPS, NB, CONFIG_TIMEOUT, TOTAL_TIMEOUT = 100, 8, 480, 1500
LANES = {0: [("eager", 8), ("eager", 16), ("eager", 32)], 1: [("eager", 16), ("compile", 16)]}
lock = threading.Lock()


def say(*a):
    with lock:
        print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


def sh(cmd, timeout=120):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return f"TIMEOUT {cmd}"


def find(name):
    found = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return found[0] if found else None


def setup():
    say("cpus", os.cpu_count(), "|", sh("free -g | sed -n 2p"), "| /dev/shm", sh("df -h /dev/shm | tail -1"))
    say("gpus", sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader").replace("\n", " ; "))
    say("pip", sh(f"{sys.executable} -m pip install -q --no-input ultralytics==8.4.175 2>&1 | tail -2", 300) or "ok")
    yaml = find("data.yaml")
    src = os.path.dirname(yaml)
    imgs = sorted(os.listdir(f"{src}/train/images"))[:200]
    t = time.time()
    n = sum(len(open(f"{src}/train/images/{f}", "rb").read()) for f in imgs)
    say(f"read /kaggle/input: {n / 1e6:.0f} MB in {time.time() - t:.1f}s = {n / 1e6 / (time.time() - t):.0f} MB/s")
    os.makedirs(f"{W}/probe", exist_ok=True)
    t = time.time()
    for f in imgs:
        shutil.copy(f"{src}/train/images/{f}", f"{W}/probe/{f}")
    n = sum(len(open(f"{W}/probe/{f}", "rb").read()) for f in imgs)
    say(f"copy to /kaggle/working {time.time() - t:.1f}s; re-read from local disk ok")
    open(f"{W}/data.yaml", "w").write(f"path: {src}\ntrain: train/images\nval: val/images\nnames:\n  0: handwriting\n  1: print\n")
    return src, find("hwdet-init.pt") or find("yolo26n-doclaynet.pt")


def config(mode, bs, src, weights, gpu):
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_yolo_dataset
    from ultralytics.data.build import build_dataloader
    from ultralytics.utils import DEFAULT_CFG

    tag = f"gpu{gpu} {mode} bs{bs}"
    cfg = get_cfg(DEFAULT_CFG, dict(imgsz=1024, batch=bs, mosaic=0.5, close_mosaic=3, fliplr=0.0, degrees=0.0, scale=0.35,
                                    translate=0.05, hsv_h=0.01, workers=2, cache=False))
    data = {"names": {0: "handwriting", 1: "print"}, "nc": 2, "channels": 3, "path": src}
    ds = build_yolo_dataset(cfg, f"{src}/train/images", bs, data, mode="train", stride=32)
    dl = build_dataloader(ds, bs, 2, shuffle=True, rank=-1)
    t = time.time()
    batches = []
    for b in dl:
        batches.append(b)
        if len(batches) == NB:
            break
    objs = [int(np.bincount(b["batch_idx"].numpy().astype(int)).max()) for b in batches]
    say(f"[{tag}] {NB} batches loaded: {(time.time() - t) / NB * 1000:.0f} ms per batch (2 workers), "
        f"max objects per image {max(objs)}")
    del dl
    dev = torch.device("cuda:0")
    model = YOLO(weights).model.to(dev).float().train()
    model.args, model.nc, model.names = cfg, 2, data["names"]
    for p in model.parameters():
        p.requires_grad = True
    model.criterion = model.init_criterion()
    fwd = torch.compile(model) if mode == "compile" else model
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler()
    torch.cuda.reset_peak_memory_stats()
    ms = []
    status = "ok"
    for i in range(STEPS):
        b = {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v) for k, v in batches[i % NB].items()}
        b["img"] = b["img"].float() / 255
        torch.cuda.synchronize()
        t = time.time()
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                preds = fwd(b["img"])
                loss = model.criterion(preds, b)[0].sum()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
        except torch.OutOfMemoryError:
            status = f"OOM at step {i}"
            say(f"[{tag}] {status}; peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
            break
        ms.append((time.time() - t) * 1000)
        say(f"[{tag}] step {i:3d} {ms[-1]:7.1f} ms  {bs / ms[-1] * 1000:6.1f} img/s  "
            f"peak {torch.cuda.max_memory_allocated() / 2**30:5.2f} GB alloc / "
            f"{torch.cuda.max_memory_reserved() / 2**30:5.2f} GB reserved")
    tal_oom = bool(getattr(model.criterion.assigner, "_oom_warned", False))
    steady = ms[10:] if len(ms) > 10 else ms
    res = {"tag": tag, "status": status, "tal_oom_retry": tal_oom, "steps": len(ms), "first_step_ms": ms[0] if ms else None,
           "median_ms": float(np.median(steady)) if steady else None, "p95_ms": float(np.percentile(steady, 95)) if steady else None,
           "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2**30, "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30}
    say(f"[{tag}] RESULT {json.dumps(res)}")
    with open(f"{W}/bench.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")


def lane(gpu, jobs, src, weights, deadline, procs):
    for mode, bs in jobs:
        left = deadline - time.time()
        if left < 30:
            say(f"gpu{gpu} {mode} bs{bs}: skipped, no time left")
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        p = subprocess.Popen([sys.executable, "-u", __file__, "run", mode, str(bs), src, weights, str(gpu)], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        procs.append(p)
        timer = threading.Timer(min(CONFIG_TIMEOUT, left), lambda p=p: os.killpg(p.pid, 9))
        timer.start()
        for line in p.stdout:
            line = line.replace("\r", "\n").strip().split("\n")[-1]
            if line.startswith("["):
                with lock:
                    print(line, flush=True)
            elif line:
                say(f"gpu{gpu} | {line[:200]}")
        p.wait()
        timer.cancel()
        if p.returncode:
            say(f"gpu{gpu} {mode} bs{bs}: process ended with code {p.returncode} (timeout kill = -9)")


def main():
    src, weights = setup()
    say("data", src, "weights", weights)
    deadline = time.time() + TOTAL_TIMEOUT
    procs = []
    threads = [threading.Thread(target=lane, args=(g, j, src, weights, deadline, procs)) for g, j in LANES.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    say("summary")
    for l in open(f"{W}/bench.jsonl") if os.path.exists(f"{W}/bench.jsonl") else []:
        print(l.strip(), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        T0 = time.time()
        config(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6])
    else:
        main()
