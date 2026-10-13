import subprocess
import sys

DATA = "/kaggle/input/datasets/pavel9860/hwdet-lang"
RUNS = {0: ("cls_pretrained", "yolo26s-cls.pt"), 1: ("det_init", f"{DATA}/det_s.pt")}
HOURS, EPOCHS, BATCH = 0.5, 20, 128
CODE = """
import glob, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.tasks import ClassificationModel

DATA, NAME, INIT, HOURS, EPOCHS, BATCH = {data!r}, {name!r}, {init!r}, {hours}, {epochs}, {batch}
dev = torch.device("cuda:0")
keys = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{{DATA}}/train/*.npz"))
assert keys == sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{{DATA}}/val/*.npz"))

def load(split):
    xs, ys, ws = [], [], []
    for i, k in enumerate(keys):
        z = np.load(f"{{DATA}}/{{split}}/{{k}}.npz")
        xs.append(z["x"]); ys.append(np.full(len(z["x"]), i)); ws.append(z["w"].astype(np.int32) + 1000 * i)
    return torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(ys)), np.concatenate(ws)

xt, yt, _ = load("train")
xv, yv, wv = load("val")
xt, yt = xt.to(dev), yt.to(dev)
model = ClassificationModel("yolo26s-cls.yaml", nc=len(keys), verbose=False)
src = YOLO(INIT).model
model.load(src)
model.to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
steps = EPOCHS * (len(xt) // BATCH)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.05)
scaler = torch.amp.GradScaler()
gray3 = lambda x: (x.float() / 255).unsqueeze(1).expand(-1, 3, -1, -1)

def augment(x):
    n = x.shape[0]
    theta = torch.zeros(n, 2, 3, device=dev)
    s = 1 + (torch.rand(n, device=dev) - 0.5) * 0.3
    a = (torch.rand(n, device=dev) - 0.5) * 0.2
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 1, 0], theta[:, 1, 1] = s * a.cos(), -s * a.sin(), s * a.sin(), s * a.cos()
    theta[:, :, 2] = (torch.rand(n, 2, device=dev) - 0.5) * 0.2
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), padding_mode="border", align_corners=False)
    return (x - 0.5) * (1 + (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.4) + 0.5

def evaluate():
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", torch.float16):
        for i in range(0, len(xv), 256):
            out.append(model(gray3(xv[i:i + 256].to(dev)))[0].float().cpu())
    p = torch.cat(out)
    top5 = p.topk(min(5, len(keys)), 1).indices
    top1 = (top5[:, 0] == yv).float().mean().item()
    top5 = (top5 == yv[:, None]).any(1).float().mean().item()
    votes = {{}}
    for w, row in zip(wv, p.log().numpy()):
        votes.setdefault(w, []).append(row)
    writer = np.mean([np.sum(v, 0).argmax() == w // 1000 for w, v in votes.items()])
    per = [(p.argmax(1)[yv == i] == i).float().mean().item() for i in range(len(keys))]
    model.train()
    return top1, top5, float(writer), per

t0 = time.time()
best = 0
model.train()
for epoch in range(EPOCHS):
    perm = torch.randperm(len(xt), device=dev)
    loss_sum = 0.0
    for i in range(len(xt) // BATCH):
        idx = perm[i * BATCH:(i + 1) * BATCH]
        with torch.autocast("cuda", torch.float16):
            loss = F.cross_entropy(model(augment(gray3(xt[idx]))), yt[idx], label_smoothing=0.1)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        loss_sum += loss.item()
    top1, top5, writer, per = evaluate()
    print(f"{{NAME}} epoch {{epoch + 1}} loss {{loss_sum / (len(xt) // BATCH):.3f}} top1 {{top1:.4f}} top5 {{top5:.4f}} "
          f"writer-vote {{writer:.4f}} {{time.time() - t0:.0f}}s", flush=True)
    if top1 > best:
        best = top1
        torch.save(model.state_dict(), f"/kaggle/working/{{NAME}}.pt")
        json.dump(dict(keys=keys, top1=top1, top5=top5, writer_vote=writer, per_class=dict(zip(keys, per))),
                  open(f"/kaggle/working/{{NAME}}.json", "w"))
    if time.time() - t0 > HOURS * 3600:
        break
"""
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.175"], check=True)
procs = [subprocess.Popen([sys.executable, "-u", "-c", CODE.format(data=DATA, name=n, init=w, hours=HOURS, epochs=EPOCHS,
                                                                    batch=BATCH), ],
                          env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": str(g)}) for g, (n, w) in RUNS.items()]
for p in procs:
    p.wait()
