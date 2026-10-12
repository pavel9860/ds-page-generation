"""Kaggle CPU kernel: read speed of /kaggle/input/hwdet-synth, cold and warm, one thread and 8 threads, and a copy to
/kaggle/working. Finishes in about a minute; prints what ultralytics' check_file_speeds measures (5 random files, one
after another, MB/s of each) next to the real throughput."""
import glob
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

T0 = time.time()


def say(*a):
    print(f"[{time.time() - T0:5.1f}s]", *a, flush=True)


src = os.path.dirname(glob.glob("/kaggle/input/**/data.yaml", recursive=True)[0])
names = sorted(os.listdir(f"{src}/train/images"))
random.Random(0).shuffle(names)
paths = [f"{src}/train/images/{n}" for n in names]
say("files", len(paths), "cpus", os.cpu_count())


def read(p):
    with open(p, "rb") as f:
        return len(f.read())


def timed(label, ps, threads=1):
    t = time.perf_counter()
    with ThreadPoolExecutor(threads) as ex:
        n = sum(ex.map(read, ps))
    dt = time.perf_counter() - t
    say(f"{label}: {len(ps)} files, {n / 2**20:.0f} MB in {dt:.2f}s = {n / 2**20 / dt:.0f} MB/s, {dt / len(ps) * 1000:.1f} ms/file")


say("ultralytics-style: each of 5 files read alone, MB/s per file")
for p in paths[:5]:
    t = time.perf_counter()
    n = read(p)
    say(f"  {n / 1024:.0f} KB {n / 2**20 / (time.perf_counter() - t):.0f} MB/s")
timed("cold, 1 thread", paths[5:205])
timed("warm, 1 thread (same files)", paths[5:205])
timed("cold, 8 threads", paths[205:1205], 8)
timed("cold, 1 thread (another 200)", paths[1205:1405])
t = time.perf_counter()
os.makedirs("/kaggle/working/copy", exist_ok=True)
for p in paths[1405:1605]:
    shutil.copy(p, "/kaggle/working/copy")
say(f"copy 200 files to /kaggle/working: {time.perf_counter() - t:.2f}s")
say("mount", [l for l in open("/proc/mounts") if "/kaggle/input" in l][:2])
