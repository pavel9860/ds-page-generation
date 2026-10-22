"""Kaggle CPU kernel: one shard of the plan. SHARD and SHARDS are set by package.py above this line."""
import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request

SHARD, SHARDS = globals().get("SHARD", 0), globals().get("SHARDS", 1)       # set by package.py above this line
WORK = "/kaggle/working"
FONTS = os.path.expanduser("~/.cache/hwsynth/fonts")

archive = glob.glob("/kaggle/input/**/handpages_src.tar.gz", recursive=True)
if archive:
    tarfile.open(archive[0]).extractall(f"{WORK}/_src")
    SRC = f"{WORK}/_src/handpages_src"
else:                                                     # Kaggle unpacked the archive
    SRC = os.path.dirname(glob.glob("/kaggle/input/**/plan.jsonl", recursive=True)[0])
print("src", os.listdir(SRC), flush=True)

subprocess.run("apt-get install -y -q libraqm0 libfribidi0 > /dev/null 2>&1", shell=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "regex", "fonttools", "pillow", "scikit-image", "pymupdf"], check=True)
os.makedirs(FONTS, exist_ok=True)
data = f"{SRC}/hwsynth_src/hwsynth/data"
for e in json.load(open(f"{data}/fonts.json"))["fonts"]:
    dst = os.path.join(FONTS, e["file"])
    if "local" in e:
        shutil.copy(f"{data}/{e['local']}", dst)
    elif not os.path.exists(dst):
        for _ in range(3):
            try:
                open(dst, "wb").write(urllib.request.urlopen(e["url"].replace("[", "%5B").replace("]", "%5D"), timeout=60).read())
                break
            except Exception as ex:
                print("font retry", e["file"], ex, flush=True)
for f in glob.glob(f"{data}/fonts_local/*.ttf"):
    shutil.copy(f, FONTS)
print(len(os.listdir(FONTS)), "fonts", flush=True)

env = dict(os.environ, PYTHONPATH=f"{SRC}/hwsynth_src:{SRC}/code", OMP_NUM_THREADS="1")
subprocess.run([sys.executable, "-c", "from PIL import features; assert features.check('raqm'), 'no raqm'"], check=True, env=env)
subprocess.run([sys.executable, "-u", "-m", "dspages.hand", "run", "--plan", f"{SRC}/plan.jsonl", "--out", WORK, "--shard", str(SHARD),
                "--shards", str(SHARDS), "--workers", str(os.cpu_count())], check=True, env=env)
