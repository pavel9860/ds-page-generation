"""Builds the Kaggle dataset (hwsynth source, the dspages.hand package, plan) as one small archive (large uploads silently vanish) and one
CPU kernel folder per shard of the plan; fonts are fetched by the kernel from the hwsynth font manifest.

python tools/kaggle_handpages/package.py --plan plan.jsonl --out DIR [--shards 4] [--name handpages] [--user USER]
kaggle datasets create -p DIR/dataset ; kaggle kernels push -p DIR/kernel<k>
"""
import argparse
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HWSYNTH = ROOT.parent / "handwriting" / "src" / "hwsynth"
IGNORE = shutil.ignore_patterns("__pycache__", "*.md")


def build(plan, out, shards=4, name="handpages", user="pavel9860"):
    out = Path(out)
    shutil.rmtree(out, ignore_errors=True)
    src = out / "src"
    shutil.copytree(ROOT / "src" / "dspages" / "hand", src / "code" / "dspages" / "hand", ignore=IGNORE)
    (src / "code" / "dspages" / "__init__.py").write_text("")
    shutil.copytree(HWSYNTH, src / "hwsynth_src" / "hwsynth", ignore=IGNORE)
    shutil.copy(plan, src / "plan.jsonl")
    (out / "dataset").mkdir()
    with tarfile.open(out / "dataset" / "handpages_src.tar.gz", "w:gz") as t:
        t.add(src, arcname="handpages_src")
    shutil.rmtree(src)
    (out / "dataset" / "dataset-metadata.json").write_text(json.dumps(
        {"title": f"{name}-code", "id": f"{user}/{name}-code", "licenses": [{"name": "CC-BY-NC-SA-4.0"}]}))
    template = (Path(__file__).parent / "kernel.py").read_text()
    for k in range(shards):
        d = out / f"kernel{k}"
        d.mkdir()
        (d / "kernel.py").write_text(f"SHARD, SHARDS = {k}, {shards}\n" + template)
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{name}-gen-{k}", "title": f"{name}-gen-{k}", "code_file": "kernel.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True, "dataset_sources": [f"{user}/{name}-code"],
            "kernel_sources": [], "competition_sources": [], "model_sources": []}, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--name", default="handpages")
    ap.add_argument("--user", default="pavel9860")
    a = ap.parse_args()
    print("ok", build(a.plan, a.out, a.shards, a.name, a.user))


if __name__ == "__main__":
    main()
