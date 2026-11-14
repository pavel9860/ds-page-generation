"""Embeds shards pages_*.npz of Kaggle datasets with SigLIP 2 on both T4 GPUs in ONE script kernel (pagetype/embed_kaggle.py globs every shard of its
datasets) and downloads the output (embeddings.npy, ids.npy, shards.json) with the CLI (the API download once returned an empty file).

  python tools/kaggle_embed.py <kernel> <out dir> <work dir> <dataset>:<files> ...      e.g. pavel9860/emb-fix out work pavel9860/pages-768x1024-14:2"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

EMBED = "/run/media/p/D/BUISNESS/corpus_collecting/pagetype/embed_kaggle.py"


def run(kernel, datasets, work, out, wait_h=4):
    """datasets: [(ref, number of files)]. Waits until their files are listed, makes them public, pushes the kernel, polls every 5 minutes."""
    from dspages.kaggle_api import api
    k = api()
    t0 = time.time()
    while True:
        got = {}
        for ref, n in datasets:
            try:
                got[ref] = len(k.dataset_list_files(ref).files)
            except Exception:
                got[ref] = 0
        if all(got[r] >= n for r, n in datasets):
            break
        if time.time() - t0 > wait_h * 3600:
            raise SystemExit(f"datasets not on Kaggle after {wait_h} h: {got}")
        print(f"[{int(time.time() - t0)} s] datasets: {got}", flush=True)
        time.sleep(120)
    for ref, _ in datasets:                                        # CLI creates private datasets; the licence is CC BY-NC-SA 4.0
        d = Path(work) / "meta" / ref.split("/")[1]
        d.mkdir(parents=True, exist_ok=True)
        (d / "dataset-metadata.json").write_text(json.dumps({"title": ref.split("/")[1], "id": ref, "licenses": [{"name": "CC-BY-NC-SA-4.0"}],
                                                             "isPrivate": False}))
        k.dataset_metadata_update(ref, str(d))
    folder = Path(work) / "kernel"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    shutil.copy(EMBED, folder / "embed_kaggle.py")
    (folder / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel, "title": kernel.split("/")[1], "code_file": "embed_kaggle.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "keywords": ["gpu"],
        "dataset_sources": [r for r, _ in datasets], "kernel_sources": [], "competition_sources": [], "model_sources": [],
        "machine_shape": "NvidiaTeslaT4"}))
    k.kernels_push(str(folder))
    print("kernel pushed", flush=True)
    t1 = time.time()
    while True:
        st = str(k.kernels_status(kernel).status).lower()
        print(f"[{int((time.time() - t1) / 60)} min] {st}", flush=True)
        if "complete" in st or "error" in st or "cancel" in st:
            break
        time.sleep(300)
    out = Path(out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output", kernel, "-p", str(out)], check=False, timeout=3600)
    ok = (out / "embeddings.npy").exists() and (out / "embeddings.npy").stat().st_size > 1e5
    print("output downloaded to", out, "| embeddings.npy", "ok" if ok else "MISSING or empty", "| kernel", st, flush=True)
    if "complete" not in st or not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    kernel, out, work, *ds = sys.argv[1:]
    run(kernel, [(d.split(":")[0], int(d.split(":")[1])) for d in ds], work, out)
