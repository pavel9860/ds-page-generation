"""Bulk UV/3D-map dataset export. Two subcommands:

`from-images` (original): sourced from the pre-rendered flat text-page
matrices (lossless grayscale, matrices/*.npz -- not the lossy .jpg) --
reuses the text_render.py warp pipeline (camera + surface bend) with the
source matrix substituted in place of render_flat_text's output. 2D
print-stage creases/noise and stage-4 shading are skipped: creases/noise
only ever ran inside render_flat_text (bypassed here), and shade_enabled is
forced off below. Mirrors the input dataset's own train/test/val split (by
directory) rather than re-splitting. One .npz per sample:
  original : (1024,1024) uint8     -- the source flat page, unmodified
  warped   : (1024,1024,3) uint8   -- warped page (stage 4, no shading)
  uv       : (256,256,2) float16   -- [0,1] flat-texture coords, NaN off-page
  map3d    : (256,256,3) float16   -- (U,V,Z) mm surface coords, NaN off-page

Run (from repo root):
    .venv/bin/python src/tools/export_uv_from_images.py from-images \\
        /run/media/me/D/ML_DS/UVTM/TextPages/100k_1024_1024_v2.0.2 \\
        /run/media/me/D/ML_DS/UVTM/TextPages/uv_pages_100k_1mp_v.2.0.0

`overfit` (new): a small fixed-size set rendered from scratch -- 100% frame
fill, zero page margin, usual font-size range, split evenly across three
language groups (en / other-EU-Latin / cyr) drawn from a corpus of
train/*.txt books -- for overfitting sanity checks. Same .npz schema as
above, plus a flat|warped|dewarped visualization PNG per sample (dewarped =
the warped photo remapped back to flat layout via the same UV chain, i.e.
render_text_raw's reverse of stage 3 -- confirms the camera/surface/UV-warp
chain is self-consistent, distinct from the direct-UV flat texture itself).

Run (from repo root):
    .venv/bin/python src/tools/export_uv_from_images.py overfit \\
        --texts_dir /run/media/me/D/ML_DS/UVTM/Texts/train \\
        --out_dir /run/media/me/D/ML_DS/UVTM/Layouts/test \\
        --n 65
"""
import argparse
import glob
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from src.synth import config as cfg
from src.synth.text_render import generate_text

SPLITS = ("train", "val", "test")
_STATE = {}


def _init():
    cv2.setNumThreads(1)


def _tex_from_matrix(gray_u8: np.ndarray) -> np.ndarray:
    """Source lossless grayscale matrix -> the same INK/PAPER-scaled
    reflectance texture render_flat_text produces, so the warp pipeline
    sees a consistent value range regardless of texture source."""
    return (cfg.INK + (cfg.PAPER - cfg.INK) * (gray_u8.astype(np.float32) / 255.0)).astype(np.float32)


def _one(args):
    split, path, out_dir = args
    stem = path.stem
    original = np.load(path)["arr_0"]
    tex = _tex_from_matrix(original)

    seed = int(stem)
    rgb, _rgb_lowres, uv_map, _flat_page, _cam, _zf, map3d = generate_text(
        seed, corpus_paths=None, font_files=None, shade_enabled=False, flat_tex=tex)

    np.savez_compressed(
        out_dir / split / f"{stem}.npz",
        original=original,
        warped=rgb,
        uv=uv_map.astype(np.float16),
        map3d=map3d.astype(np.float16),
    )
    return split


def main(matrices_root, out_dir, workers):
    matrices_root = Path(matrices_root)
    out_dir = Path(out_dir)
    jobs = []
    for split in SPLITS:
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted((matrices_root / "matrices" / split).glob("*.npz")):
            jobs.append((split, p, out_dir))
    print(f"n_jobs={len(jobs)}", flush=True)

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        for _ in pool.map(_one, jobs, chunksize=4):
            done += 1
            if done % 500 == 0:
                el = time.time() - t0
                print(f"{done}/{len(jobs)} elapsed={el:.0f}s rate={done / el:.2f}/s "
                     f"eta={(len(jobs) - done) / (done / el) / 60:.1f}min", flush=True)
    print(f"done: n={len(jobs)} wall={time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# overfit: from-scratch small set, 100% fill / 0 margin, 3 language groups,
# npz export + flat|warped|dewarped visualization.
# ---------------------------------------------------------------------------

_STOP_EN = set("the and of to in is was that he she it with as for on at by "
              "an be this his her they were are not".split())
_CYR_RE = re.compile("[Ѐ-ӿ]")
LANG_GROUPS = ("en", "eu", "cyr")


def classify_language(path: str) -> str:
    """en / eu (other European, Latin script) / cyr, by script + English
    stopword density on a sample of the file (no external langdetect dep)."""
    txt = open(path, encoding="utf-8", errors="ignore").read(20000)
    letters = [c for c in txt if c.isalpha()]
    if not letters:
        return "eu"
    cyr_ratio = sum(1 for c in letters if _CYR_RE.match(c)) / len(letters)
    if cyr_ratio > 0.3:
        return "cyr"
    words = re.findall(r"[a-zA-Z]+", txt.lower())
    score = sum(1 for w in words if w in _STOP_EN) / max(1, len(words))
    return "en" if score > 0.15 else "eu"


def build_corpus_groups(texts_dir: str) -> dict:
    paths = sorted(glob.glob(os.path.join(texts_dir, "*.txt")))
    if not paths:
        raise RuntimeError(f"no .txt files under {texts_dir}")
    groups = {g: [] for g in LANG_GROUPS}
    for p in paths:
        groups[classify_language(p)].append(p)
    missing = [g for g in LANG_GROUPS if not groups[g]]
    if missing:
        raise RuntimeError(f"no corpus files classified into group(s) {missing} under {texts_dir}")
    return groups


def _counts_for(n: int) -> dict:
    base = n // len(LANG_GROUPS)
    counts = {g: base for g in LANG_GROUPS}
    for g in LANG_GROUPS[: n - base * len(LANG_GROUPS)]:
        counts[g] += 1
    return counts


def _make_triptych(flat_u8: np.ndarray, warped_rgb: np.ndarray, dewarped_rgb: np.ndarray) -> np.ndarray:
    """flat | warped | dewarped, side by side, all resized to a common
    square height so panel widths match regardless of source resolution."""
    h = max(flat_u8.shape[0], warped_rgb.shape[0], dewarped_rgb.shape[0])
    flat_bgr = cv2.cvtColor(flat_u8, cv2.COLOR_GRAY2BGR)
    panels = [cv2.resize(p, (h, h), interpolation=cv2.INTER_AREA) for p in
             (flat_bgr, warped_rgb, dewarped_rgb)]
    return np.concatenate(panels, axis=1)


def _one_overfit(job):
    idx, group, seed, corpus_paths, font_files, font_pt_range, out_dir, vis_dir = job
    from src.synth import config as cfg
    from src.synth.render import emulate_photo
    from src.synth.text_render import PAGE_MM, PAGE_PX, render_text_raw, rectify_backward
    from src.synth.text_texture import render_flat_text, sample_snippet

    rng = np.random.default_rng(seed)
    snippet = sample_snippet(corpus_paths, rng)
    tex = render_flat_text(snippet, rng, PAGE_PX, PAGE_MM, font_files, font_pt_range=font_pt_range)

    img, depth, page, cam, zf, uv_map, tex, map3d = render_text_raw(
        seed, corpus_paths=None, font_files=font_files, flat_tex=tex)

    rng2 = np.random.default_rng(seed * cfg.EXPORT_SEED_MULT + 3)
    warped, _ = emulate_photo(img, depth, page, cam, cfg.TEXT_CANVAS, rng2, bg_value=cfg.TEXT_BG_GRAY,
                              blur_scale=cfg.TEXT_BLUR_SCALE, bad_area_enabled=False,
                              border_jitter_enabled=False, shade_enabled=True)
    flat_page = np.clip(tex * 255, 0, 255).astype(np.uint8)
    dewarped = rectify_backward(zf, cam, warped, cfg.TEXT_CANVAS, page_px=PAGE_PX,
                                bg_value=cfg.TEXT_BG_GRAY)

    stem = f"{idx:03d}_{group}_{seed}"
    np.savez_compressed(
        out_dir / f"{stem}.npz",
        original=flat_page,
        warped=warped,
        uv=uv_map.astype(np.float16),
        map3d=map3d.astype(np.float16),
    )
    cv2.imwrite(str(vis_dir / f"{stem}.png"), _make_triptych(flat_page, warped, dewarped))
    return stem


def main_overfit(texts_dir, out_root, n, seed0, workers, font_pt_min, font_pt_max):
    out_dir = Path(out_root) / f"overfitting_set_{n}"
    vis_dir = Path(out_root) / f"overfit_{n}_vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    groups = build_corpus_groups(texts_dir)
    from src.synth.text_texture import find_fonts
    font_files = find_fonts()
    font_pt_range = cfg.TEXT_FONT_PT_RANGE if font_pt_min is None else (font_pt_min, font_pt_max)

    orig_margin, orig_fill = cfg.TEXT_MARGIN_MM, cfg.TEXT_FILL_FRAC_RANGE
    cfg.TEXT_MARGIN_MM = 0.0            # 100% fill, no margins
    cfg.TEXT_FILL_FRAC_RANGE = (1.0, 1.0)
    try:
        jobs = []
        idx = 0
        for group, count in _counts_for(n).items():
            for _ in range(count):
                jobs.append((idx, group, seed0 + idx, groups[group], font_files,
                            font_pt_range, out_dir, vis_dir))
                idx += 1

        t0 = time.time()
        done = 0
        with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
            for stem in pool.map(_one_overfit, jobs):
                done += 1
                print(f"{done}/{len(jobs)} {stem}", flush=True)
        print(f"done: n={len(jobs)} wall={time.time() - t0:.0f}s -> {out_dir}, {vis_dir}", flush=True)
    finally:
        cfg.TEXT_MARGIN_MM, cfg.TEXT_FILL_FRAC_RANGE = orig_margin, orig_fill


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_img = sub.add_parser("from-images")
    p_img.add_argument("matrices_root", help="dataset root with matrices/{train,val,test}/*.npz")
    p_img.add_argument("out_dir")
    p_img.add_argument("--workers", type=int, default=6)

    p_of = sub.add_parser("overfit")
    p_of.add_argument("--texts_dir", default="/run/media/me/D/ML_DS/UVTM/Texts/train")
    p_of.add_argument("--out_dir", default="/run/media/me/D/ML_DS/UVTM/Layouts/test")
    p_of.add_argument("--n", type=int, default=65)
    p_of.add_argument("--seed0", type=int, default=0)
    p_of.add_argument("--workers", type=int, default=6)
    p_of.add_argument("--font_pt_min", type=float, default=None)
    p_of.add_argument("--font_pt_max", type=float, default=None)

    args = ap.parse_args()
    if args.cmd == "from-images":
        main(args.matrices_root, args.out_dir, args.workers)
    else:
        main_overfit(args.texts_dir, args.out_dir, args.n, args.seed0, args.workers,
                     args.font_pt_min, args.font_pt_max)
