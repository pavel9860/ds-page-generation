import glob
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_textures import MAX_PHOTO_PATCH_FRACTION, MIN_PATCH_FILL_SCAN, _margin_limited_box, predict_fill_ok
from text_bounds import classify_patches_scan, detect_content_box_scan

L = "/run/media/me/D/ML_DS/UVTM/Layouts"
N = 200

mark_font = ImageFont.load_default(size=36)


def image_scores(path: str) -> tuple[float, float]:
    img = Image.open(path).convert("RGB")
    bgr = np.array(img)[:, :, ::-1].copy()
    h, w = bgr.shape[:2]
    content_box = detect_content_box_scan(bgr)
    measure_box = _margin_limited_box(content_box, w, h)
    n_empty, n_photo, n_text = classify_patches_scan(bgr, measure_box)
    n_occupied = n_photo + n_text
    fill = n_occupied / (n_empty + n_occupied)
    photo_frac = n_photo / n_occupied if n_occupied else 0.0
    return fill, photo_frac


def run(sources: list[str], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    random.Random(11).shuffle(sources)

    n_done = n_kept = n_filtered = 0
    for f in sources:
        if n_done >= N:
            break
        try:
            fill, photo_frac = image_scores(f)
            kept = predict_fill_ok("image", f, 0)
        except Exception:
            continue

        rel = os.path.relpath(f, L)
        row = {"format": "image", "file": rel, "page": 0}
        try:
            from build_textures import render_source_raster
            img = render_source_raster(row)
        except Exception:
            continue

        tag = "KEPT" if kept else "FILTERED"
        color = (0, 160, 0) if kept else (200, 0, 0)

        thumb = img.copy()
        thumb.thumbnail((700, 900))
        draw = ImageDraw.Draw(thumb)
        label = f"{tag}  fill={fill:.3f} photo={photo_frac:.3f}"
        draw.rectangle([0, 0, thumb.width, 46], fill=color)
        draw.text((8, 6), label, font=mark_font, fill=(255, 255, 255))

        out_name = f"{n_done:03d}_{tag.lower()}_fill{fill:.3f}_photo{photo_frac:.3f}.jpg"
        thumb.convert("RGB").save(os.path.join(out_dir, out_name), quality=88)

        n_kept += kept
        n_filtered += not kept
        n_done += 1

    print(f"[image] done {n_done} pages: kept={n_kept} filtered={n_filtered} -> {out_dir}")
    print(f"  thresholds: MIN_PATCH_FILL_SCAN={MIN_PATCH_FILL_SCAN} MAX_PHOTO_PATCH_FRACTION={MAX_PHOTO_PATCH_FRACTION}")


image_sources = []
image_sources += glob.glob(f"{L}/commonforms_val_subset/*.png")
image_sources += glob.glob(f"{L}/DocLayNet-v1.2/*/*.png")

run(image_sources, os.path.join(L, "test", "fill_filter_check_image"))
