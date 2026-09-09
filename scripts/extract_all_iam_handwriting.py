"""Batch-extracts the handwriting region from every IAM Handwriting form scan and saves it
directly under the dataset root (filenames are unique across subfolders, confirmed before
writing), leaving the original per-form subfolders untouched.
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from extract_iam_handwriting import fill_missing_lines, find_horizontal_lines

ROOT = "/run/media/me/D/ML_DS/UVTM/Layouts/IAM Handwriting forms"


def main() -> None:
    files = sorted(glob.glob(f"{ROOT}/*/*.png"))
    print(f"{len(files)} source forms")

    n_ok = n_filled = n_failed = 0
    for i, f in enumerate(files, 1):
        try:
            gray = np.array(Image.open(f).convert("L"))
            detected = find_horizontal_lines(gray)
            lines, is_detected = fill_missing_lines(detected)
            y0, y1 = lines[1], lines[2]
            color_img = Image.open(f).convert("RGB")
            crop = color_img.crop((0, y0, color_img.width, y1))
            out_path = os.path.join(ROOT, os.path.basename(f))
            crop.save(out_path)
            n_ok += 1
            if not (is_detected[1] and is_detected[2]):
                n_filled += 1
        except Exception as e:
            n_failed += 1
            print(f"  [{i}/{len(files)}] FAILED {os.path.basename(f)}: {e}", file=sys.stderr)
        if i % 200 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] ok={n_ok} filled={n_filled} failed={n_failed}")

    print(f"\ndone. {n_ok} crops written to {ROOT}, {n_filled} used a median-filled line "
          f"for the 2nd/3rd rule, {n_failed} failed.")


if __name__ == "__main__":
    main()
