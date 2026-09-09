"""Extracts every page image from DocLayNet-v1.2's parquet shards, applies a vertical-only
stretch to the A4 portrait ratio (width unchanged, height = width * sqrt(2)), and saves as PNG
under a directory named by the page's doc_category.
"""
import glob
import io
import os

import pyarrow.parquet as pq
from PIL import Image

SRC = "/run/media/me/D/ML_DS/UVTM/Layouts/DocLayNet-v1.2"
OUT = "/run/media/me/D/ML_DS/UVTM/Layouts/test/DocLayNet_pages_a4"
A4_RATIO = 1754 / 1240   # height/width for A4 portrait, vertical-only stretch


def main() -> None:
    shards = sorted(glob.glob(f"{SRC}/*.parquet"))
    print(f"{len(shards)} shards")

    n_total = 0
    for si, shard in enumerate(shards, 1):
        table = pq.read_table(shard, columns=["image", "metadata"])
        images = table.column("image").to_pylist()
        metas = table.column("metadata").to_pylist()
        for img_rec, meta in zip(images, metas):
            cat = meta["doc_category"]
            out_dir = os.path.join(OUT, cat)
            os.makedirs(out_dir, exist_ok=True)

            img = Image.open(io.BytesIO(img_rec["bytes"])).convert("RGB")
            w, h = img.size
            new_h = round(w * A4_RATIO)
            stretched = img.resize((w, new_h), Image.LANCZOS)

            orig_stem = os.path.splitext(meta["original_filename"])[0]
            name = f"{orig_stem}_p{meta['page_no']}_{meta['page_hash'][:8]}.png"
            stretched.save(os.path.join(out_dir, name))
            n_total += 1
        print(f"  [{si}/{len(shards)}] {shard.split('/')[-1]}: {len(images)} pages "
              f"(running total {n_total})")

    print(f"\ndone. {n_total} pages written under {OUT}/<doc_category>/")


if __name__ == "__main__":
    main()
