import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from build_textures import build_texture

manifest = sys.argv[1] if len(sys.argv) > 1 else "/run/media/me/D/ML_DS/UVTM/Layouts/test/text_pdf_manifest.csv"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "/run/media/me/D/ML_DS/UVTM/Layouts/test/textures_pdfimg_100"
n = int(sys.argv[3]) if len(sys.argv) > 3 else 100
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 11

with open(manifest, newline="", encoding="utf-8") as fh:
    rows = [r for r in csv.DictReader(fh) if r["format"] in ("pdf", "image")]

random.seed(seed)
sample = random.sample(rows, min(n, len(rows)))

os.makedirs(out_dir, exist_ok=True)
ok = fail = 0
for i, row in enumerate(sample):
    try:
        rng = random.Random(int(row["seed"]) if row.get("seed") else hash(row["id"]))
        img = build_texture(row, rng)
        out_name = f"{i:03d}_{row['category']}_{row['format']}.jpg"
        img.convert("RGB").save(os.path.join(out_dir, out_name), quality=92)
        ok += 1
    except Exception as e:
        fail += 1
        print(f"  FAILED row {row['id']} ({row['file']}, page {row['page']}): {e}", file=sys.stderr)

print(f"built {ok}/{len(sample)} textures to {out_dir} ({fail} failed)")
