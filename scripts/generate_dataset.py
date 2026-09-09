import csv
import os
import random
import sys
import time
import collections

sys.path.insert(0, os.path.dirname(__file__))
from build_textures import build_texture

MANIFEST = "/run/media/me/D/ML_DS/UVTM/Layouts/test/text_pdf_manifest.csv"
OUT_ROOT = "/run/media/me/D/ML_DS/UVTM/TextPages/100k_1024_1024_v2.0.0"
SPLIT_SEED = 1234
SPLITS = [("train", 0.96), ("val", 0.02), ("test", 0.02)]


def assign_splits(rows: list[dict]) -> None:
    groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in rows:
        groups[(r["category"], r["format"], r["lang"])].append(r)

    rng = random.Random(SPLIT_SEED)
    for key, group in groups.items():
        rng.shuffle(group)
        n = len(group)
        i = 0
        for split_name, frac in SPLITS[:-1]:
            take = round(n * frac)
            for r in group[i:i + take]:
                r["split"] = split_name
            i += take
        for r in group[i:]:
            r["split"] = SPLITS[-1][0]


def main() -> None:
    with open(MANIFEST, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assign_splits(rows)

    for split_name, _ in SPLITS:
        os.makedirs(os.path.join(OUT_ROOT, split_name), exist_ok=True)

    counts = collections.Counter(r["split"] for r in rows)
    print(f"total rows: {len(rows)}; split sizes: {dict(counts)}")

    t0 = time.time()
    ok = fail = 0
    for i, row in enumerate(rows):
        try:
            rng = random.Random(int(row["seed"]))
            img = build_texture(row, rng)
            out_path = os.path.join(OUT_ROOT, row["split"], f"{int(row['id']):06d}.jpg")
            img.convert("RGB").save(out_path, quality=92)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"  FAILED row {row['id']} ({row['file']}, page {row['page']}): {e}", file=sys.stderr)
        if (i + 1) % 2000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(rows) - i - 1) / rate
            print(f"  [{i+1}/{len(rows)}] {rate:.1f} img/s, ETA {eta/60:.1f} min, ok={ok} fail={fail}")

    print(f"done: {ok} ok, {fail} failed, {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()
