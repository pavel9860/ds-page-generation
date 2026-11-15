"""N real handwritten pages (dspages.multilang.hwpick) with boxes around their text lines (dspages.multilang.hwlines). python scripts/forms/hw_pages.py OUT_DIR [N]"""
import sys
from collections import Counter

from dspages.multilang import hwlines, hwpick

if __name__ == "__main__":
    out, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 800
    meta = hwpick.select(out, n)
    c = Counter(v["lang"] for v in meta.values())
    print(len(meta), "pages,", len(c), "languages; per language", min(c.values()), "to", max(c.values()))
    print(sum(hwlines.label_dir(out).values()), "boxes")
