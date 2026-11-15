"""One handwritten page from pagegen alone (hwsynth + the numbers of GenCfg). python samples/generate_page.py OUT.png [SEED] [LAYOUT]"""

import random
import sys

import cv2

from dspages.pagegen.config import GenCfg
from dspages.pagegen.layouts import LAYOUTS
from dspages.pagegen.page import handwritten_page

POOL = {
    "heading": ["Meeting notes", "Next steps", "Open questions"],
    "para": [
        "The committee agreed to move the review to the second week of March and to send the draft to all members beforehand.",
        "Budget figures are still missing for the northern office; they will be added once the accounts are closed.",
        "Several members asked for a shorter agenda and for the minutes to be circulated within two days.",
    ]
    * 3,
    "item": ["send the draft", "collect the figures", "book the room", "call the printer"] * 3,
}

if __name__ == "__main__":
    out, seed = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
    layout = sys.argv[3] if len(sys.argv) > 3 else "plain"
    assert layout in LAYOUTS, sorted(LAYOUTS)
    page = handwritten_page(layout, POOL, GenCfg(), random.Random(seed))
    if page is None:
        sys.exit("the page was rejected, try another seed")
    cv2.imwrite(out, cv2.cvtColor(page.img, cv2.COLOR_RGB2BGR))
    print(out, page.img.shape, page.words, "words")
