"""Pages as the 768 x 1024 JPEG shards of the corpus (tools/form_pages.py: no padding, q90, 4:2:0): scan pictures decoded at reduced scale, other pages
rendered by MuPDF, PIL / pdftoppm as fallbacks, a crash of the pool splits the jobs until the file that kills MuPDF is found."""

import io
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from PIL import Image

from .paths import CC_ROOT

W, H, QUALITY = 768, 1024, 90
WORKERS, DOC_SECONDS = 6, 120
pymupdf.TOOLS.mupdf_display_errors(False)
cv2.setNumThreads(1)


class Timeout(BaseException):
    pass


def _alarm(*_):
    raise Timeout()


def encode(rgb):
    if rgb.shape[:2] != (H, W):
        rgb = cv2.resize(
            rgb, (W, H), interpolation=cv2.INTER_AREA if rgb.shape[0] * rgb.shape[1] > W * H else cv2.INTER_CUBIC
        )
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=QUALITY)
    return buf.getvalue()


def fallback(path, d, p):
    try:
        if path.lower().endswith(".pdf"):
            with tempfile.TemporaryDirectory() as tmp:
                subprocess.run(
                    [
                        "pdftoppm",
                        "-jpeg",
                        "-f",
                        str(p + 1),
                        "-l",
                        str(p + 1),
                        "-scale-to-x",
                        str(W),
                        "-scale-to-y",
                        str(H),
                        path,
                        f"{tmp}/o",
                    ],
                    check=True,
                    timeout=120,
                    capture_output=True,
                )
                rgb = np.asarray(Image.open(sorted(Path(tmp).glob("o*.jpg"))[0]).convert("RGB"))
        elif p == 0:
            rgb = np.asarray(Image.open(path).convert("RGB"))
        else:
            return None
        return d, p, encode(rgb)
    except Exception:
        return None


def image_job(path, d):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    return d, 0, encode(rgb)


def render_job(args):
    """(path, [(doc, page)]) -> [(doc, page, jpeg)]. An image file is page 0 of its doc."""
    path, items = args
    out = []
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(DOC_SECONDS * 3)
    try:
        if not path.lower().endswith(".pdf"):
            out.append(image_job(path, items[0][0]))
        else:
            sys.path.insert(0, str(CC_ROOT))
            from pagetype import thumbs as T

            with pymupdf.open(path) as doc:
                for d, p in items:
                    try:
                        pg = doc[p]
                        rgb = T.scan_image(doc, pg, H)
                        if rgb is None:
                            pm = pg.get_pixmap(
                                matrix=pymupdf.Matrix(W / pg.rect.width, H / pg.rect.height),
                                colorspace=pymupdf.csRGB,
                                alpha=False,
                            )
                            rgb = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, 3)
                        out.append((d, p, encode(rgb)))
                    except Timeout:
                        raise
                    except Exception:
                        pass
    except BaseException:
        pass
    finally:
        signal.alarm(0)
    done = {(d, p) for d, p, _ in out}
    out += [r for d, p in items if (d, p) not in done and (r := fallback(path, d, p))]
    return out


def run_jobs(jobs, workers=WORKERS):
    try:
        with ProcessPoolExecutor(workers if len(jobs) > 1 else 1) as ex:
            return [x for r in ex.map(render_job, jobs, chunksize=8) for x in r]
    except BrokenProcessPool:
        if len(jobs) == 1:
            print("skipped (crash):", jobs[0][0], flush=True)
            return []
        h = len(jobs) // 2
        return run_jobs(jobs[:h], workers) + run_jobs(jobs[h:], workers)
