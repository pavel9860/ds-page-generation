"""Demo: run text-body bounds detection on 5 PDF pages and 5 scanned pages, save annotated
images to docs/text_bounds_examples/."""
import random
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np

from text_bounds import arxiv_sidebar_bbox, detect_content_box_scan, detect_text_box_pdf

LAYOUTS = Path("/run/media/me/D/ML_DS/UVTM/Layouts")
OUT = Path("/run/media/me/D/BUISNESS/pages_generation/docs/text_bounds_examples")
OUT.mkdir(parents=True, exist_ok=True)

RENDER_DPI = 150


def draw_box(img: np.ndarray, box, color=(0, 0, 255), thickness=4) -> np.ndarray:
    x0, y0, x1, y1 = [round(v) for v in box]
    out = img.copy()
    cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness)
    return out


def demo_pdfs(n=5):
    pdfs = sorted((LAYOUTS / "scientific_paper" / "arxiv_pdfs").glob("*.pdf"))
    random.Random(2).shuffle(pdfs)
    done = 0
    for pdf_path in pdfs:
        if done >= n:
            break
        try:
            doc = fitz.open(pdf_path)
            page = doc[0]
            box = detect_text_box_pdf(page)
            if box is None:
                doc.close()
                continue
            zoom = RENDER_DPI / 72
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if pix.n == 3 else cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            scaled_box = [c * zoom for c in box]
            annotated = draw_box(img, scaled_box)
            out_path = OUT / f"pdf_{done:02d}_{pdf_path.stem}.png"
            cv2.imwrite(str(out_path), annotated)
            print("wrote", out_path)
            doc.close()
            done += 1
        except Exception as e:
            print("skip", pdf_path.name, e)


def demo_scans(n=5):
    scans = sorted((LAYOUTS / "commonforms_val_subset").glob("*.png"))
    random.Random(1).shuffle(scans)
    done = 0
    for img_path in scans:
        if done >= n:
            break
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        box = detect_content_box_scan(img)
        annotated = draw_box(img, box)
        out_path = OUT / f"scan_{done:02d}_{img_path.stem}.png"
        cv2.imwrite(str(out_path), annotated)
        print("wrote", out_path)
        done += 1


if __name__ == "__main__":
    demo_pdfs()
    demo_scans()
