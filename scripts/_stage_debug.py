import random
import glob

import numpy as np
from PIL import Image, ImageDraw

from build_textures import render_source_raster, image_crop_box, random_margins, IMAGE_SIZE
from text_bounds import detect_content_box_scan

sources = []
sources += glob.glob('/run/media/me/D/ML_DS/UVTM/Layouts/scientific_paper/arxiv_pdfs/*.pdf')[:3]
sources += glob.glob('/run/media/me/D/ML_DS/UVTM/Layouts/DocLayNet-v1.2/*/*.png')[:6]
sources += glob.glob('/run/media/me/D/ML_DS/UVTM/Layouts/commonforms_val_subset/*.png')[:3]
random.Random(5).shuffle(sources)


def panel(img, w=280):
    scale = w / img.width
    return img.resize((w, round(img.height * scale)))


n = 0
for f in sources:
    if n >= 6:
        break
    fmt = 'pdf' if f.endswith('.pdf') else 'image'
    rel = f.split('Layouts/')[1]
    row = {'format': fmt, 'file': rel, 'page': 0}
    rng = random.Random(200 + n)

    img = render_source_raster(row)
    W, H = img.size
    bgr = np.array(img)[:, :, ::-1].copy()
    bx0, by0, bx1, by1 = detect_content_box_scan(bgr)

    p1 = img.copy()

    p2 = img.copy()
    d2 = ImageDraw.Draw(p2)
    d2.rectangle([bx0, by0, bx1, by1], outline=(255, 0, 0), width=6)

    m_left, m_right, m_top, m_bottom = random_margins(rng)
    side, mx, sx, lx, my, sy, ly = image_crop_box(bx0, by0, bx1, by1, m_left, m_right, m_top, m_bottom, rng)
    side_i = round(side)

    pad = side_i
    canvas_dbg = Image.new('RGB', (max(W, side_i) + pad, max(H, side_i) + pad), (255, 255, 255))
    off = pad // 2
    canvas_dbg.paste(img, (off, off))
    d3 = ImageDraw.Draw(canvas_dbg)
    box_x0 = off + (sx - mx)
    box_y0 = off + (sy - my)
    d3.rectangle([box_x0, box_y0, box_x0 + side_i, box_y0 + side_i], outline=(0, 0, 255), width=6)
    p3 = canvas_dbg

    canvas = Image.new('RGB', (side_i, side_i), (255, 255, 255))
    sx0, sy0, lx_i, ly_i = round(sx), round(sy), round(lx), round(ly)
    ix0, iy0 = max(sx0, 0), max(sy0, 0)
    ix1, iy1 = min(sx0 + lx_i, W), min(sy0 + ly_i, H)
    if ix1 > ix0 and iy1 > iy0:
        canvas.paste(img.crop((ix0, iy0, ix1, iy1)), (round(mx) + ix0 - sx0, round(my) + iy0 - sy0))
    p4 = canvas

    p5 = canvas.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

    panels = [panel(p1), panel(p2), panel(p3), panel(p4), panel(p5)]
    h = max(p.height for p in panels)
    strip = Image.new('RGB', (sum(p.width for p in panels) + 4 * 10, h), (240, 240, 240))
    x = 0
    for p in panels:
        strip.paste(p, (x, (h - p.height) // 2))
        x += p.width + 10
    strip.save(f'/tmp/stages_{n:02d}.jpg', quality=90)
    n += 1
print('done', n)
