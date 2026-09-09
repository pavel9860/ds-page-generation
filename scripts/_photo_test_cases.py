import numpy as np
from PIL import Image

from build_textures import _margin_limited_box
from text_bounds import classify_patches_scan, detect_content_box_scan

L = "/run/media/me/D/ML_DS/UVTM/Layouts"

# expect_photo=True means this page IS photo-dominant and SHOULD be filtered (high photo_frac)
# expect_photo=False means this page is text/table/form and should NOT be filtered
CASES = [
    (f"{L}/DocLayNet-v1.2/financial_reports/NYSE_CHK_2010_p14_ad95fab7.png", True),
    (f"{L}/commonforms_val_subset/2929470-0.png", True),
    (f"{L}/DocLayNet-v1.2/laws_and_regulations/amt_general_handbook_p35_e02fe387.png", True),
    (f"{L}/commonforms_val_subset/4303406-6.png", False),
    (f"{L}/commonforms_val_subset/2731541-109.png", False),
    (f"{L}/commonforms_val_subset/4691088-34.png", False),
    (f"{L}/DocLayNet-v1.2/financial_reports/NYSE_HRL_2004_p32_698480a4.png", False),
    (f"{L}/commonforms_val_subset/0522016-1.png", False),
    (f"{L}/DocLayNet-v1.2/financial_reports/ASX_KCN_2013_p60_f989c6f6.png", False),
    (f"{L}/commonforms_val_subset/1297561-2.png", False),
    (f"{L}/DocLayNet-v1.2/financial_reports/NYSE_SMFG_2011_p179_10ecf012.png", False),
    (f"{L}/commonforms_val_subset/0355988-1.png", False),
    (f"{L}/commonforms_val_subset/1222377-0.png", False),
    (f"{L}/commonforms_val_subset/2580439-1.png", False),
    (f"{L}/commonforms_val_subset/0729491-59.png", False),
    (f"{L}/commonforms_val_subset/0390831-149.png", False),
    (f"{L}/commonforms_val_subset/1634751-0.png", False),
]

n_correct = 0
for path, expect_photo in CASES:
    img = Image.open(path).convert("RGB")
    bgr = np.array(img)[:, :, ::-1].copy()
    h, w = bgr.shape[:2]
    box = detect_content_box_scan(bgr)
    mbox = _margin_limited_box(box, w, h)
    n_empty, n_photo, n_text = classify_patches_scan(bgr, mbox)
    n_occ = n_photo + n_text
    photo_frac = n_photo / n_occ if n_occ else 0.0
    got_photo = photo_frac >= 0.6
    ok = got_photo == expect_photo
    n_correct += ok
    print(f"{'OK ' if ok else 'FAIL'} photo_frac={photo_frac:.3f} expect_photo={expect_photo} {path.split('/')[-1]}")

print(f"\n{n_correct}/{len(CASES)} correct")
