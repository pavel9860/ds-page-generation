# Handwriting vs print detector: handoff (2026-10-11)

Goal: CPU detector of handwriting vs print on pages (filter filled vs blank forms), all scripts.
Target: at most 2 FP and 2 FN of handwriting per page.

## Code
- `tools/hwdet.py`: `text` (text pool), `bank` (hwsynth lines), `para` (hwsynth paragraphs), `pages` (composed
  pages + YOLO labels, 0 handwriting / 1 print), `paras` (paragraph pages), `eval MODEL` (FP/FN per page on val
  by script group, hw boxes per page on real folders). Resumable, `--workers 2`.
- `tools/kaggle_hwdet/`: Kaggle kernel `pavel9860/hwdet-yolo26n` (2xT4, batch 64, imgsz 1024, cache off, training
  in a subprocess with a 50 min timeout, log in `train.log`). Init: `hwdet-init.pt` if in the dataset, else
  `yolo26n-doclaynet.pt`. `time=0.35` h (round 2 value).
- hwsynth installed editable into `.venv` (`pip install -e ../handwriting`).

## Data: /run/media/me/D/ML_DS/UVTM/hwdet
- `text.pkl` (88 languages + 153 scripts), `bank/` (15,915 items), `para/` (290 paragraphs).
- `ds_r2/`: round 2 dataset (8,300 train, 600 val). Uploaded as Kaggle dataset `pavel9860/hwdet-synth`.
- `ds/`: round 3 dataset, generated, NOT uploaded, NOT trained: 9,000 train + 263 paragraph pages (9000+), 600 val.
- `models/r1`, `models/r2`: Kaggle outputs (`hwdet-yolo26n.pt/.onnx`, `train.log`, runs).
- `pics/`: annotated real pages of r2; `pics/run1000/<folder>/` 250 pages each of filled_forms, handwritten,
  forms, books_pdf (filename starts with hw box count), counts in `<folder>.jsonl`.

## Results
- YOLOE-26n zero-shot ("handwriting" prompt): 0 boxes on 24 handwritten/filled pages.
- YOLO26n DocLayNet (hantian/yolo-doclaynet, ONNX `end2end=False`) runs in `layout/pictures.py` unchanged:
  421 vs 530 ms/page; same boxes on 62 of 96 pages. `PictureCfg` still points to YOLO11n.
- r1 (6,000 pages, 21 epochs, 42 min): real handwritten 40/100 pages with 0 hw boxes.
- r2 (8,300 pages, from r1, 8 epochs, 21 min): synthetic val hw mAP50 0.952; real handwritten 15/100 with 0 hw.
  run1000: pages with 0 hw boxes: filled_forms 190/250, handwritten 44/250, forms 231/250, books 223/250.
- r2 failures reported by the user: Arabic print, pictograms, parts of images, corners, lines detected as hw;
  hw in form fields (e.g. `000hw_156d1ed8964a_government_it`, `000hw_ee76716b1585_education_fr`,
  `000hw_5fd57691541e_education_fr`) and in dense handwriting (`000hw_handwritten_adnansyedwiki.com_66b6cf7ae0`)
  detected as print.

## Causes found in the r2 data (fixed in ds/, effect not measured)
1. hwsynth marks (underline, circle, cross, check) were labelled handwriting. Now dropped.
2. 138 of 153 rare scripts use fallback print fonts in hwsynth; they were 20% of hw keys. Now only scripts with a
   handwriting font (`coverage() == "hand"`).
3. Free-space mask (gray < 200) let hw go onto light pictures and tints. Now paper = value >= 225 and saturation <= 40.
4. hw never placed on form fields; print boxes spanned whole `Label : ......` lines. Now print boxes split at
   leader runs (>= 4 of `.…_·‥`) and wide gaps; 75% of pages fill 30-100% of leaders and long rules.
5. Scan look: added downscale-upscale (p 0.2) and bitonal (p 0.15) to `degrade`.

## Next steps
1. Upload `ds/` (copy `yolo26n-doclaynet.pt` into it, write `data.yaml`, no `hwdet-init.pt` to start fresh) as a
   new version of `pavel9860/hwdet-synth`; set `time` in `train.py`; push the kernel.
2. `python tools/hwdet.py eval <model>`, rerun the run1000 check, compare with r2 on the pages above.
3. No labelled real test set exists; FN on real pages is not measured.
4. Open question from the user: script/language recognition by appearance (print and hw, math, music).
   Proposed and not started: a separate line-crop classifier fed by the detector boxes.
5. Kaggle notes: `kaggle kernels status` gives no log while running; an earlier run with `cache="ram"` and 2 GPUs
   ran 80 min with no output and was cancelled.
