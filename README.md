# ds-page-generation

Synthetic photos of document pages with exact ground truth (UV map, 3D map, flat page) for training
page-dewarping networks. Three independent parts:

1. **Layout** — a flat sheet from a source manifest (book texts rendered with real fonts, rasterized PDF pages,
   scanned forms) with paper, print and ageing effects.
2. **Geometry** — the 3D page grid from physics: a paper strip on the table with line supports, a clamped edge
   and plastic folds (discrete elastica, SQP with contact), blended across the page between two end profiles,
   shell sag outside the support strip, shallow creases or a deep crease network.
3. **Render** — the layout on the page seen by a pinhole camera: rasterized with a z-buffer, lit by a point or
   area light with cast shadows, on a table, through camera effects (defocus, motion, vignetting,
   exposure, ISO noise, JPEG). All images are single-channel luminance.

## Install

```bash
pip install -e ".[dev]"
```

Paths to the corpora, the manifest, fonts and the output root are in `dspages/config.py` (`Paths`).

The layout finds pictures on source pages with YOLO11n trained on DocLayNet (ONNX). Its weights are AGPL-3.0 and not
part of this repository; fetch and export them once:

```bash
pip install -e ".[export]"
python -m dspages.prep.layout_model
```

## Run

```bash
python -m dspages all --preset full --out OUT --n 1000 --workers 12
```

`plan` writes `OUT/plan.jsonl` (stratified conditions); `layout`, `geometry` and `render` run each part alone;
`render` uses saved parts of the same index when present. Per sample and part:

| file | arrays |
|---|---|
| `layouts/<i>.npz` | `page` sheet luminance uint8, `gray` clean sheet uint8 |
| `geometry/<i>.npz` | `X, Y, Z` page grid [mm] float32, rows down the sheet, table at z = 0 |
| `samples/<i>.npz` | `flat` canvas, `warped` photo (luminance uint8), `uv` page coords in [0, 1] of each photo pixel (NaN off the page, float16), `map3d` X, Y, Z [mm] (float16), `mask` |

Each `.npz` holds `meta` (JSON: source, scene, paper, camera, light, effects) and has a `.jpg` preview.

## Presets

| | demo | full |
|---|---|---|
| sheet | 150 × 150 mm | A4 |
| flat / warped | 1024 × 1024 | 1024 × 1365 (A4 fitted, padded) |
| UV / 3D maps | 256 × 256 | 225 × 300 / 212 × 300 |
| margins | 0 | 0–10 % |
| layout effects, photo effects | off | on |
| background | black | table texture |

Both draw paper types of 50, 60, 80, 100 and 120 g/m².

## Conditions

`dspages/conditions.py` lists every factor of a preset (scene combination, near-flat, two profiles, orientation,
bend direction, paper, creases, camera distance and tilt, light, margins) with its probability. A plan gives each
factor value its exact share and pairs factors by independent shuffles; `cells(P, group)` enumerates the
combinations of a group.

## Check

```bash
pytest                                  # geometry, layout, render, conditions (~20 s)
python tools/analyze.py OUT             # distributions and limits -> OUT/report/
python tools/gallery.py OUT             # contact sheet -> OUT/report/gallery.jpg
```

Data preparation: `python -m dspages.prep.books` (book PDFs to text), `python -m dspages.prep.manifest` (one entry
per page), `python tools/inventory.py` (pages per script and language against the plan's quotas).

## License

[CC BY-NC 4.0](LICENSE)
