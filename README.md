# ds-page-generation

Synthetic photos of document pages with exact ground truth (UV map, 3D map, flat page) for training
page-dewarping networks. Three independent parts:

1. **Layout** — a flat sheet from a source manifest (digital-born PDF pages restyled in other fonts and sizes, rasterized PDF pages,
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
python -m dspages all --preset full --out OUT --n 1000 --workers 12 [--preview]
```

`plan` writes `OUT/plan.jsonl` (stratified conditions). `all` writes the samples only: the layout and geometry of each
are made in memory. `layout` and `geometry` save their parts on request; `render` uses saved parts of the same index when
present. `--preview` writes `.jpg` previews to `OUT_preview/` (off by default). Per sample and part:

| file | arrays |
|---|---|
| `samples/<i>.npz` | `flat` sheet, `warped` photo (luminance uint8, h × w), `grid2d` (2, Hg, Wg) photo x, y of each page grid node in [-1, 1], align_corners=True (float32, the grid_sample grid; no NaNs), `grid3d` (3, Hg, Wg) X, Y, Z [mm] of each node on the surface (float32), `mask` photo pixels on the page (bool) |
| `layouts/<i>.npz` | `page` sheet luminance uint8, `gray` clean sheet uint8 (`layout` only) |
| `geometry/<i>.npz` | `X, Y, Z` page grid [mm] float32, rows down the sheet, table at z = 0 (`geometry` only) |

Sample metadata (source page, scene, paper, camera, light, effects) goes to `OUT/meta.jsonl`; the part files hold theirs
as `meta`. A full-preset sample is about 2.9 MB (70k samples: about 200 GB).

## Presets

| | demo | full |
|---|---|---|
| sheet | 150 × 150 mm | A4 |
| flat / warped | 1024 × 1024 | 1024 × 1440 |
| UV / 3D maps (page grid) | 256 × 256 | 256 × 360 |
| margins | 0 | 0–10 % |
| layout effects, photo effects | off | on |
| background | black | table texture |

Both draw paper types of 50, 60, 80, 100 and 120 g/m².

## Conditions

`dspages/conditions.py` lists every factor of a preset (scene combination, two profiles, orientation,
bend direction, paper, creases, camera distance and tilt, light, margins) with its probability. A plan gives each
factor value its exact share and pairs factors by independent shuffles; `cells(P, group)` enumerates the
combinations of a group.

## Check

```bash
pytest                                  # geometry, layout, render, conditions (~20 s)
python tools/analyze.py OUT             # distributions and limits -> OUT/report/
python tools/gallery.py OUT             # contact sheet -> OUT/report/gallery.jpg
```

Data preparation: `python -m dspages.prep.files` (the listing of the source folders, `LayoutCfg.sources`), then
`python -m dspages.prep.manifest --out <manifest>` (one entry per page). PDFs of `LayoutCfg.raster_only` folders are only
rasterized, never restyled.

## License

[CC BY-NC 4.0](LICENSE)
