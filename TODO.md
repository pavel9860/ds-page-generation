# Tasks (delete a task when it is done)

## Triplets (tools/triplets.py, manifest m3 = the first 600-page manifest with its garbled pages replaced)
- 046, 066, 080: check by position that the frames are filled where the stream lasts and that nothing is missing; they are single-page sources, so the stream ends on the first page.
- 047: find what is lost at 11 pt (formatting? by position metric).
- 075: gaps in the text columns (frames filled?).
- Overlaps left: widened single-line blocks over a neighbour (cases 8, 9, 38, 40, 55): limit the widening by every block of the line band.
- Rebuild the manifest and the triplets after the script groups and the control-code check (cases 8, 123, 147).

## Code
- Delete the old placement code in layout/refont.py (place, row_groups, follow_top, Mapper use, pour, column_gap, flow, layout, their config numbers) and move what the template uses into one place.
- Tests: cover layout/template.py (frames, fixed boxes, obstacles, rows, pictures); replace the tests of the old placement.
- Manifest: the listing has to be rebuilt once for the new language groups (python -m dspages.prep.files).

## Pipeline (do not touch geometry solve_strip / solve_sag)
- Refactor, simplify, consolidate the pipeline (generate, render, paper effects).
- Speed: noise (fbm in paper and render), cold start; measure first.
- 200 UV samples (flat, warped, dewarped, UV map, 3D map) with the new layout and the noise scale: python -m dspages all, then tools/gallery.py.
