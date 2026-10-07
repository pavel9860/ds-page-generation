# Tasks (delete a task when it is done)

## Triplets (tools/triplets.py, manifest m3 = the first 600-page manifest with its garbled pages replaced)
- Overlaps left: widened single-line blocks over a neighbour (cases 8, 9, 38, 40, 55): limit the widening by every block of the line band.

- Spans of some PDFs (046: Korean) hold words without the spaces the page shows (the spacing is only in the glyph positions): read the characters' positions (rawdict) and put a space where the gap is a space.
- Frames filled short (stream ended): the spread of lines is skipped when a box lies in the frame; look at 080 (frame 1 at 77%).

## Code
- Delete the old placement code in layout/refont.py (place, row_groups, follow_top, Mapper use, pour, column_gap, flow, layout, their config numbers) and move what the template uses into one place.
- Tests: cover layout/template.py (frames, fixed boxes, obstacles, rows, pictures); replace the tests of the old placement.
- Manifest: the listing has to be rebuilt once for the new language groups (python -m dspages.prep.files).

## Pipeline (do not touch geometry solve_strip / solve_sag)
- Refactor, simplify, consolidate the pipeline (generate, render, paper effects).
- Speed: noise (fbm in paper and render), cold start; measure first.
- 200 UV samples (flat, warped, dewarped, UV map, 3D map) with the new layout and the noise scale: python -m dspages all, then tools/gallery.py.
