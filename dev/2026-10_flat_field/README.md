# October 2026: flat-field measurements and comparison figures (archive)

Two tools made on 16-17 Oct 2026 while the flat-field correction (the `shading` step) and the patch-grid
tile shapes were developed. They were in `tools/` until 26 Nov 2026 and moved here because they served
that development and are not part of how a volume is made or checked. They are not maintained or
tested.

## tools/

| Tool | What it was for |
|---|---|
| `shading_measure.py` | Within-tile shading: mean brightness by position in the tile, relative to each thumbnail's own median, per tile layout segment, for tissue and for plain bright resin separately (every 20th slice, from the preview thumbnails). The shading step's own QC shows tissue only, which is what its fit used then; it was committed minutes before the change that lets plain resin into the field fit (`shading.resin_weight`). |
| `triptych.py` | Side-by-side PNGs, before / A / B, from two folders of before/after stacks of the same slices (as `tools/shading_before_after.py` writes them), optionally cropped: used to compare correction variants, and CLAHE variants with `dev/2026-10_experiments/tools/clahe_stack.py`. |

`tools/shading_before_after.py` and `tools/seam_steps.py` stay in `tools/`: the before/after stacks are
the input of `seam_steps.py`, the seam brightness check the QC summary lists.

## Running one again

From the repository root: `python dev/2026-10_flat_field/tools/shading_measure.py CONFIG OUT_DIR` (it puts
the repository root on the import path itself; it needs the preview step's thumbnails and
`work/check/slices.csv`) or `python dev/2026-10_flat_field/tools/triptych.py DIR_A DIR_B LABEL_A LABEL_B
OUT_DIR` (no pipeline imports). Neither has been run since October 2026: check that
`preview.thumb_values`, `background.mask` and the columns of `work/preview/stats.csv` they read are
still the same.
