# October 2026: stitching, alignment, normalisation and QC checks

Made while building em-pipe and running it on 35h (first a two-day test, then the whole volume) and the
first 37i data. Outputs went to the dataset's (or test run's) `qc/` folder, or to a scratch folder.

## scripts/

| Script | Question | What came of it |
|---|---|---|
| `make_realsize.py` | Does the pipeline cope with real tile sizes (memory, speed)? | A synthetic dataset of 2x2 tiles at full size with known overlaps and drift; used to find memory peaks (e.g. preview) before running on real data. |
| `eval_solvers.py` | Which stitch solver fits the real point matches best? | Comparison on matches dumped from the cluster; led to the affine-regularised-towards-rigid montage model (`stitch.model: affine_rigid`). |
| `qc_jumps.py` | Where do tiles jump or the alignment fail over a whole run? | `qc/tile_jumps.png` plus a summary; the flagged stretches became 35h's tile-level solve windows (`align.local_windows`). Now a general tool: `tools/tile_jumps.py`. |
| `spot_gifs.py`, `spot_compare.py` | What do the flagged stretches look like, before and after the tile-level solve? | Spot-check GIFs and side-by-side ImageJ stacks per window (`qc/spots`, `qc/spots_before_after`). |
| `tile_affine_check.py`, `change_check.py`, `field_check.py`, `local_experiment.py` | How much of the mismatch at a tile-layout change can per-tile affines, the tile-level solve or a smooth field remove? | The frame fit plus an elastic displacement field across layout changes (`align.segment_change_model`, `align.elastic_*`). |
| `seam_weight_check.py` | How strongly should seams be held in the tile-level solve? | Mismatch and seam movement for several weights (`align.local_seam_weight`). |
| `window_options.py`, `zprofile.py`, `norm_compare.py`, `annotate_norm.py` | Which display windows keep brightness steady through z and across tile layouts? | Tissue-only 1-99 % windows were chosen for 35h over median-anchored and all-pixel windows (`qc/normalisation_compare.tif`). |
| `masked_stats.py`, `mask_preview.py` | Can tissue-only statistics be taken from the preview thumbnails? | The background mask (`pipeline/background.py`) and the tissue-only percentiles used by `intensity`. |
| `shading_test.py` | Does the flat-field correction remove the dark tile edges? | Before/after stacks; the correction became the `shading` step. |
| `clahe_compare.py` | Which CLAHE settings look right? | Variants side by side; 35h's CLAHE volume uses clip 1.5, 256 px tiles. |

## inline/

54 scripts that were only run inline: the two-day test's checks (stitch models, joint residuals, layout,
seams, flicker and contrast variants), slice and band probes at the 3x3 -> 2x2 change, stitching a short
range of slices with the pipeline's own function, background-mask sweeps, strip-read benchmarks, seam
measurements, patch-grid predictions, clipping and speckle probes, neuropil picks, and the first checks
of 37i's raw data (the single-tile approach and its junction with the 2x2 layout). Each file's header
says when it ran, what it was for, and where.
