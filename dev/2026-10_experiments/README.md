# October 2026: tile-intensity and CLAHE experiments (archive)

Two sets of experiment scripts from October 2026 (added to the repository on 17 Oct 2026), moved here
from `tools/experiments/` and `tests/experiments/` on 26 Nov 2026. They are kept as records of how
two parts of the pipeline were arrived at. They are not maintained and not tested: the tests in
`tests/` here are no longer part of the pipeline's suite (`pytest tests`).

- The `janelia_*` scripts tried Janelia-style intensity matching (after saalfeldlab/render's
  intensity-adjust client): brightness relations per pair of patches in the tile overlaps and a
  smooth intensity map per tile solved from them. This became the patch relations of the overlaps
  step (`overlaps.patch_grid`) and the `patchgrid` step (`pipeline/patchgrid.py`).
- The `clahe_*` scripts compared CLAHE settings and where CLAHE is applied (on the 32 nm test
  slices, or on the full 8 nm plane as render does), and measured what CLAHE does to the brightness
  steps at tile seams. They are the background to the CLAHE settings (`render.clahe`).

## tools/

| Script | What it was for |
|---|---|
| `janelia_bins.py` | Step 1 of the Janelia-style matching: the brightness relation of every pair of patches in every tile overlap (each tile split into a grid of patches). |
| `janelia_solve.py` | Step 2: a smooth linear intensity map per tile solved from those relations, and the common display window per slice (variant A: solved at every slice; variant B: a within-tile shape per tile position plus one level per tile and slice). |
| `janelia_windows.py` | Per-(z, tile) display windows without tile balance, as input for the solve. |
| `janelia_render.py` | Test slices rendered at 32 nm with the solved patch maps, as before/after ImageJ stacks. |
| `janelia_tune.py` | Parameter sweep of the solve on the predicted seam steps, without rendering. |
| `janelia_seam_steps.py` | Brightness steps across tile seams, in tissue and in plain resin, for before/after stacks: a copy of `tools/seam_steps.py` with the output CSV as first argument, a slice list from a file, and array tasks. |
| `janelia_across.py` | Whether the brightness relation of two tiles changes across the width of their overlap (tile-edge effects). |
| `janelia_flicker.py` | Slice-to-slice brightness change of tile interiors in before/after stacks of consecutive slices. |
| `clahe_variants.py` | CLAHE applied to the pipeline's 32 nm test slices, for a set of clip limits and tile sizes. |
| `clahe_8nm.py` | CLAHE computed as the real volume gets it: on the full 8 nm plane, then downsampled to 32 nm like the pyramid. |
| `clahe_stack.py` | 2-frame stacks for `triptych.py` (`dev/2026-10_flat_field/tools/`, in `tools/` until 26 Nov 2026) from single 32 nm planes. |
| `clahe_ownership.py` | Which tile owns each 32 nm pixel, and where each pair of tiles meets. |
| `clahe_seam_steps.py` | The seam steps of `tools/seam_steps.py`, with the local contrast next to each seam and measures of CLAHE artefacts (resin noise, resin spread, halo). |
| `clahe_summary.py` | Summary tables of `clahe_seam_steps.py`'s output. |

Usage is in each script's docstring. After the move only the paths were changed: the scripts find
the repository root and `tools/` (for `tools/seam_steps.py`) from their new place, and the usage
lines name the new paths. Checked on 26 Nov 2026: every script still starts (its imports resolve
and it prints its usage or stops at its first missing argument), and the functions of `pipeline/`
and `tools/seam_steps.py` that the scripts name still exist. None was run on data again, so it is
not known whether those functions (`render.plane_meta`, `overlaps.fit_relation`,
`intensity.compute_levels`, render's internals such as `_Renderer._warp`) still take the arguments
and give the results these scripts expect.

## tests/

| File | What |
|---|---|
| `test_janelia_solve.py` | Synthetic test of `janelia_solve.py` (4 tests): four tiles with within-tile gradients that one level per tile cannot match and the patch maps can. Runs from here with `python -m pytest dev/2026-10_experiments/tests`. |
