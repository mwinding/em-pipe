# November 2026: setting the contrast standard (archive)

Three tools made on 19-20 Nov 2026 while the contrast standard of `intensity.window: reference` was set
(neuropil median at grey 150, neuropil p5-p95 width 130 greys; README, "Consistent contrast across
volumes"; now in `configs/presets/standard.yaml`). They were in `tools/` until 26 Nov 2026 and moved here
because they served that one decision: the standard is settled and kept fixed, so they are not part of
how a volume is made or checked. They are not maintained, and their test is no longer part of the
pipeline's suite (`pytest tests`).

## tools/

| Tool | What it was for |
|---|---|
| `reference_from_volume.py` | The starting point: the reference volume's (35h's) neuropil numbers in greys under the levels it was rendered with (refstats' per-slice median, p5 and p95 mapped through `work/intensity/levels.csv`): median, p5, p95, p5-p95 width and lc32, per sampled slice and as the median over slices. |
| `standard_compare.py` | The choice of the width: crops of several volumes (columns) at candidate p5-p95 widths (rows), with the share of each crop and of the volume's median slice that clips to black or white, and the pixel noise against the rounding to whole greys; optionally the same crops after render's CLAHE. Two of its figures are in the explainer (`docs/pipeline_explainer.html`, the width comparison). |
| `refstats_compare.py` | Two refstats runs of one volume lined up by z and compared per segment (median, p5-p95 width, lc32, their ratios): used on 37i with and without decurtaining, which showed that decurtaining leaves the neuropil median as it is and narrows the width by about 1 % (2-3 % where the curtains are strongest). Its figure is in the explainer (the decurtaining section). |

`tools/reference_preview.py`, which previews new reference levels on crops of a rendered volume, stays in
`tools/`: the README names it as the preview step when an existing volume is put on the standard.
`standard_compare.py` imports its crop picking and level mapping from there.

## tests/

| File | What |
|---|---|
| `test_standard_compare.py` | `standard_compare.py`'s other widths (a stretch around the median grey), its pixel-noise estimate and its whole-volume clipping, on synthetic data (3 tests). Runs from here: `python -m pytest dev/2026-11_reference_standard/tests`. |

## Running one again

From the repository root, e.g. `python dev/2026-11_reference_standard/tools/standard_compare.py SPEC.json
OUT.png` (usage in each docstring). The tools put the repository root (for `pipeline/`) and `tools/` (for
`reference_preview.py` and `refmask_check.py`) on the import path themselves. They were last run on
20 Nov 2026; before relying on one, check that the pipeline functions it calls still take the same
arguments and give the same columns: `intensity.load_refstats`, `balanced_scale`, `common_window`,
`levels_path`, `reference_settings`, `_qc_name`, `grey`, the reference QC's `_samples.csv` columns, and
`render._clahe`.
