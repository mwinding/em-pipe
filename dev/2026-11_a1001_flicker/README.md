# A1001: where the slice-to-slice flicker comes from (Nov 2026)

The A1001 volumes show horizontal stripes in x-z and y-z views: slice-to-slice brightness flicker.
Measured with `tools/zflicker.py` (whole run, the run's config for raw tiles and windows).

## Scripts

| Script | What |
|---|---|
| `scripts/tile_jump_mean_median.py` | From a `zflicker.py --config` CSV: does the flicker left after the windows follow the mean minus the median of the tiles' jumps, the raw common jump, the windows' common shift, or the middle tile's jump? |
| `scripts/levels_terms.py` | Recomputes the run's levels with em-pipe's functions, keeping their terms apart (common window, balance offset, per-slice jump), and shows which carries the left-over flicker. |
| `scripts/zrelations_check.py` | The zrelations step's slice-to-slice relations (each tile against itself in the next slice) on the corrected values u, against the same on the raw thumbnails and against the preview t_p50's change, in greys of the windows. |
| `scripts/breaks_flicker.py` | Two or more level sets of one run compared before rendering: the predicted flicker (as `zflicker.py --config`) and seam steps (from intensity's in-slice relations), over the whole run and in windows around every block boundary (segment changes, tile layout changes, seams), with the step in the slices' tissue grey across each. |

## Findings so far

- The raw tiles' tissue medians jump from slice to slice: on the display scale, by about 23 greys
  (robust SD) independently per tile and 11 greys shared by the slice.
- The display windows follow each tile's own jumps almost exactly (0.15 greys left), but leave a part
  shared by the whole slice: 6.3 greys predicted (tissue median minus window, per slice).
- That left-over does not follow the raw tiles' common jump (r 0.08), the mean minus the median of the
  tiles' jumps (r 0.06) or the middle tile's jump (r 0.0); it does follow the windows' own common shift
  (r -0.47): part of it is added by the correction.
- The pyramid scales are downsampled in z, so the first measurement of the rendered volumes (scale 3)
  averaged 8 slices per plane. At full resolution (`--scale 0 --box-px 512`, 36 boxes) the plain volume
  flickers by 6.36 greys shared by the slice and 0.25 between boxes, the same as predicted from the raw
  tiles and windows (6.32), with the same worst slices (z 273, 287, 329, 248, 341: 18-27 greys).
  Decurtaining doesn't change it (6.43); CLAHE reduces it to 2.0. Lag-1 -0.11: independent jumps from
  slice to slice, not waves.
- `levels_terms.py` (lo = (L - o) / g + jump; greys, minus a 9-slice running mean, median over tiles):

  | term | shared by the slice | per tile | corr. with left-over |
  |---|---|---|---|
  | raw tile median t_p50 | 10.84 | 23.09 | +0.08 |
  | common window L (= M, the median over tiles of the balanced starting windows) | 7.23 | 0 | -0.27 |
  | balance offset -o/g | 6.72 | 22.91 | -0.06 |
  | per-slice jump | 11.23 | 0 | -0.25 |
  | lo | 11.60 | 22.91 | -0.47 |
  | left-over t_p50 - lo | 6.32 | 0.15 | 1 |

  The balance offsets carry each tile's own jump (22.9 of 23.1), which is why tiles stay matched. The
  slice-wide level is set by three per-slice statistics over the slice's tiles (the median of the
  balanced windows, the offsets' mean-zero anchor, the median jump), each moving by 7-11 greys as the
  tiles jump independently; no single one carries the left-over, and together they miss the raw
  slice-wide jump by 6.3 greys.
- The same prediction (`zflicker.py --config` alone) for the two other volumes, which use the same
  settings (window: reference or percentile, balance from overlaps every slice, per_slice: slice):

  | volume | raw jump per tile | raw jump shared by the slice | left after the windows |
  |---|---|---|---|
  | A1001 | 23.1 | 10.8 | 6.3 |
  | 35h (its current levels) | 5.7 | 45.2 | 2.9 |
  | 37i decurtained | 2.1 | 1.5 | 0.9 |

  The left-over grows with how much the tiles jump independently, as expected if it comes from
  estimating the slice-wide level from the slice's few tiles.
- Established methods (reviewed 25 Nov 2026 from code and papers; unverified where marked):
  - Janelia render (saalfeldlab/render, `newsolver` branch, Java, GPL-2.0): the one method found that
    solves tiles and slices together. Each tile in each slice gets gain/offset per 8x8 sub-tile, fitted
    from percentile matches in the in-slice overlaps and, with `--zDistance 1`, against the same tile in
    the neighbouring slices; weak pulls (0.01) towards offset-only and no change; one sub-tile pinned;
    `--tetherCrossLayers` against wrong matches at tissue-resin edges in z. Used on Janelia FIB-SEM and
    multi-SEM test stacks (how widely in production: unverified).
  - TrakEM2 "Match intensities (layer-wise)": the original of render's model (matches up to 5 slices away).
  - BigStitcher: the same scheme in 3D, for light-sheet.
  - Hemibrain: slice-wise CLAHE (and CycleGAN for segmentation input). FAFB: Kazhdan et al. 2015
    (z-smoothing of coarse brightness, then a Poisson solve per slice to keep the detail; mkazhdan/DMG).
    OpenOrganelle, FlyWire, MICrONS main text, SOFIMA: no slice-to-slice brightness step stated. MANC and
    male CNS: unverified.
  - A per-slice running-median shift (as `intensity.slice_level`) is not a named step in any of these;
    the closest are Kazhdan's z-smoothing and slice-wise CLAHE.
  - em-pipe's balance is render's model without the cross-slice matches (one gain/offset per tile,
    overlaps in the slice only). Adding the same tile's match to its neighbouring slices to that solve
    would be the established way to remove the slice-wide flicker, in place of the per-slice jump and
    slice_level.
- Possible fix (made as `intensity.slice_level`, 25 Nov 2026, to test on A1001): a last per-slice shift of every window by the left-over itself (the median
  over tiles of t_p50 - lo, minus its running mean), which the tiles share (0.15 greys apart).
- Plan 2, the established fix (made as `intensity.balance_z_distance`, 25 Nov 2026): tiles and slices balanced in one
  solve as render does with `--zDistance`. The zrelations step relates each tile to itself in the next slice from the
  tissue percentiles of the preview thumbnails (u: flat-field and patch shape applied, as the in-slice relations);
  intensity solves the gain and offset of all 487 x 9 tile slices at once from the 50,648 in-slice patch relations and
  the 4374 z relations (weak pulls to no change, 0.01 of a z relation; z relations 0.1 of a tile's in-slice ones).
  - The z relations (`zrelations_check.py`): slice-to-slice changes of 34.6 greys (robust SD, each tile at its
    centre); on u and on the raw thumbnails they differ by 0.010 greys, and from the preview t_p50's change by 0.056
    (0.04-0.10 per tile, ~130,000-320,000 tissue pixels per tile). The raw thumbnails would have done as well; u is
    used so the z relations are on the scale of the in-slice ones.
  - The solve closes the z relations to 3.5 raw units (~0.3 greys) at their centres.
  - Predicted from the raw tiles and the new windows (`zflicker.py --config`, decurtained): left after the windows
    0.055 greys shared by the slice (was 6.32) and 0.145 between tiles (was 0.15); largest single slice 0.36 greys.
  - The reference standard holds better: neuropil predicted at 149.3-151.2 greys (p5-p95 of the 17 samples; the
    per-slice balance: 134.1-158.3, as the samples fall on slices that jumped), width 129.1-130.7, no clipping
    warnings. refstats had to be run again: it measures the neuropil on the balanced scale.
- Result, rendered volumes measured at full resolution (`zflicker.py --scale 0 --box-px 512`, 36 boxes; greys):

  | volume (decurtained) | shared by the slice | between boxes | worst slice |
  |---|---|---|---|
  | plain, current | 6.43 | 0.25 | 27.3 |
  | plain, slice_level | 0.53 | 0.25 | 3.0 |
  | plain, balance_z_distance 1 | 0.065 | 0.24 | 0.31 |
  | CLAHE, current | 2.03 | 0.77 | 9.1 |
  | CLAHE, slice_level | 0.20 | 0.62 | 1.0 |
  | CLAHE, balance_z_distance 1 | 0.10 | 0.64 | 0.40 |

  Both remove the stripes in x-z and y-z views; the joint solve leaves about a tenth of what slice_level leaves,
  without a correction after the fact, and holds the neuropil standard tighter (149.3-151.2 vs 143.6-152.9).

## 35h: the joint solve on 17,689 slices (26 Nov 2026)

`intensity.balance_z_distance: 1` on all of 35h (6 segments, 3x3 then 2x2 tiles, 17 blocks: 5 segment changes, all
also tile size changes, one from 3x3 to 2x2 tiles, and 11 seams), other settings as 35h's current levels (tissue 1-99 %
percentile windows, overlaps + patchgrid balance).

- Scaling: zrelations 87,117 relations, 52 min wall over a 30-task array (3x3 chunks ~1.5 min per 100 slices, 2x2
  chunks of the larger tiles ~6-7 min), at most 2.1 GB per task; merge 75 s; intensity (joint solve per block, up to
  12,636 tile slices in one) 41 s and 0.8 GB.
- Predicted (`breaks_flicker.py`): flicker shared by the slice 2.89 -> 0.037 greys, between tiles 0.175 -> 0.093, seam
  steps unchanged (median 1.35 greys, p95 4.8); around every break 0.03-0.07 (was 1.6-3.7).
- Measured on a 512-slice slab across the 3x3 -> 2x2 change (full resolution, 36 boxes): 1.84 -> 0.081 greys.
- Steps at breaks: each block's common scale is anchored to the raw scale only by the weak prior, so blocks sit on
  scales hundreds of raw units apart; with window: percentile (or median) the common window is taken on that same
  scale, so the anchor cancels and the steps are as with the per-slice balance (predicted: largest -6.3 vs -4.5 greys
  at one segment change, others the same or smaller; rendered slab: +10.2 vs +10.0 at the 3x3 -> 2x2 change, from the
  per-block windows). With window: reference the neuropil statistics (refstats) must be measured on the joint scale:
  statistics from the per-slice balance, reused, place each block's neuropil off by the difference between the two
  scales.
