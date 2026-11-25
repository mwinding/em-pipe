# A1001: where the slice-to-slice flicker comes from (Nov 2026)

The A1001 volumes show horizontal stripes in x-z and y-z views: slice-to-slice brightness flicker.
Measured with `tools/zflicker.py` (whole run, the run's config for raw tiles and windows).

## Scripts

| Script | What |
|---|---|
| `scripts/tile_jump_mean_median.py` | From a `zflicker.py --config` CSV: does the flicker left after the windows follow the mean minus the median of the tiles' jumps, the raw common jump, the windows' common shift, or the middle tile's jump? |
| `scripts/levels_terms.py` | Recomputes the run's levels with em-pipe's functions, keeping their terms apart (common window, balance offset, per-slice jump), and shows which carries the left-over flicker. |

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
- Possible fix (not made): a last per-slice shift of every window by the left-over itself (the median
  over tiles of t_p50 - lo, minus its running mean), which the tiles share (0.15 greys apart).
