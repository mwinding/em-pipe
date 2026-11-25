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
