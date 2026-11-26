# 26 November 2026: shared small helpers, checked before and after

Two small helpers that were written out in many places became shared functions, without changing any
result:

- `pipeline/stats.py` `robust_sd` (1.4826 x the median absolute deviation from the median), in place of
  the formula written out in `align._reject`, `tilesolve.solve_window`, `curtains.peak` and
  `measure_window` (twice), `overlaps.fit_relation`, `intensity.follow_slice_level`, `sample_noise` and
  `_log_joint`, and in the tools `zflicker.py` (its `robust_sd`, which keeps dropping non-finite values),
  `slice_quality.py` (`robust_z`, `lone_jump`), `refmask_check.py` (`metrics`) and `qc_summary.py`.
- `pipeline/features.py` `stretch_to_uint8` ((img - lo) / (hi - lo) x 255, clipped, truncated to uint8),
  in place of the copies in `raw_stats.py` (`to8`, removed), `slice_quality.py` (`to8`),
  `clipping_check.py` (`tissue_mask`) and `seam_steps.py` (`classes`).

Each replacement computes the same expression in the same order on the same dtype, so results are equal
bit for bit. `tests/test_stats.py` pins the helpers against the written-out expressions.

## The check

`scripts/helper_outputs.py` runs every function above on fixed synthetic data (point-match residuals with
outliers, a tile solve with bad matches, curtain windows with and without EsB, overlap pairs with
outliers, slice-level and joint-balance inputs, flicker and quality series with NaNs, tissue and resin
tiles and planes) and saves the outputs; values that are only logged are taken from the log records,
unrounded. Run on a worktree of the commit before the change (4329672) and on the changed tree:

    python dev/2026-11_shared_helpers/scripts/helper_outputs.py BEFORE_CHECKOUT before.npz
    python dev/2026-11_shared_helpers/scripts/helper_outputs.py . after.npz
    python dev/2026-11_shared_helpers/scripts/helper_outputs.py --compare before.npz after.npz

Result on 26 Nov 2026: 36 of 36 outputs identical (same dtype, shape and bytes).

## Left as they were, and why

- `align._fit_frame`'s threshold `3 * 1.4826 * MAD`: multiplied in that order, it could round differently
  in the last bit from 3 x `robust_sd`.
- `destreak.noise_sd` (the MAD of a wavelet band about zero / 0.6745, Donoho's estimator) and
  `standard_compare.noise_sd` (Immerkaer's, now in `dev/2026-11_reference_standard/`): other estimators.
- `features.to_uint8` multiplies by 255 / (hi - lo) where the tools divide by (hi - lo): on 20 million
  test pixels 151 came out one grey apart, so the pipeline's SIFT inputs keep their own function.
- `features.downsample` (float32, whole image, rounded output size) and `raw_stats.downsample` (crops to
  whole blocks, averages in the input's dtype, so integer data come out rounded; up to 0.5 apart);
  `pyramid.downsample` is another operation (2 x 2 x 2 blocks of the pyramid).
- `sample_z`: `refstats` (every n-th slice per segment and the last), `stitch` (also the slices either side
  of each seam, with excluded rows) and `raw_stats` (n slices spread between 5 % and 95 %) choose different
  slices.
- `running_median`: `intensity` (a pandas Series, at least 1 value) and `slice_quality` (columns of an
  array, at least 5 values, else NaN) differ in the minimum count.
- md5 of a file: `upload.md5_b64` (base64, as GCS reports it) and `copy_verify.md5_file` (hex, in a tool
  that uses only the standard library); `split_stack` hashes byte ranges.
- Slice lists read by hand (`qc_summary`, `segment_overview`, `reference_preview`): they read
  `work/check/slices.csv` as written, without the config's selection or timezone check (and `qc_summary`
  also when the file is missing, `segment_overview` with the timestamps as text); `slices.load_slices`
  would change which slices they see or what they print.
