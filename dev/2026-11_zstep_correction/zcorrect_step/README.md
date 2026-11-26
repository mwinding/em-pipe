# The zcorrect step (slice-thickness correction)

Removed from the pipeline on 26 Nov 2026. `zcorrect` estimated each slice's position along z from
how alike neighbouring slices are (after Hanslovsky et al. 2017, simplified), and render then placed
planes at even spacing, interpolating between slices. It was an optional step between intensity and
render, off by default. The step, its Slurm script and its tests are kept here as they were.

**Why:** it was tested on two volumes (A1001 and 37i) and is not used. The per-slice estimate mostly
followed how different a slice looks (noise, streaks) and not its thickness, independent tiles of the
same slices did not agree on it, and applying it blended nearly every plane from two slices without
making shapes more even; the real thick cuts, after long pauses, are breaks the step leaves at
nominal spacing. The measurements and the decision are in the parent folder's
[README](../README.md).

| File | What |
|---|---|
| `zcorrect.py` | The step (was `pipeline/zcorrect.py`), imports changed so it runs from here: `python dev/2026-11_zstep_correction/zcorrect_step/zcorrect.py run\|solve --config CONFIG`. Writes `work/zcorrect/` and `qc/zcorrect.png` as before. The scripts in `../scripts/` import it from here. |
| `zcorrect.sbatch` | Its Slurm script (was `slurm/zcorrect.sbatch`), calling the file above. |
| `render_plane_plan.py` | The code taken out of `pipeline/render.py`: reading `positions.csv` and placing planes between slices, with notes on where render called it. A record; nothing imports it. |
| `test_zcorrect.py`, `conftest.py` | The step's unit tests (was `tests/test_zcorrect.py`, 8 tests). They pass from here: `python -m pytest dev/2026-11_zstep_correction/zcorrect_step/test_zcorrect.py`. |
| `archived_test_render.py` | From `tests/test_render.py`: `test_zcorrect_plane_mapping` and the positions half of `test_plane_sources`. Not collected by pytest; cannot pass while render does not read positions. |
| `archived_test_e2e.py` | From `tests/test_e2e.py`: `test_zcorrect_positions_drive_the_planes` and the zcorrect part of the `corrected_run` fixture. Not collected by pytest; cannot pass while the step is not a pipeline module. |

Its working files, for output folders that still hold them (`work/zcorrect/`):
`ncc/chunk_{z0:06d}-{z1:06d}.npz` (arrays `z_a, z_b, ncc`, plus `zs`, `crop_px`, `factor`) and
`positions.csv` (`z, timestamp, position_nm`; uncorrected slices sit at z × voxel_z, the spacing at
a seam or segment change is nominal).

## What the pipeline still knows about it

- `pipeline/preflight.py` accepts a config whose `zcorrect` section only says `enabled: false`, and
  `slurm.zcorrect` / `slurm.zcorrect_solve` entries, each with a note that it has no effect.
  `zcorrect.enabled: true` or any other key in the section is an error (render init refuses
  `enabled: true` too), so a config for running `zcorrect.py` from here does not pass
  `run_pipeline.sh`.
- `render.json` keeps the key `zcorrected` (always false now): it is part of the plan's digest, so
  existing volumes still count as unchanged. `render.append` refuses a volume whose `render.json`
  says it was z-corrected.
- `tools/qc_summary.py` lists the estimated spacing, as information, where
  `work/zcorrect/positions.csv` is left from an earlier run.

## What it would need to be revived

- **render's plane plan** (`render_plane_plan.py`): `_positions` and the positions branch of
  `plane_sources` back in `pipeline/render.py`; `init` passes the positions (positions.csv to planes
  interpolated between slices) and sets `"zcorrected": positions is not None`; `_append` refuses
  `zcorrect.enabled` again; the check that refuses `zcorrect.enabled: true` goes. `run` needs
  nothing.
- **The module, the config section and the Slurm jobs:** `zcorrect.py` back to
  `pipeline/zcorrect.py` (package-relative imports) and `zcorrect.sbatch` to `slurm/`; `zcorrect` in
  `STEPS`, `STEPS_IN_ORDER`, `JOBS` (`zcorrect`, `zcorrect_solve`) and `ARRAY_JOBS` of
  `pipeline/config.py`, with the default `slurm.zcorrect: {array: 10}`, and out of `REMOVED_STEPS` /
  `REMOVED_JOBS`; the section and the Slurm entry in `configs/example.yaml`; the two jobs in
  `pipeline/resources.py` (`JOBS` `(2, 16, 20)` and `(2, 16, None)`, `PER_GPX` 6, `FIXED_HOURS` 1
  for the solve, array units `ceil(slices / zcorrect.chunk_slices)`).
- **run_pipeline.sh's step:** `zcorrect` in `STEPS` after intensity, `zcorrect.enabled` among the
  values read from the config, and the block that submits `em-zcorrect-run` (array) and
  `em-zcorrect-solve` when it is enabled.
- **The tests:** `test_zcorrect.py` back to `tests/`; the archived tests back into
  `tests/test_render.py` and `tests/test_e2e.py`; and the cases in `tests/test_run_pipeline.py`,
  `tests/test_preflight.py`, `tests/test_onecommand.py` and `tests/test_qc_summary.py` that now test
  that the step is gone.
- **The documents:** `README.md` (diagram, step table), `docs/design.md` (step and job tables,
  `work/zcorrect/`) and the QC page's "Slice thickness" row.

The parent README's recommendation comes first: the estimate needs a per-slice scaling factor and
resin masking, and tiles must agree on the spacing (`../scripts/step_stats.py`), before it is
applied to a volume.
