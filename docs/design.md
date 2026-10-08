# em-pipe design

Developer reference: conventions, file formats, and the contract between steps.
Every step reads the dataset config, reads earlier steps' outputs from
`output_dir`, and writes only into its own `output_dir/<step>/` folder.
Raw data is never written to.

## Steps

| Step | Module | Parallel? | Reads | Writes |
|---|---|---|---|---|
| check | `pipeline.check` | no | raw TIFF headers | `check/` |
| preview | `pipeline.preview` | array over z-chunks, then `merge` | raw pixels | `preview/` |
| stitch | `pipeline.stitch` | array over sampled slices, then `merge` | raw pixels (sampled slices) | `stitch/` |
| align | `pipeline.align` | array over z-chunks (`run`), then `solve` | raw pixels, `stitch/` | `align/` |
| intensity | `pipeline.intensity` | no | `preview/`, `stitch/` | `intensity/` |
| destreak | `pipeline.destreak` | no (`test` only; applied inside render) | raw pixels | `destreak/` |
| zcorrect | `pipeline.zcorrect` | array over z-chunks (`run`), then `solve` | raw pixels, `stitch/`, `align/` | `zcorrect/` |
| render | `pipeline.render` | `init`, then array over z-slabs (`run`) | everything above | `render/volume.ome.zarr` (scale 0) |
| pyramid | `pipeline.pyramid` | array over shards, one job per scale | `render/` | `render/volume.ome.zarr` (scales 1..N) |
| serve | `pipeline.serve` | interactive | `render/` | – |

Order: check → preview → stitch → align → intensity → zcorrect (optional) → render → pyramid.
destreak has no batch stage: it is a function applied per tile inside render when
`destreak.enabled: true`; its CLI only writes before/after test images for tuning.

## Conventions

- **Arrays** are indexed `[z, y, x]`. **Points** are `(x, y)` = (column, row), in pixels.
- **Tile ID** is the string `"r-c"` from the filename `tile{r}-{c}`. Code must not assume
  whether `r` is the y or x direction — stitch detects the layout from the images.
- **Global slice index `z`**: 0-based position of a slice timestamp in time order over the
  whole acquisition (all non-ignored files, including excluded ones). All tiles of a slice
  share one timestamp and one `z`. New days append at the end, so `z` is stable as data grows.
  Outputs that are keyed by `z` also carry the `timestamp` column where practical.
- **Segment**: maximal run of `z` with the same tile set and tile shape (e.g. the 3×3 phase vs
  the 2×2 phase). Integer id, increasing with time. Stitch layouts are per segment.
- **Seam**: `seam=True` on the first slice after a time gap larger than
  `check.gap_factor × median interval`, or at the start of a `_partN` file with N > 1, or at a
  segment change. Alignment treats seams more carefully.
- **Transforms** are 2×3 affine matrices `A = [[a, b, tx], [c, d, ty]]` mapping source points to
  destination points: `dst = A @ [x, y, 1]`. CSV columns are `a, b, tx, c, d, ty`.
  Use `pipeline.transforms` for compose / invert / apply / CSV I/O.
  - stitch: **tile pixel → montage** coordinates of that slice (per segment the montage frame has
    tile `0-0` at the origin before any global shift; the stitch merge makes min corner = (0, 0)).
  - align: **montage → aligned** (the common frame of the whole volume).
  - render composes `align ∘ stitch` (apply stitch first) for each (z, tile).
- **Selection**: `selection:` in the config (date range and/or z range) restricts every step after
  check. Use `pipeline.slices.load_slices(cfg)`, which applies it and drops excluded rows.
- **Parallel tasks**: steps take `--task-id/--num-tasks` (default from
  `SLURM_ARRAY_TASK_ID - SLURM_ARRAY_TASK_MIN` and `SLURM_ARRAY_TASK_COUNT`; outside Slurm a
  single task does everything). Work is split into chunks and task *i* takes chunks
  `i, i+n, i+2n, …` (`pipeline.cli.my_chunks`). So array sizes come from the config, not the data.
- **Idempotent & resumable**: each task writes its outputs atomically (write to `*.tmp`, then
  rename) and skips work whose output already exists unless `--overwrite` is given.

## Output files

All paths are relative to `output_dir`.

### check/
- `files.csv` — one row per raw TIFF matching `raw.file_pattern`:
  `file` (path relative to `raw_dir`), `month, day, part, tile, tile_row, tile_col,
  n_slices, height, width, dtype, byteorder, data_offset, contiguous, size_bytes, mtime,
  voxel_x_nm, voxel_y_nm, voxel_z_nm, first_timestamp, last_timestamp, sample_hashes,
  status` (`ok | pending | error`), `excluded` (bool), `exclude_reason`.
- `slices.csv` — one row per (slice timestamp, tile):
  `z, timestamp` (ISO 8601, no timezone), `tile, tile_row, tile_col, file, index` (slice index
  within the file), `height, width, segment, seam` (bool), `excluded` (bool), `exclude_reason, label`.
- `issues.csv` — `severity` (`ERROR | WARN | INFO`), `code, file, timestamp, z, message, known` (bool).
- `report.md` — human-readable summary.
- `cache/` — per-file header results keyed by (path, size, mtime), so re-runs only read new files.
- Exit code: 1 if any ERROR that is not listed in `check.known_issues`, else 0.

### preview/
- `stats.csv` — per (z, tile): `z, timestamp, tile, mean, std, p0_5, p1, p50, p99, p99_5, min, max,
  frac_zero, frac_saturated` computed on the downsampled tile (uint16 values).
- `thumbs/z{z0:06d}-{z1:06d}_tile{r}-{c}.npy` — uint8 stack `[n, h/f, w/f]` (z0 inclusive,
  z1 exclusive), each slice autoscaled to its own p0.5–p99.5, `f = preview.factor`. Plus
  `thumbs/index.csv` (`z, tile, npy, i`) written by `merge`.
- `sheets/{YYYY-MM-DD}.png` — contact sheet per day; `stats.png` — intensity over time.

### stitch/
- `layout.json` — per segment: detected grid (tile → (grid_x, grid_y)), nominal overlap in px,
  model, mode (`fixed` or `per_slice`).
- `pairs.csv` — measured pairwise offsets per sampled z: `z, segment, tile_a, tile_b, model,
  a, b, tx, c, d, ty` (maps tile_b pixels → tile_a pixels), `n_inliers, residual_px`.
- `tiles.csv` — final transform for **every selected (z, tile)**: `z, tile, a, b, tx, c, d, ty`
  (tile pixels → montage), plus `segment`.
- `stitch.png` — offsets over z.

### align/
- `matches/chunk_{z0:06d}-{z1:06d}.npz` — arrays `z_a, z_b, pa (N×2), pb (N×2), w` with
  points in **montage** coordinates, for pairs `z_b - z_a ∈ 1..align.neighbors`.
- `transforms.csv` — every selected z: `z, timestamp, a, b, tx, c, d, ty` (montage → aligned).
- `residuals.csv` — per matched pair: `z_a, z_b, n, rms_px, max_px, rejected` (bool).
- `drift.png` — tx, ty vs z with seams marked; residuals vs z.

### intensity/
- `levels.csv` — per (z, tile): `z, tile, lo, hi` — uint16 values that render maps to 0 and 255
  (before CLAHE). Smooth over z.

### zcorrect/
- `ncc/chunk_{z0:06d}-{z1:06d}.npz` — arrays `z_a, z_b, ncc`.
- `positions.csv` — `z, position_nm` (estimated true axial position of each selected slice).
- `zcorrect.png`.

### render/
- `volume.ome.zarr` — OME-Zarr 0.5 (Zarr v3, `sharding_indexed`), uint8, scales `s0..sN`,
  axes z, y, x in nanometres; written with tensorstore.
- `render.json` — output geometry: canvas origin in aligned coordinates, shape, voxel size,
  downsample factor, source of each output plane.
- `done/` — one marker file per finished slab / shard (resume support).

## Command line

Every step is `python -m pipeline.<step> [subcommand] --config C [--raw-dir R] [--output-dir O]
[--task-id i --num-tasks n] [--overwrite] [-v]` (shared options from `pipeline.cli.base_parser`).
Each step module defines `DEFAULTS = {"<section>": {...}}` and `main(argv=None)`.

| Command | Job type | Notes |
|---|---|---|
| `pipeline.check` | single | exit 1 on new (unacknowledged) ERRORs |
| `pipeline.preview run` / `merge` | array / single | |
| `pipeline.stitch run` / `merge` | array / single | |
| `pipeline.align run` / `solve` | array / single | |
| `pipeline.intensity` | single | |
| `pipeline.zcorrect run` / `solve` | array / single | only when `zcorrect.enabled` |
| `pipeline.destreak test --z Z --tile r-c` | interactive | writes before/after PNGs |
| `pipeline.render init` / `run` | single / array | |
| `pipeline.pyramid run --scale s` | array, one job per scale (s = 1..num_scales-1) | |
| `pipeline.serve [--port 8000]` | interactive | HTTP + CORS + Range for neuroglancer |

## Slurm

`slurm/<step>.sbatch` takes the config path as its first argument and forwards the rest:
`sbatch slurm/align.sbatch configs/x.yaml solve`. Each sbatch `cd`s to the repo root
(`$EM_PIPE_ROOT`, else `$SLURM_SUBMIT_DIR`) and sources `slurm/env.sh` (modules + conda env).
Default resources are in the `#SBATCH` header; `run_pipeline.sh` overrides them per job from the
config's `slurm:` section (`array`, `cpus`, `mem`, `time`, `partition`, `gres`) and sets the log
path to `output_dir/logs/`. Array sizes never depend on the data (strided chunk assignment).
