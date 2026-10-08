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
  Caveat: `z` is stable only while the set of readable timestamps grows. If an old file becomes
  pending again (re-copied or touched within `check.min_age_minutes`), unreadable or removed, its
  slices drop out for that run and every later `z` shifts until it is back; outputs keyed by `z`
  from such a run are then stale (re-run the affected steps with `--overwrite`).
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
  Selected z may have holes (exclusions): neighbours are by position in the sorted selected z.
- **Voxel size**: `pipeline.slices.voxel_size_nm(cfg, files)` — per axis the median over the used
  files in `check/files.csv`, else 8 nm. render and zcorrect both use it.
- **Parallel tasks**: steps take `--task-id/--num-tasks` (default from
  `SLURM_ARRAY_TASK_ID - SLURM_ARRAY_TASK_MIN` and `SLURM_ARRAY_TASK_COUNT`; outside Slurm a
  single task does everything). Work is split into chunks and task *i* takes chunks
  `i, i+n, i+2n, …` (`pipeline.cli.my_chunks`). So array sizes come from the config, not the data.
- **Idempotent & resumable**: each task writes its outputs atomically (write to `*.tmp`, then
  rename) and skips work whose output already exists unless `--overwrite` is given. Chunk files
  record what they were computed from (slices, settings) and are redone when that changes;
  merge/solve steps refuse missing or out-of-date chunks.

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
  The tile comes from the file name. (z, tile) is unique. Timestamps are as written in the labels,
  or naive UTC when `check.timezone` is set (selection start/end stay in local time and are
  converted).
- `issues.csv` — `severity` (`ERROR | WARN | INFO`), `code, file, timestamp, z, message, known` (bool).
  Codes: ERROR `UNREADABLE, TRUNCATED, LABEL_COUNT, LABEL_PARSE, TILE_MISMATCH, NON_MONOTONIC,
  TIMESTAMP_MISMATCH, DUPLICATE_CONTENT` (a byte copy of another tile; the copy is excluded),
  `DUPLICATE_SLICE` (same timestamp and tile in two files; one kept), `TILE_SHAPE` (tile shaped unlike
  the others of its slices; excluded), `MISSING_TILE` (also where the grid shrinks with the same tile
  shape), `NO_FILES`; WARN `DTYPE, VOXEL_SIZE, TIME_GAP, UNAVAILABLE`; INFO `RESTART, SEGMENT_CHANGE,
  KNOWN_ISSUE_RESOLVED` (a known_issues entry with no ERROR or WARN now).
- `report.md` — human-readable summary.
- `cache/` — per-file header results keyed by (path, size, mtime), so re-runs only read new files.
- Pending files (modified within `check.min_age_minutes`) are not read; the other files of their
  (month, day, part) are excluded ("waiting for pending ...") but keep their z, so no slice is
  processed with tiles missing. A z whose tiles are all excluded keeps the previous z's segment, and
  a seam on it moves to the next usable z.
- `known_issues` entries `{file, note, action}`: `exclude` excludes the file's slices (reason = note);
  `ignore` only acknowledges. `file` is the path relative to `raw_dir` or the file name.
- Exit code: 1 if any ERROR that is not listed in `check.known_issues`, else 0.

### preview/
- `stats.csv` — per (z, tile): `z, timestamp, tile, mean, std, p0_5, p1, p50, p99, p99_5, min, max,
  frac_zero, frac_saturated` computed on the downsampled tile (uint16 values).
- `thumbs/z{z0:06d}-{z1:06d}_tile{r}-{c}.npy` — uint8 stack `[n, h/f, w/f]` (z0 = first selected z
  of the chunk, z1 = last + 1), each slice autoscaled to its own p0.5–p99.5, `f = preview.factor`.
  Plus `thumbs/index.csv` (`z, tile, npy, i`) written by `merge`: `npy` is a file name in `thumbs/`,
  `i` the position in that stack. `stats/chunk_{z0:06d}-{z1:06d}.csv` holds each chunk's stats.
- Chunks are fixed blocks of global z, one per (segment, z // `preview.chunk_slices`): they never span
  a segment change, so a tile's thumbnails in one chunk share a shape. `run` recomputes a chunk whose
  stats lack a selected (z, tile); `merge` reads only the chunks of the current selection.
- `sheets/{YYYY-MM-DD}.png` — contact sheet per day; `stats.png` — intensity over time.

### stitch/
- `samples/z{z:06d}.json` — one per sampled slice (every `stitch.sample_every`-th selected z of a
  segment, its first and last, and the slices either side of each seam): `z, timestamp, segment,
  model, reference` (the segment's smallest tile id, fixed at identity), `settings, ok, reason,
  tiles` (tile → montage 2×3), `pairs` (`tile_a, tile_b, coarse_tx, coarse_ty, coarse_inliers, A,
  n_inliers, residual_px, solve_residual_px, used, note`). `run` redoes samples whose segment,
  reference or settings no longer match; `merge` refuses them.
- `layout.json` — `{"segments": {"<id>": {segment, z_first, z_last, tile_shape [h, w],
  reference_tile, model, mode, max_deviation_px, n_samples, n_good_samples, failed_samples [z],
  grid_shape [ny, nx], row_axis "y"|"x"|null, overlap_px {x, y}, tiles {tile: {grid_x, grid_y, x, y}}}}}`.
- `pairs.csv` — measured pairwise offsets per sampled z: `z, segment, tile_a, tile_b, model,
  a, b, tx, c, d, ty` (maps tile_b pixels → tile_a pixels), `n_inliers, residual_px,
  solve_residual_px` (after the per-slice solve), `used` (kept in that solve).
- `tiles.csv` — final transform for **every selected (z, tile)**: `z, timestamp, tile, a, b, tx, c,
  d, ty` (tile pixels → montage), `segment`. One shift per segment puts the montage min corner over
  all its slices at (0, 0).
- `stitch.png` — offsets over z.

### align/
- `matches/chunk_{z0:06d}-{z1:06d}.npz` — the selected z in the global-z block `[z0, z1)` of
  `align.chunk_slices`, each paired with its next `align.neighbors` selected z (by position, so
  `z_b - z_a` can exceed `neighbors`). Arrays `z_a, z_b, w`; points in tile pixels `qa, qb` (N×2) with
  tile indices `ta, tb` into `tiles`; `pa, pb` the same in montage coordinates at run time; `slices`
  ('z:tile' read) and `settings` (redone when either changes). `solve` maps `qa, qb` with the
  current stitch/tiles.csv, so re-running stitch never needs `align run --overwrite`. Within a
  segment only the same tile is matched; across a segment change every tile pair.
- `transforms.csv` — every selected z: `z, timestamp, a, b, tx, c, d, ty` (montage → aligned).
- `residuals.csv` — per matched pair: `z_a, z_b, n, rms_px, max_px, rejected` (bool): the global fit
  before trend removal, over the points used (all points of a rejected pair). A pair is rejected
  when fewer than half its points survive outlier rejection.
- `drift.png` — tx, ty vs z with seams marked; residuals vs z.

### intensity/
- `levels.csv` — per (z, tile): `z, tile, lo, hi` — uint16 values that render maps to 0 and 255
  (before CLAHE). Smooth over z.

### zcorrect/
- `ncc/chunk_{z0:06d}-{z1:06d}.npz` — arrays `z_a, z_b, ncc`, plus `zs` (the z measured over),
  `crop_px, factor`; chunks are runs of `zcorrect.chunk_slices` selected z (z1 = last + 1).
- `positions.csv` — `z, timestamp, position_nm` with position_nm = voxel_z × (first selected z +
  cumulative estimated spacing in slices): uncorrected slices sit at z × voxel_z. The spacing at a
  seam or segment change is nominal, a hole in the selection counts as several nominal slices, and the
  mean spacing is held at nominal per seam-free block and over a rolling `zcorrect.nominal_window`.
- `zcorrect.png`.

### destreak/
- `z{Z}_tile{r-c}_{before,after}.png` (whole slice, ≤ 2000 px), `_{before,after}_crop.png` (full
  resolution), `_{before,after}_fft.png` (log spectrum of the crop); before and after share contrast.

### render/
- `<render.name>` (default `volume.ome.zarr`) — OME-Zarr 0.5 (Zarr v3, `sharding_indexed`), uint8,
  scales `s0..sN`, axes z, y, x in nanometres; written with tensorstore.
- `<stem>/` — the volume's work folder (`<stem>` = render.name without `.ome.zarr`, e.g. `volume/`),
  holding the three items below. Per volume, so several renders (e.g. a 32 nm overview and a
  full-resolution region with its own `render.name` and `render.bbox`) can share one output_dir.
- `<stem>/tiles.csv` — the plan frozen by `init`, one row per selected (z, tile): `z, tile, file, index,
  height, width, a, b, tx, c, d, ty` (tile pixel → aligned = align ∘ stitch, rounded when
  `integer_shifts`), `lo, hi` (NaN where intensity has no levels: the tile's own p0.5/p99.5 is used).
- `<stem>/render.json` — `volume, origin_xy` (aligned px of output pixel (0, 0)), `canvas_size_xy, shape
  [z, y, x], voxel_nm [z, y, x]` (× downsample), `downsample, zcorrected, shard, settings` (render
  section minus threads), `destreak` (section or null), `planes` (`[[z, weight], ...]` per plane),
  `digest` (of all the above and tiles.csv). `run` reads only these and the raw tiles.
- `<stem>/done/slab_{k:06d}` — render's marker per finished slab, holding the digest (counts only for the
  current plan); `done/s{s}_{index:06d}` — pyramid's marker per shard of scale s (index in
  `omezarr.shard_boxes` C order), valid only if newer than `s{s}/zarr.json` and its source shards.
- `init` with unchanged inputs and settings does nothing; if they changed it exits 1 unless
  `--overwrite`, which deletes the volume and every marker and re-creates them. New data therefore
  needs `init --overwrite` and a full re-render (align's global solve changes earlier transforms).

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
| `pipeline.serve [--port 8000] [--bind 127.0.0.1]` | interactive | HTTP + CORS + Range for neuroglancer (`serve.port`, `serve.bind`); an SSH tunnel through the login host needs `--bind 0.0.0.0` |
| `pipeline.config --get KEY... / --sbatch-args [JOB...]` | – | config values (step defaults included) and sbatch options for `run_pipeline.sh` |

## Slurm

`slurm/<step>.sbatch` takes the config path as its first argument and forwards the rest:
`sbatch slurm/align.sbatch configs/x.yaml solve`. Each sbatch `cd`s to the repo root
(`$EM_PIPE_ROOT`, else `$SLURM_SUBMIT_DIR`) and sources `slurm/env.sh` (modules + conda env).
Default resources are in the `#SBATCH` header; `run_pipeline.sh` overrides them per job from the
config's `slurm:` section (`array`, `cpus`, `mem`, `time`, `partition`, `gres`) and sets the log
path to `output_dir/logs/`. Array sizes never depend on the data (strided chunk assignment).

`./run_pipeline.sh CONFIG [--steps a,b,...] [--from STEP] [--dry-run]` submits one chain, each job
`--dependency=afterok` on the previous one (`--kill-on-invalid-dep=yes`, so a failed job, or check
exiting 1 on new errors, cancels the rest): check; preview run → merge; stitch run → merge; align
run → solve; intensity; zcorrect run → solve (only if `zcorrect.enabled`); render init → run;
pyramid run --scale s for s = 1..num_scales-1. Job names are `em-<step>[-<sub>]` (e.g.
`em-preview-run`, `em-pyramid-s2`), logs `output_dir/logs/%x-%A_%a.out` (arrays) or `%x-%j.out`.
`slurm:` keys per job: `check, preview, preview_merge, stitch, stitch_merge, align, align_solve,
intensity, zcorrect, zcorrect_solve, render_init, render, pyramid` (array sizes apply to the array
jobs `preview, stitch, align, zcorrect, render, pyramid`), plus `slurm.mail_user`. `time` must be a
quoted string. `EM_PIPE_SKIP_ENV=1` skips sourcing `slurm/env.sh` in the script itself.
