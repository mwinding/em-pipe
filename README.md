# em-pipe

Processing pipeline for tiled FIB-SEM acquisitions saved as one ImageJ TIFF stack per tile per day
(e.g. `M10_D06_tile1-1.tif`). It checks the raw data, stitches the tiles of each slice, corrects
slice-to-slice drift, normalises intensity and writes a multi-resolution volume for
[neuroglancer](https://github.com/google/neuroglancer). It runs as Slurm jobs on NEMO.

```
check → preview → stitch → align → intensity → [zcorrect] → render → pyramid
```

| Step | What it does |
|---|---|
| check | Inventory of the raw files: truncated or unreadable files, copied tiles, label/filename mismatches, missing tiles, time gaps, restarts and grid changes. Writes the global slice list every later step uses. |
| preview | Per-slice intensity statistics and thumbnails (contact sheet per day). |
| stitch | Tile layout within each slice: SIFT matches in the tile overlaps (overlap found automatically) and a least-squares solve. |
| align | Slice-to-slice drift: SIFT matches between each slice and its next few neighbours, then one global least-squares solve. |
| intensity | Per-tile contrast levels, balanced across tile overlaps and smoothed over time. |
| zcorrect | Optional slice-thickness correction (off by default). |
| render | Applies everything in one pass and writes OME-Zarr (Zarr v3, sharded), optionally downsampled; CLAHE; optional destreaking. |
| pyramid | Lower-resolution scales of the volume. |

## Setup on NEMO (once)

```bash
ml Anaconda3/2024.10
conda env create -f environment.yml -p /camp/lab/windingm/home/shared/conda-envs/em-pipe
```

`slurm/env.sh` loads this environment in every job (set `EM_PIPE_ENV` to use another one).

## Running

Each dataset has a YAML config in `configs/` with `raw_dir`, `output_dir`, an optional
`selection` (date or z range) and per-step settings. `configs/example.yaml` lists every setting
with its default.

```bash
./run_pipeline.sh configs/P667_test_2day.yaml --dry-run   # show the job chain
./run_pipeline.sh configs/P667_test_2day.yaml             # submit it
squeue --me                                               # follow; logs in <output_dir>/logs/
./run_pipeline.sh configs/P667_test_2day.yaml --from align   # re-run from a step
```

Every job waits for the previous one (`afterok`), so a failed step cancels the rest. Steps skip
work whose output already exists, so re-submitting resumes. Array sizes, CPUs, memory and time
come from the config's `slurm:` section.

Any step can also run directly, e.g. on a Mac against the mounted share:

```bash
python -m pipeline.check --config configs/P667_test_2day.yaml \
    --raw-dir /Volumes/proj-efibsem/WindingM/P667_EM05024_35h --output-dir ~/em-test
```

## Outputs (in `output_dir`)

- `check/report.md`: what was found in the raw data. Acknowledge known problems in
  `check.known_issues` (`action: exclude` drops a file), otherwise check stops the pipeline.
- `preview/sheets/*.png`, `preview/stats.png`: thumbnails and intensity over time.
- `stitch/stitch.png`, `stitch/layout.json`: tile offsets over time, measured overlap.
- `align/drift.png`: drift correction over time and match residuals.
- `intensity/intensity.png`
- `render/volume.ome.zarr`: the volume.

## Viewing in neuroglancer

On a NEMO compute node (e.g. `srun --pty bash`), then from your Mac:

```bash
python -m pipeline.serve --config configs/P667_test_2day.yaml --bind 0.0.0.0   # on the node
ssh -L 8000:<node>:8000 <user>@login.nemo.thecrick.org                         # on the Mac
```

In https://neuroglancer-demo.appspot.com add a layer with source
`zarr3://http://localhost:8000/volume.ome.zarr/` (the server prints the exact URL).

## Things to know

- **New data:** align's global solve changes earlier slices' transforms, so after adding days,
  re-create the volume: `sbatch slurm/render.sbatch CONFIG init --overwrite`, then
  `./run_pipeline.sh CONFIG --from render`.
- **Clock change (25 Oct 2026):** if the label clock is UK local time, set
  `check.timezone: Europe/London` before then (the repeated hour would otherwise break the slice
  order). Re-run check after changing it; later steps refuse a mismatched timezone.
- **Several volumes** (e.g. a 32 nm overview and a full-resolution `render.bbox` region) can share
  one `output_dir` if they have different `render.name`s.
- **destreak and zcorrect** are off by default; tune destreak with
  `python -m pipeline.destreak test --config C --z Z --tile 0-0` first.
- Design notes and file formats: `docs/design.md`. Tests: `python -m pytest tests`.
