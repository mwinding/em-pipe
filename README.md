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
| stitch | Tile layout within each slice: SIFT matches in the tile overlaps (overlap found automatically) and a least-squares solve, on every 50th slice. Where tiles move between two of those by more than `stitch.refine_px` (5 px), the slices in between are stitched too (bisection), so a sudden tile jump is placed at the right slice instead of being interpolated. |
| align | Slice-to-slice drift: SIFT matches between each slice and its next few neighbours, then one global least-squares solve. Where the tile layout changes (e.g. 3×3 → 2×2 tiles), one affine transform between the two montage frames is fitted first (`align.segment_change_model`; results in `work/align/segment_changes.csv`). Where that leaves a median mismatch above `align.elastic_min_px` (5 px; 35h: only 3×3 → 2×2), a smooth displacement field is fitted to the matches across the change and split between the slices either side (`pipeline/elastic.py`; `work/align/elastic_interfaces.csv` reports the mismatch before and after); render warps those slices and cuts sharply between tiles there (`render.local_blend`). A tile-level affine solve (`align.local_*`, off by default) is also available. |
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
squeue --me                                               # follow; logs in <output_dir>/work/logs/
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

```
<name>.ome.zarr        the stitched volume: open in neuroglancer (below)
<name>_<voxel>nm.tif   quick look in Fiji (finest scale under export.max_gb)
qc/                    check_report.md and plots: stitch, align_drift, intensity, preview_stats, daily sheets
work/                  working files of every step and the Slurm logs (work/logs/); no need to open
```

`qc/check_report.md` lists what was found in the raw data. Acknowledge known problems in
`check.known_issues` (`action: exclude` drops a file), otherwise check stops the pipeline.

## Viewing a volume (anyone in the lab)

A shared copy of this repo lives on the lab share, so there is nothing to install: you need Python 3
(the Mac's built-in `python3` is fine), Chrome, and the lab share mounted (Finder → Go → Connect to
Server → `smb://data2.thecrick.org/lab-windingm`).

```bash
cd /Volumes/lab-windingm/home/shared/repos/em-pipe
python3 -B -m pipeline.serve /Volumes/lab-windingm/home/users/windinm/em-pipe_P667_test-M10-D01-D02
```

(`-B` stops Python writing cache files into the shared copy. Please don't edit code there: it only
`git pull`s from GitHub; develop in your own clone.)

It prints a neuroglancer link with every volume in that folder as a layer (click a layer's name
to show or hide it). Keep the terminal open while viewing; Ctrl-C stops it. Pass any folder that
contains `*.ome.zarr` volumes, or a single volume.

Full resolution streams slowly over the share. For speed, run the same command on a NEMO node with
`--bind 0.0.0.0` (from `/camp/lab/windingm/home/shared/repos/em-pipe`; data paths start with `/camp/lab/windingm/...` there) and tunnel from the Mac with
`ssh -L 8000:<node>:8000 login.nemo.thecrick.org`; the link is the same.

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
