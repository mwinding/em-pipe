# em-pipe

Processing pipeline for tiled FIB-SEM acquisitions saved as one ImageJ TIFF stack per tile per day
(e.g. `M10_D06_tile1-1.tif`). It checks the raw data, stitches the tiles of each slice, corrects
slice-to-slice drift, normalises intensity and writes a multi-resolution volume for
[neuroglancer](https://github.com/google/neuroglancer). It runs as Slurm jobs on an HPC cluster.

```
check → preview → stitch → align → [shading] → [overlaps] → [patchgrid] → intensity → [zcorrect] → render → pyramid
```

| Step | What it does |
|---|---|
| check | Inventory of the raw files: truncated or unreadable files, copied tiles, label/filename mismatches, missing tiles, time gaps, restarts and grid changes. Writes the global slice list every later step uses. |
| preview | Per-slice intensity statistics and thumbnails (contact sheet per day). |
| stitch | Tile layout within each slice: SIFT matches in the tile overlaps (overlap found automatically) and a least-squares solve, on every 50th slice. Where tiles move between two of those by more than `stitch.refine_px` (5 px), the slices in between are stitched too (bisection), so a sudden tile jump is placed at the right slice instead of being interpolated. |
| align | Slice-to-slice drift: SIFT matches between each slice and its next few neighbours, then one global least-squares solve. Where the tile layout changes (e.g. 3×3 → 2×2 tiles), one affine transform between the two montage frames is fitted first (`align.segment_change_model`; results in `work/align/segment_changes.csv`). Where that leaves a median mismatch above `align.elastic_min_px` (5 px), a smooth displacement field is fitted to the matches across the change and applied to every slice before the change (`align.elastic_side`; across other layout changes up to the previous corrected one), warping them into the later layout's geometry (`pipeline/elastic.py`; `work/align/elastic_interfaces.csv` reports the mismatch before and after); render cuts sharply between tiles in those slices (`render.local_blend`). A tile-level affine solve (`align.local_*`, off by default) is also available. |
| shading | Optional: a flat-field per tile position and acquisition segment (the SEM's brightness fall-off across a tile), estimated from the thumbnails as in BaSiC/CIDRE (`qc/shading.png`); overlaps, intensity and render divide it out of every tile slice. |
| overlaps | Optional: the brightness relation (gain, offset) of every overlapping tile pair, measured on the raw tiles at 32 nm in every slice by a robust fit on pixel pairs, tissue and resin alike (`qc/overlaps.png`). With `overlaps.patch_grid` > 0 also one relation per pair of tile patches (each tile cut into a grid of patches), for patchgrid. |
| patchgrid | Optional: smooth brightness maps within each tile for what the flat-field leaves near tile edges and corners (after Janelia's tile intensity correction, saalfeldlab/render): a gain and offset per patch, solved from the patch relations; their within-tile shape, the median over every 20th slice, per tile position and segment (`qc/patchgrid.png`). Intensity then balances the tiles on the patch relations with the shapes applied, and render applies flat-field and shape before each tile's window. |
| intensity | Per-tile contrast levels, balanced across tile overlaps and smoothed over time. The balance comes from the preview thumbnails every few slices (`balance_source: thumbnails`) or from the overlaps step, slice by slice (`balance_source: overlaps`; with patchgrid, from the patch relations). |
| zcorrect | Optional slice-thickness correction (off by default). |
| render | Applies everything in one pass and writes OME-Zarr (Zarr v3, sharded), optionally downsampled; CLAHE; optional destreaking. |
| pyramid | Lower-resolution scales of the volume. |
| export | Quick-look TIFF of the volume for Fiji (finest scale under `export.max_gb`). |
| precomputed | Optional: the volume as Neuroglancer precomputed (all scales, sharded), e.g. for segmentation with CloudVolume-based tools. |
| upload | Optional: copies the precomputed export to a Google Cloud Storage bucket (resumable, checksummed). |

## Setup (once per cluster)

```bash
conda env create -f environment.yml -p /path/to/conda-envs/em-pipe
cp slurm/site.example.sh slurm/site.sh    # then fill it in
```

`slurm/site.sh` holds what is specific to your cluster: the conda env's path (`EM_PIPE_ENV`), any
modules to load first, the FFmpeg module for movies and the Slurm partition. `slurm/env.sh` reads it
in every job. It is git-ignored, so each clone has its own; a group can keep one copy somewhere shared
and symlink it into each clone. Its `SBATCH_PARTITION` applies to the jobs `run_pipeline.sh` submits;
a direct `sbatch slurm/<step>.sbatch` needs `--partition` unless the cluster's default partition will do.

## Running

Each dataset (volume) has a YAML config with `raw_dir`, `output_dir`, an optional `selection` (date
or z range) and per-step settings. Start from `configs/example.yaml`, which lists every setting with
its default. Dataset configs hold paths on your storage, so keep them outside the repository:
`configs/` ignores everything but `example.yaml`, and a symlink such as `configs/lab` pointing at
your group's config folder keeps the commands short.

```bash
./run_pipeline.sh configs/lab/my_volume.yaml --dry-run       # show the job chain
./run_pipeline.sh configs/lab/my_volume.yaml                 # submit it
squeue --me                                                  # follow; logs in <output_dir>/work/logs/
./run_pipeline.sh configs/lab/my_volume.yaml --from align    # re-run from a step
```

Every job waits for the previous one (`afterok`), so a failed step cancels the rest. Steps skip
work whose output already exists, so re-submitting resumes. Array sizes, CPUs, memory, time and
partition come from the config's `slurm:` section. Jobs read their config when they start, so
editing a config changes jobs that are still queued.

A variant of a volume (another render, a movie, an upload) is a small config with
`base: my_volume.yaml` and only the settings that differ.

Any step can also run directly, e.g. on a laptop against the mounted storage:

```bash
python -m pipeline.check --config configs/lab/my_volume.yaml \
    --raw-dir /path/to/raw/stacks --output-dir ~/em-test
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

## Viewing a volume

`pipeline.serve` needs only Python 3's standard library (a Mac's built-in `python3` is fine) and
Chrome, so a clone on a network share lets a whole group view volumes without installing anything:

```bash
cd /path/to/em-pipe
python3 -B -m pipeline.serve /path/to/output_dir
```

(`-B` stops Python writing cache files into a shared copy.)

It prints a neuroglancer link with every volume in that folder as a layer (click a layer's name
to show or hide it). Keep the terminal open while viewing; Ctrl-C stops it. Pass any folder that
contains `*.ome.zarr` volumes, or a single volume.

Full resolution streams slowly over a network share. For speed, run the same command on a cluster
node with `--bind 0.0.0.0` and tunnel from your computer with
`ssh -L 8000:<node>:8000 <login-host>`; the link is the same.

## Neuroglancer precomputed and bucket upload

In the config: `precomputed.enabled: true` (export after the pyramid), and for the upload
`upload.enabled: true` with `upload.destination: gs://bucket/prefix`. Then
`./run_pipeline.sh CONFIG --steps precomputed,upload` (or as part of a longer chain). Check what an
upload would send first: `python -m pipeline.upload --config CONFIG run --dry-run`.

The upload uses a Google service-account key file. **Put it in place yourself** on the cluster,
at `~/.cloudvolume/secrets/google-secret.json` (CloudVolume's default; or set `upload.credentials`),
and `chmod 600` it. Never
put a key in the repository, a config, a chat or an email: the upload step refuses a key file inside
the repository and only ever passes its path to the Google client.

The export serves like any volume: `python3 -m pipeline.serve FOLDER` lists `*.precomputed` too.

## Things to know

- **New data:** align's global solve changes earlier slices' transforms, so after adding days,
  re-create the volume: `sbatch slurm/render.sbatch CONFIG init --overwrite`, then
  `./run_pipeline.sh CONFIG --from render`.
- **Clock changes:** if the label clock is local time, set `check.timezone` to its zone (an IANA
  name such as `Europe/Paris`) before a daylight-saving change (the repeated hour would otherwise
  break the slice order). Re-run check after changing it; later steps refuse a mismatched timezone.
- **Several volumes** (e.g. a 32 nm overview and a full-resolution `render.bbox` region) can share
  one `output_dir` if they have different `render.name`s.
- **destreak and zcorrect** are off by default; tune destreak with
  `python -m pipeline.destreak test --config C --z Z --tile 0-0` first.
- **patchgrid** needs `overlaps.patch_grid` > 0 (e.g. 8) and `intensity.balance_source: overlaps`.
  Setting or changing `overlaps.patch_grid` re-measures every overlap chunk; new shapes change the
  levels and the render plan (render init then asks for `--overwrite`, and render run refuses to go on). With patchgrid off and
  `patch_grid: 0`, every output is as before the step existed.
- Design notes and file formats: `docs/design.md`. Tests: `python -m pytest tests`.

## Fly-through movie

`pipeline/movie.py` films a rendered volume: the whole slice, then a smooth zoom into a target point
down to full resolution, scrolling through z the whole time (settings: `movie:` in the config, see
`configs/example.yaml`). Pick the target in neuroglancer (x, y, z as it shows them), then on the
cluster

```bash
sbatch --output=LOGFILE slurm/movie.sbatch CONFIG          # frames, then output_dir/<name>.mp4
python -m pipeline.movie --config CONFIG frames --frames 0,300,599   # a few frames to check first
```

Each frame reads only what is on screen from the matching pyramid level, so a short movie reads a
few GB whatever the volume's size. Frames go to `work/movie/<name>/`; a re-run keeps them unless the
timeline changed.

## tools/ and qc/

`tools/` holds one-off analysis scripts that read a run's working files (e.g. `shading_measure.py`,
`shading_before_after.py`; usage in each docstring). `qc/` in a clone is git-ignored: a place to
copy a run's `qc/` plots and comparison stacks for looking at locally.
