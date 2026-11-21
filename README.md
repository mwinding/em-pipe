# em-pipe

Processing pipeline for tiled FIB-SEM acquisitions saved as one ImageJ TIFF stack per tile per day
(e.g. `M10_D06_tile1-1.tif`). It checks the raw data, stitches the tiles of each slice, corrects
slice-to-slice drift, normalises intensity and writes a multi-resolution volume for
[neuroglancer](https://github.com/google/neuroglancer). It runs as Slurm jobs on an HPC cluster.

```
check → preview → [refmask] → stitch → align → [shading] → [overlaps] → [patchgrid] → [refstats] → intensity → [zcorrect] → render → pyramid
```

A plain-language walk through the whole pipeline, with pictures: `docs/pipeline_explainer.html`.

| Step | What it does |
|---|---|
| check | Inventory of the raw files: truncated or unreadable files, copied tiles, label/filename mismatches, missing tiles, time gaps, restarts and grid changes. Writes the global slice list every later step uses. |
| preview | Per-slice intensity statistics and thumbnails (contact sheet per day). |
| refmask | Optional (for `intensity.window: reference`): where the reference tissue (neuropil, with tracts) is, as a yes/no per ~2 µm square of each thumbnail, from a small trained classifier (`qc/refmask.png`). Never applied to the image; it only chooses where refstats measures. See *Consistent contrast across volumes* below. |
| stitch | Tile layout within each slice: SIFT matches in the tile overlaps (overlap found automatically) and a least-squares solve, on every 50th slice. Where tiles move between two of those by more than `stitch.refine_px` (5 px), the slices in between are stitched too (bisection), so a sudden tile jump is placed at the right slice instead of being interpolated. |
| align | Slice-to-slice drift: SIFT matches between each slice and its next few neighbours, then one global least-squares solve. Where the tile layout changes (e.g. 3×3 → 2×2 tiles), one affine transform between the two montage frames is fitted first (`align.segment_change_model`; results in `work/align/segment_changes.csv`). Where that leaves a median mismatch above `align.elastic_min_px` (5 px), a smooth displacement field is fitted to the matches across the change and applied to every slice before the change (`align.elastic_side`; across other layout changes up to the previous corrected one), warping them into the later layout's geometry (`pipeline/elastic.py`; `work/align/elastic_interfaces.csv` reports the mismatch before and after); render cuts sharply between tiles in those slices (`render.local_blend`). A tile-level affine solve (`align.local_*`, off by default) is also available; it leaves out matches across elastic interfaces. A slice whose own montage is off right at an interface (e.g. the first after a long pause) can be left out of those fits (`align.interface_skip_slices`) and its tiles fitted with a one-slice local window; `align.local_warp_px` adds a smooth warp per tile there, for a distortion within a tile that no affine captures, `align.local_seams: matched` holds the seams matched in the image instead of where the stitch put them, and `align.local_rematch` matches the window's slices again block by block, so a distorted tile gets matches over all of it. |
| shading | Optional: a flat-field per tile position and acquisition segment (the SEM's brightness fall-off across a tile), estimated from the thumbnails as in BaSiC/CIDRE (`qc/shading.png`); overlaps, intensity and render divide it out of every tile slice. |
| overlaps | Optional: the brightness relation (gain, offset) of every overlapping tile pair, measured on the raw tiles at 32 nm in every slice by a robust fit on pixel pairs, tissue and resin alike (`qc/overlaps.png`). With `overlaps.patch_grid` > 0 also one relation per pair of tile patches (each tile cut into a grid of patches), for patchgrid. |
| patchgrid | Optional: smooth brightness maps within each tile for what the flat-field leaves near tile edges and corners (after Janelia's tile intensity correction, saalfeldlab/render): a gain and offset per patch, solved from the patch relations; their within-tile shape, the median over every 20th slice, per tile position and segment (`qc/patchgrid.png`). Intensity then balances the tiles on the patch relations with the shapes applied, and render applies flat-field and shape before each tile's window. |
| refstats | Optional (for `intensity.window: reference`): the reference tissue's histogram (median, 5th–95th percentiles) at every `refstats.every`-th slice, measured inside the refmask on the 8 nm tiles with every correction render applies (flat-field, patch shapes, tile balance, and decurtaining when destreak is on), before any contrast window (`qc/refstats.png`). |
| intensity | Per-tile contrast levels (the raw values render maps to 0 and 255), balanced across tile overlaps and smoothed over time. The balance comes from the preview thumbnails every few slices (`balance_source: thumbnails`) or from the overlaps step, slice by slice (`balance_source: overlaps`; with patchgrid, from the patch relations). The window: `median` (the tissue's median at a set grey), `percentile`, or `reference` (the reference tissue at a set grey and width, so volumes match; below). |
| zcorrect | Optional slice-thickness correction (off by default). |
| render | Applies everything in one pass and writes OME-Zarr (Zarr v3, sharded), optionally downsampled; CLAHE; optional destreaking (filter, or swap from a second detector). |
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
qc/                    check_report.md and plots: stitch, align_drift, intensity, preview_stats, daily sheets;
                       with window: reference also refmask, refstats and intensity_reference (.png, .json)
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

**Straight to a bucket (recommended for delivery):** `precomputed.enabled: true` and
`precomputed.destination: gs://bucket/prefix`, then `./run_pipeline.sh CONFIG --steps precomputed`. The export
is written into the bucket as it is made: no local copy (an export of a whole volume is terabytes: lossless
compression shrinks EM data little), no separate upload. `verify` reads samples back from the bucket and only then writes
`info` and `provenance`, so the volume only looks complete when it is. A re-run continues where it stopped (the
progress is recorded in `work/precomputed/<name>/`). A destination that already holds a volume is only replaced
with `init --overwrite`, in place, and only if it has the same size and export settings; writing to a new
prefix is cleaner when someone may be reading the old one.

**Local folder, then upload:** leave `precomputed.destination` null (the export goes to
`output_dir/<name>.precomputed`), and set `upload.enabled: true` with `upload.destination: gs://bucket/prefix`.
Then `./run_pipeline.sh CONFIG --steps precomputed,upload`. Check what an upload would send first:
`python -m pipeline.upload --config CONFIG run --dry-run`. Files already in the bucket are skipped only when
size and MD5 match.

Both use a Google service-account key file. **Put it in place yourself** on the cluster,
at `~/.cloudvolume/secrets/google-secret.json` (CloudVolume's default; or set `precomputed.credentials` /
`upload.credentials`), and `chmod 600` it. Never put a key in the repository, a config, a chat or an email:
both steps refuse a key file inside the repository and only ever pass its path on (to tensorstore as
`GOOGLE_APPLICATION_CREDENTIALS`, or to the Google client).

A local export serves like any volume: `python3 -m pipeline.serve FOLDER` lists `*.precomputed` too.

## Things to know

- **New data:** align's global solve changes earlier slices' transforms, so after adding days,
  re-create the volume: `sbatch slurm/render.sbatch CONFIG init --overwrite`, then
  `./run_pipeline.sh CONFIG --from render`.
- **Clock changes:** if the label clock is local time, set `check.timezone` to its zone (an IANA
  name such as `Europe/Paris`) before a daylight-saving change (the repeated hour would otherwise
  break the slice order). Re-run check after changing it; later steps refuse a mismatched timezone.
- **Diagnostic volumes during an acquisition:** with `render.append: true`, re-running the chain on
  new data adds the new slices to the existing volume instead of asking for `--overwrite`. Planes
  already rendered (whole slabs) are kept as they are and the new ones are placed in the volume's
  frame; pyramid redoes only what changed. Give the volume room for drift with
  `render.append_margin_px` when it is first created. Not for final volumes: those need one render
  from a single global alignment (`init --overwrite`).
- **Several volumes** (e.g. a 32 nm overview and a full-resolution `render.bbox` region) can share
  one `output_dir` if they have different `render.name`s.
- **destreak and zcorrect** are off by default; tune destreak with
  `python -m pipeline.destreak test --config C --z Z --tile 0-0` first. If the microscope recorded a
  second detector that doesn't see the curtains (e.g. backscatter next to in-lens SE, the same scan),
  `destreak.reference_from` / `reference_to` swap the stripe part from it instead of removing it, so
  structure that runs along the curtains is kept; `stripe_px` sets what counts as a stripe by length,
  `core_threshold` fills the cores of strong curtains from the matched reference, and `band_match`
  matches the broad bands that clusters of curtains darken to the reference, only around the curtains
  (`band_max_px` limits it to bands up to that wide, so wider regions the detectors show differently keep
  the image's look).
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

## tools/, dev/ and qc/

`tools/` holds checks and helpers that work on any dataset; usage is in each docstring:

| Tool | What it does |
|---|---|
| `raw_stats.py` (+ `slurm/raw_stats.sbatch`) | Raw detector values per tile at sampled slices, by class: tissue, resin, off-sample (and neuropil vs other tissue where the reference-tissue mask is available). Histograms, per-tile percentiles, plots; `compare` draws several datasets together. Resin comes from the background mask and is not validated as a reference yet. |
| `copy_verify.py` (+ `slurm/copy_verify.sbatch`) | Copy raw files to another place, md5 both sides, and write a checksum manifest next to the copy only when every file matches. Never deletes; refuses nested folders. |
| `tile_jumps.py` | Tile jumps and alignment problems over a whole run (plot + summary): candidates for `align.local_windows`. |
| `reference_from_volume.py`, `reference_preview.py`, `standard_compare.py`, `refstats_compare.py`, `segment_overview.py`, `refmask_check.py`, `refmask_train.py` | The reference contrast (below): a volume's neuropil numbers under its levels, before/after crops of new levels, volumes side by side at candidate standards, two refstats runs compared, one slice per segment with its tile outlines, the neuropil mask drawn over a volume, the mask model rebuilt from labelled boxes. |
| `seam_steps.py`, `shading_measure.py`, `shading_before_after.py`, `clipping_check.py`, `map_point.py`, `triptych.py` | Seam brightness steps, flat-field measurements and before/after stacks, clipping, mapping a point between volume and raw tiles, side-by-side figures. |
| `volume_compare.py` | Two or more rendered volumes at the same places, side by side with grey histograms: whole slices and full-resolution crops. Reads local OME-Zarr and precomputed volumes, including a delivered copy in a Google Cloud Storage bucket (credentials from the environment), e.g. a re-render against what was delivered. |
| `check_private.sh` | Run before every push: fails if the files or commits about to be pushed contain site-specific details (paths, hosts, names, dataset IDs). Generic patterns are in `tools/private_patterns.txt`; put your own (cluster, share, people) in the git-ignored `tools/private_patterns.local.txt`. |

```bash
# raw statistics of a dataset (8 slices; on the cluster, in parallel, then merge)
A=$(sbatch --parsable --array=0-7 slurm/raw_stats.sbatch configs/lab/my_volume.yaml --num-tasks 8)
sbatch --dependency=afterok:$A slurm/raw_stats.sbatch configs/lab/my_volume.yaml merge
python tools/raw_stats.py compare --out cmp A=/path/to/a/qc/raw_stats B=/path/to/b/qc/raw_stats

# copy raw data and verify it (a Slurm chain: copy array, md5 array, verify)
python tools/copy_verify.py submit /path/to/raw/my_volume /path/to/copy/my_volume --group-rw
```

Keep the record of which data were copied where (and other site details) in your own configuration
repository, not here.

`dev/` holds development analyses: the checks and experiments behind the pipeline's design and the
datasets' settings, kept so the reasoning can be traced. They are not pipeline code and not tested;
see [dev/README.md](dev/README.md). `qc/` in a clone is git-ignored: a place to copy a run's `qc/`
plots and comparison stacks for looking at locally.

## Consistent contrast across volumes (`intensity.window: reference`)

Volumes stained or imaged differently come out of the median or percentile windows looking different,
because those windows follow all the tissue in a slice, whatever its mix. The reference window instead
puts one tissue, neuropil, at a fixed standard in every volume: its median at grey `reference.median`
and the width of its histogram between the 5th and 95th percentile at `reference.contrast` greys (e.g.
150 and 130). Which percentiles the window clips at is not chosen; it follows from matching the
neuropil, and the QC reports it. The standard starts from a reference volume
(`tools/reference_from_volume.py`: its neuropil numbers under the levels it was rendered with), is
settled once (`tools/standard_compare.py` shows volumes side by side at candidate widths, with the
tissue each clips and the pixel noise against the rounding to whole greys, before and after CLAHE) and
then kept fixed.

Three steps:

1. **refmask** marks the neuropil per ~2 µm square of every preview thumbnail (`pipeline/refmask.py`,
   below).
2. **refstats** measures the neuropil's histogram inside that mask at every `refstats.every`-th slice
   of each segment, on the common scale the tiles are balanced to, with every correction render
   applies (decurtaining too, when `destreak.enabled`), before any window. `refstats.region` limits it
   to part of the volume; `refstats.output` keeps a variant's statistics apart (e.g. with and without
   decurtaining; `tools/refstats_compare.py` compares two).
3. **intensity** sets each slice's window:
   - its **width** is 255 × the neuropil's p5–p95 width / `reference.contrast`;
   - its **position** follows every slice's tissue, as the median window places it, so fast changes
     from slice to slice are followed exactly; it is shifted so the neuropil's median comes out at
     `reference.median`;
   - the width and that shift follow the neuropil **slowly** along z (`reference.per: slice`, the
     default: a running median of the sampled slices over `reference.smooth_slices`, 2001), restarting
     at every segment change (a change of tile layout) and at every z in `reference.breaks` (e.g. a
     microscope setting changed mid-run). `per: segment` keeps one width and one shift per segment
     instead.

The QC (`qc/intensity_reference[_<output>].png`, `.json`, `_samples.csv`) predicts the neuropil's
grey and width at every sampled slice, the tissue clipped at 0 or 255, and the step across every
segment change, next to the same for the previous levels. To put a volume on the standard, add to a
variant config (`base: my_volume.yaml`):

```yaml
refmask: {enabled: true}
refstats: {enabled: true, every: 64}
intensity:
  window: reference
  output: reference                  # own levels, next to the volume's existing ones
  reference: {median: 150, contrast: 130}
render:
  name: my_volume_reference.ome.zarr
```

then `./run_pipeline.sh CONFIG --steps refmask,refstats,intensity`, look at the QC, and
`--steps render,pyramid`. `tools/reference_preview.py` draws before/after crops of the new levels
without rendering (`levels` mode maps the existing volume through the old and new windows exactly).

### The neuropil mask

`pipeline/refmask.py` marks a reference tissue (neuropil, with tracts) on the preview thumbnails, per
~2 µm cell, so that intensity statistics can be measured in the same tissue in every volume. It is a
small trained classifier (`pipeline/refmask_neuropil.json`, gradient-boosted trees in
`pipeline/boost.py`) on texture features that ignore gain, offset and rotation;
`tools/refmask_check.py` draws it over a volume and measures neuropil statistics
inside it, and `tools/refmask_train.py` rebuilds the model from labelled boxes
(`pipeline/refmask_neuropil_labels.json`). `refmask_train.py propose` draws new label boxes from the
current classifier's own mask (a box and a 10 µm margin all inside the mask, or all in tissue
without it). That is self-referential: the labels are only validated by a person checking the
pictures (`refmask_train.py draw --outline`, every box with an ID), box by box; the label file
records what was confirmed and what was rejected. "Other" boxes must be cell bodies only, never gut,
oesophagus or trachea: `propose` can't tell those apart (it only keeps new boxes off regions
rejected before), so the person checking has to reject them.
