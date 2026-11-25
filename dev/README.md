# dev/: development analyses (not pipeline code)

One-off checks, experiments and measurements made while building and tuning em-pipe on real data:
why a setting was chosen, what a fault turned out to be, how a fix was tested. They are kept so the
reasoning behind the pipeline's design and the datasets' configs can be traced and re-run. They are
**not** part of the pipeline: they are not tested, not maintained against later changes, and may need
adapting before they run again. General-purpose checks live in `tools/` instead.

| Folder | When | What |
|---|---|---|
| [2026-10_qc_stitch_align/](2026-10_qc_stitch_align/) | Oct 2026 | stitching and alignment checks (solvers, layout changes, tile-level windows, seams), display-window and normalisation options, background mask, flat-field tests, CLAHE variants, spot-check stacks, a real-size synthetic dataset |
| [2026-11_37i_processing/](2026-11_37i_processing/) | 12-14 Nov 2026 | checks made while processing the 37i volume: duplicate/label checks of re-exported files, a segment whose tile columns could not be matched, the junction after a long pause, slice-thickness (z-step) profiles |
| [2026-11_contrast_matching/](2026-11_contrast_matching/) | 13 Nov 2026 | matching 37i's plain and CLAHE contrast to 35h's on neuropil (the basis of 37i's `intensity.window: median` settings) |
| [2026-11_a1001_flicker/](2026-11_a1001_flicker/) | 24 Nov 2026 | where A1001's slice-to-slice flicker comes from: raw tile jumps, and what the display windows leave of them |
| [_assembly/](_assembly/) | 19 Nov 2026 | the scripts used to put this folder together (see below) |

Curtain and EsB work (destreak tests, the curtain "arches") is in a separate repository,
[curtain-analysis](https://github.com/mwinding/curtain-analysis).

## Running a script

Each folder has `scripts/` (files that existed as scripts) and `inline/` (code that was only ever run
inline, as a heredoc in a shell command, saved here as files with a header giving the time, the note
written just before it, and whether it ran locally or on the cluster).

Site-specific paths are placeholders, resolved from the environment: copy
[paths.example.sh](paths.example.sh) to `paths.local.sh` (git-ignored), fill it in and `source` it.
Python wraps them in `os.path.expandvars(...)`; shell uses them as `${VAR}`.

Outputs: scripts that take a dataset config or folder write under that folder's `qc/` (as the
tools do), or to `$DEV_OUT` where they once wrote to a fixed location. Nothing here deletes files.
`masked_stats.py` is the one script that writes a pipeline file (`<test>/work/preview/stats.csv`,
to build a test copy); it refuses to run when the test folder is the main one.

## How this folder was assembled

The original files were mostly gone (session scratch folders are cleared), so the code was rebuilt from
the coding-assistant session records:

1. Scripts that had been saved as files were rebuilt from the recorded file writes and edits.
2. Inline scripts were extracted from the recorded shell commands, then sorted with
   [_assembly/classify_inline.py](_assembly/classify_inline.py). Left out were: programmatic edits of
   this repository's own files (their result is in the git history), movie work, anything that
   deleted files or touched credentials, pure job submission/monitoring, and trivial status checks.
   Shell quoting left by remote commands was undone with
   [_assembly/fix_ssh_quoting.py](_assembly/fix_ssh_quoting.py).
3. Site-specific details (paths, hosts, names, dataset IDs) were replaced by placeholders with the
   sanitiser from curtain-analysis (`tools/sanitize.py` there, with private rules kept outside the
   repository), and [_assembly/patch_dev.py](_assembly/patch_dev.py) made the few changes listed in it.

Known differences from what ran:
- `2026-11_37i_processing/scripts/dupcheck_37i.py`: one later edit could not be replayed.
- `2026-11_37i_processing/scripts/zstep_merge.py`: two later edits could not be replayed.
- `2026-11_contrast_matching/scripts/clahe_match.sbatch`: patched once more by
  `inline/20261113-2158_thanks-what-assumed-m-writing.py` (kept alongside, not applied).
- Not recovered: scripts that existed only in a cluster scratch folder and were never written through
  the session (`setup_tissue1-99.sh`, `submit_overview.sh`, `spot_tiffs.py`, `mask_preview_v2.py`,
  `mask_preview_v3.py`, `mask_sweep_v3.py`), and a few later versions of recovered scripts made there.
- Left out on purpose: a clean-up script that deleted an old export, and a re-run script that
  re-created volumes with `--overwrite`; a draft of the shading step (it became `pipeline/shading.py`).

New analysis code goes straight into a repository (this folder, `tools/`, or a project repository),
never only into a scratch folder or an inline command.
