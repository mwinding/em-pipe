# 12-14 November 2026: processing the 37i volume

Checks made while taking 37i (about 18,000 slices: a single-tile approach, then 2x2 tiles) through the
pipeline. Outputs went to the 37i output folder's `qc/` (e.g. `qc/zstep_profile/`) or to scratch.

| Scripts | Question | What came of it |
|---|---|---|
| `dupcheck_37i.py`, `dupcheck2_37i.py`, `inline/*s583*` | Are re-exported raw files complete, and is any tile a copy of another? | A tile file that was a byte copy of its neighbour and a slice with an empty label were found and re-exported; check now flags both (`DUPLICATE_CONTENT`, label checks). |
| `seg12.py`, `seg12_median.py`, `spikes.py` | Why does stitching fail in one late segment? | The tissue splits in two and the seam between tile columns runs through resin; that segment uses one fixed layout from the samples that could be matched (`stitch.segment_modes`). |
| `bins_seg12.py`, `bins_dbg.py`, `logsolve.py` | Why are some tile gains off where tiles only overlap in resin? | Tile balance now solves gains as logs, so tiles linked only by resin stay at gain 1. |
| `z2083.py`, `z2083_bands.py`, `junction2083.py`, `window2083.py`, `jump_check.sbatch`, `inline/*local_warps*`, `inline/*baseline*` | Why is the first slice after a long pause misplaced? | Its first tile was scanned before the beam settled (stretched and shifted). It is left out of the frame and elastic fits (`align.interface_skip_slices`) and fitted on its own with a smooth per-tile warp, seams matched in the image and matches redone block by block (`align.local_warp_px`, `local_seams`, `local_rematch`). |
| `restart_firsts.py` | Are first slices after restarts generally off? | Only that one needed special treatment. |
| `zgap.sbatch`, `zstep_sessions.sbatch`, `zstep_profile.sbatch`, `zstep_profile2.sbatch`, `zstep_merge.py`, `zstep_2083_fine.sbatch`, related `inline/` files | How thick are the slices, especially across pauses? | The cut across the long pause is ~4-5 slices thick and the following slices thicker than nominal for ~80 slices; after each restart slices come out ~10 % thinner and drift back. Recorded in the dataset config; no correction applied. |
| `inline/*tile-matching-fixed*` | Is a left/right brightness difference real once tile matching is fixed? | Checked in slices before the tissue split. |
