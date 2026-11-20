# 13 November 2026: matching 37i's contrast to 35h's

37i looked more contrasty than 35h with the same settings (tissue 1-99 % windows, CLAHE clip 1.5). These
scripts measured why, comparing neuropil with neuropil, and found settings that match. Outputs went to
the 37i output folder's `qc/clahe_match/`.

| Script | What it does |
|---|---|
| `overviews.py` | Coarse overviews of both volumes with a lettered grid, to pick neuropil cells by eye. |
| `clahe_match.sbatch` | Re-applies CLAHE offline (exactly as render does) to tile-aligned crops of the plain volumes and measures fine contrast (8 nm), contrast at 32 nm, pixel noise and grey levels, for a sweep of clip limits. |
| `neuropil_sheet.sbatch` | A contact sheet of every neuropil cell used (`neuropil_patches_*.png`). |
| `window_match.sbatch` | Simulates `intensity.window: median` on the rendered crops: which target median and span make 37i's neuropil match 35h's, plain and with CLAHE. |
| `window_pics.sbatch` | Side-by-side pictures of 35h and 37i neuropil, before and after the match. |
| `inline/*metrics_neuropil*` | Summaries of the neuropil metrics. |

Result: on neuropil, 37i had about a third more contrast than 35h and a darker median. A median window
(tissue median at grey 151, tissue p5-p95 over 123 greys, i.e. contrast x0.75) matched it, and with the
same CLAHE settings the CLAHE volumes then matched within a few percent. These are 37i's `intensity`
settings. The neuropil cells were picked by eye by the coding assistant, not by an expert; the
reference-tissue normalisation work replaces this hand matching with an automatic neuropil mask.
