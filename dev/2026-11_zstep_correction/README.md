# Does slice-thickness (z-step) correction matter for our volumes? (25-26 Nov 2026)

FIB-SEM slices are not all equally thick, so a volume stacked at a fixed z-step can draw structures
out or squash them along z. The pipeline's `zcorrect` step (after Hanslovsky et al. 2017) estimates
each slice's position from how alike neighbouring slices are, and render then resamples planes at
even spacing. It is off in every dataset config. This folder measures what it estimates on A1001
(487 slices, 3x3 tiles) and 37i (17,927 slices, 2x2 tiles), checks how far the estimate can be
trusted, and compares test slabs rendered with and without it.

**Result: on these two datasets the per-slice estimate is mostly not slice thickness, and applying it
changes shapes very little while blending almost every plane from two slices. The real thickness
errors found earlier (the first cuts after long pauses) are at breaks, where the step keeps the
nominal spacing by design.**

## Scripts

| Script | What |
|---|---|
| `scripts/tile_ncc.py` | The zcorrect step's slice-to-slice NCC, kept per tile (the step keeps the median over tiles), so the spacing can be solved from each tile alone. |
| `scripts/step_stats.py` | Distribution of the estimated spacing, near-duplicate and thick steps, the most uneven stretches, the cuts across breaks, the agreement between tiles, and a model with one factor per slice next to the spacing. |
| `scripts/quality_sim.py` | Simulated slice pairs: what the solve makes of slices that differ only in image quality, not in thickness. |
| `scripts/pick_box.py` | A square of tissue in a rendered volume, as a `render.bbox` for a test slab. |
| `scripts/shape_measure.py` | Per plane, the correlation length of fine structure along z against x and y (anisotropy), for slabs with and without the correction; in-plane detail by how much a plane is a blend. |
| `scripts/shape_sim.py` | Checks that the anisotropy reads a known stretch on a synthetic volume (slices 0.8x as thick: 1.22, expected 1.25; 1.25x: 0.84, expected 0.80). |

Outputs are under each dataset's `qc/zstep_correction/` (statistics, per-tile NCC, and one folder
per test slab with the x-y, x-z and y-z figures, a neuroglancer link and `shape.json`). The step
itself wrote `work/zcorrect/` and `qc/zcorrect.png`, from a test config that builds on the dataset's
config with `zcorrect.enabled: true`; the datasets' own configs keep it off.

## What was known before

`dev/2026-11_37i_processing` (z-step profiles of the rendered 37i volume, 14 Nov 2026): the cut
across the 38 h pause is about 4.6 slices thick and the slices after it are thicker than nominal for
about 80 slices (2x at first, 1.1x by 80 slices on: about 35 slices, 0.28 um, more depth than the
volume shows); the cut across a 2.2 h pause is 3-4 slices thick; short pauses have no thick cut; and
after each restart slices come out about 10 % thinner and drift back over 1000-2000 slices. No
correction was applied; the thick slices are recorded in the dataset config.

## The estimated spacing

Whole stacks, default settings (`max_distance` 8, 1024 px crop at each tile's centre, mean spacing
held at nominal over a rolling 200 slices and per block between breaks).

| | A1001 | 37i |
|---|---|---|
| slices, breaks | 487, 0 | 17,927, 21 |
| NCC of neighbouring slices (median; p5-p95) | 0.73; 0.69-0.76 | 0.82; 0.74-0.89 |
| estimated step, median | 7.90 nm | 7.95 nm |
| p5-p95 | 5.4-10.7 nm | 5.3-11.2 nm |
| SD | 1.75 nm (22 %) | 2.00 nm (25 %) |
| steps within 10 % / 20 % of nominal | 43 % / 69 % | 38 % / 65 % |
| steps below 0.5x (near-duplicates) | 1 (at the 0.25x limit, in the first 33 slices) | 85 (35 at the 0.25x limit) |
| steps above 1.5x / 2x | 11 / 1 | 539 / 81 |
| largest step | 21.9 nm (last 35 slices of the stack) | 78.7 nm (z 2083 to 2084, see below) |
| a slice's distance from its uncorrected place (median / p95 / max) | 21 / 52 / 60 nm | 15 / 67 / 253 nm |
| most uneven 64-slice stretches (SD of step / nominal) | z 453-486: 0.36; z 49-113: 0.25 | z 2026-2090: 1.14; z 17602-17666: 0.59; z 3165-3229: 0.54 |
| typical 64-slice stretch | 0.20 | 0.21 |

Taken at face value this says a quarter of the steps are more than 20 % off. The checks below say
most of that is not thickness.

## How far the estimate can be trusted

1. **Independent areas of the same slices do not agree.** A thickness change is shared by the whole
   slice. Solving the spacing from each tile's crop alone and correlating the per-slice results
   between tiles gives a median r of 0.07 (A1001, 9 tiles) and 0.09 (37i, 4 tiles); averaging over 9
   slices does not help (0.05, 0.06). Two halves of the tiles agree at r 0.30 (A1001) and 0.09 (37i)
   and differ by 1.7 / 1.6 nm (median) per step. The part the two halves share has an SD of 1.15 nm
   (A1001) and 0.59 nm (37i): an upper limit on real per-slice thickness variation of about 14 % and
   7 % of the step, and the shared part is not necessarily thickness (next point).
2. **A low NCC is mostly a slice that looks different, not a thick cut.** zcorrect reads every NCC as
   a distance. A slice with more noise, a focus or charge change, or scan-line streaks correlates
   less with all its neighbours on both sides, whereas a thick cut lowers only the pairs that cross
   it. A model with one factor per slice (at nominal spacing) explains 62 % (A1001) and 93 % (37i) of
   the NCC's variance about its per-lag median; the spacing zcorrect solved explains 28 % and 2 %.
   Hanslovsky et al. estimate a scaling factor per section together with the positions; em-pipe's
   simplified step has no such factor. (Janelia's render client sets that factor's regulariser to
   1.0 for FIB-SEM, with the comment "won't correct for noisy slices", and masks resin by default.)
3. **The similarity curve is shallow, so small NCC changes become large steps.** NCC falls from 1 to
   0.73-0.82 between neighbours (pixel noise) and then by only 0.03-0.05 per further slice: a
   slice pair whose NCC is that much low reads as one whole extra slice. In `quality_sim.py`, slices of
   exactly equal thickness whose image quality varies by 3 % (SD of the per-slice factor; measured:
   2.2 % on A1001, 7.0 % on 37i) come out with a spacing SD of 0.19 x nominal, the same size as the
   measured 0.22-0.25.
4. **The largest steps are bad slices.** The 78.7 nm step is between z 2083 and 2084 in 37i: z 2083
   is the first slice after the 38 h pause, scanned before the beam settled (stretched and shifted
   tile; recorded in the dataset config). The real thick cut there is one slice earlier, between
   z 2082 and 2083, which is a break (the field of view changes), and zcorrect holds breaks at
   nominal. In the x-z views of z 3165-3229 (looked at, not measured) the slab has slices with
   scan-line streaks, and structures continue smoothly through the stretch with the 17-28 nm steps.
5. **Breaks are left out by design.** Pairs across seams and segment changes are not used and the
   spacing there is nominal. Read against the 60 slices before each break, the cut across the 2.2 h
   pause (z 17333) is like 5.0 slices (earlier profile: 3-4) and the two restarts just after it like
   2.4-2.7; the short pauses are like 1.0-1.4 slices. A control that is no pause at all (the clock
   change, z 5782) reads as 2.0 slices on an NCC of 0.78 against 0.80-0.82 at the short pauses, which
   shows the same sensitivity as point 3. Segment changes (including z 2083) cannot be measured by the step
   at all.
6. **Slow drift is removed by construction.** Holding the rolling 200-slice mean at nominal means
   the ~10 % thinner slices after each restart are neither measured nor corrected. A similarity
   curve cannot tell slow thickness drift from a slow change in tissue texture, so this is the
   safer choice, but it means the step only addresses slice-to-slice unevenness.

## What it does to the images

Four test slabs of 128 slices, a 2560 px square of tissue each, rendered with the delivery
settings (decurtained) twice: `zcorrect.enabled` false and true, nothing else changed. Figures per
slab in `qc/zstep_correction/<slab>/`: `ortho_*_{xy,xz,yz}.png` (uncorrected above z-corrected,
1 px per voxel), `neuroglancer_link.txt` (layers "uncorrected" and "z-corrected", 4 panels) and
`shape.json`.

Shape measure (`shape_measure.py`): per plane, the correlation length of fine structure (band-pass
2-6 voxels) along z over that along x and y. 1 = as deep as wide; uneven spacing shows as a value
that changes from plane to plane (its SD along z). If the correction removed real unevenness, that
SD would fall.

| Slab | Why chosen | Anisotropy, uncorrected: median (SD along z; range) | Anisotropy, z-corrected | Planes: slices to planes |
|---|---|---|---|---|
| 37i z 10309-10436 | typical stretch | 1.01 (0.017; 0.97-1.03) | 1.03 (0.076; 0.92-1.20) | 128 to 128 |
| 37i z 3133-3260 | third most uneven estimate (single steps of 17-28 nm) | 1.17 (0.114; 0.68-1.27) | 1.16 (0.182; 0.82-1.78) | 128 to 128 |
| 37i z 2019-2146 | most uneven estimate; the 38 h pause at z 2083 | 0.82 (0.195; 0.46-1.05) | 0.91 (0.317; 0.51-2.21) | 128 to 144 |
| A1001 z 17-144 | second most uneven estimate of A1001 | 1.15 (0.064; 1.08-1.35) | 1.29 (0.138; 0.94-1.47) | 128 to 131 |

- **In the typical stretch there is nothing to correct.** Uncorrected, the anisotropy is 1.01 and
  varies by 1.7 % from plane to plane, while zcorrect's estimate for the same slices has an SD of
  21 % of the step. (The length is read over about 5 planes; 21 % independent step errors would
  still leave about 9 %.) After the correction the spread is 4.5 times larger.
- **In every slab the spread along z is larger after the correction** (1.6 to 4.5 times), and in
  A1001 the median moves away from 1. Part of this is the blending itself: a plane that shares a
  source slice with its neighbour correlates more with it, which lengthens the z-length of some
  planes and not others. The measure cannot separate that from a geometric change; either way no
  slab becomes more even.
- **The estimate does not predict the stretch seen in the images** where slices are ordinary: the
  correlation between the uncorrected planes' anisotropy and the estimated local spacing is -0.14
  (typical slab) and -0.09 (A1001). It is 0.59 and 0.38 in the two slabs with bad slices, where a
  slice that looks different lowers both measures.
- **The pause is the one place with a real, visible z-error.** In the uncorrected x-z and y-z views
  structures break off at z 2083 and the planes after it are squashed along z (anisotropy down to
  0.46; the thick slices measured earlier). The correction adds 16 planes to the slab, 9 of them
  in one band between z 2083 and 2084: those planes are blends of the distorted slice 2083 and
  slice 2084 (vertical smears in x-z, anisotropy up to 2.2), not recovered tissue, and the break in
  the structures one slice earlier stays. The median anisotropy moves from 0.82 to 0.91.
- In the other three slabs the x-z and y-z views with and without the correction show the same
  shapes; the corrected views are slightly smoother (looked at, not measured).

## Costs and conflicts

- **Almost every plane becomes a blend of two slices.** render places planes at even spacing and
  interpolates linearly between the two slices either side (as Janelia's render does when exporting
  with a z-coordinate file). In the four corrected slabs 1-2 of 128-144 planes come from a single
  slice; the mean largest weight is 0.72-0.76. Blending averages pixel noise and softens fine
  detail, and it does so unevenly from plane to plane (fine detail, which includes pixel
  noise, is 4-13 % lower in planes that are an even mix than in planes that are almost one slice,
  and it follows the blend weight with r 0.28-0.70; the slab medians are 5-9 % lower than
  uncorrected; `crop_512/` in two slab folders shows a single slice above an even mix of that slice
  and the next).
- **Planes are no longer slices.** The plane count changes (a 128-slice stretch became 128-144
  planes), so a z in the volume is not an acquisition slice any more: slice-based records (the
  config notes, QC by slice, the per-slice levels) and anything already annotated or segmented on
  the delivered volumes would not line up with a corrected re-render.
- **Run time is small.** Measuring: 37 task-minutes for A1001 and 500 task-minutes for 37i (30
  tasks, the longest 30 min; 2 CPUs each); the solve takes under a minute. Rendering the corrected
  slabs took about as long as the uncorrected ones (a blended plane reuses the slice it shares with
  the next plane).
- **`render.append`** refuses to run with z-correction (planes of an appended volume must not
  change; corrected positions change when slices are added).
- **Decurtaining and `intensity.balance_z_distance`** work per slice, before the blend, so there is
  no conflict in the code. Two interactions: curtains and streaks lower a slice's NCC and so move its
  estimated position (point 2), and blending averages the slice-to-slice brightness of two slices
  in some planes and not in others, which re-shapes whatever flicker is left.

## Recommendation

Keep `zcorrect.enabled: false` for deliveries of these datasets.

- Measured: the per-slice estimate is not reproducible between areas of the same slices, is mostly
  explained by per-slice image differences, and its largest steps are known bad slices; applying it
  does not make shapes measurably more even, and it blends nearly every plane.
- Measured earlier and confirmed here: the real, large thickness errors are the first cuts after
  long pauses. They are single places, already recorded in the dataset config, and the step does not
  correct them (they are breaks).
- Judged: Janelia runs this step on stacks where slice thickness is known to wander, with the
  per-slice scaling factor and resin masking that em-pipe's version lacks. If a later dataset shows
  visible stretching in x-z (not just a pause), the step would need those two things first, and
  `step_stats.py` gives the test: tiles must agree on the spacing before it is applied.

## Sources read

- Hanslovsky P, Bogovic JA, Saalfeld S. Image-based correction of continuous and discontinuous
  non-planar axial distortion in serial section microscopy. Bioinformatics 33(9):1379, 2017 (the
  article page: the similarity-versus-distance assumption, the per-section scaling factors, and its FIB-SEM
  example with estimated spacing from 0.14 to 10.2 px).
- saalfeldlab/render, `render-ws-java-client/.../zspacing/`: `ZPositionCorrectionClient.java`
  (cross-correlation in batches then a solve, resin masking on by default, an option that sets the
  first and last layers' corrections to 1), `HeadlessZPositionCorrection.java` (default FIB-SEM
  options: comparison range 10, 100 iterations, scaling-factor regulariser 1.0), and
  `ThicknessCorrectionData.java` with the n5 export client's `--z_coords` option (planes
  interpolated linearly between the two layers either side of each corrected z).
