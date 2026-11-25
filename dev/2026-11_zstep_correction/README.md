# Does slice-thickness (z-step) correction matter for our volumes? (25 Nov 2026)

Measurements of the slice spacing along z with the pipeline's `zcorrect` step (after Hanslovsky et al.
2017), on A1001 and 37i, and test slabs rendered with and without the correction. Findings are added
below as the runs finish.

## Scripts

| Script | What |
|---|---|
| `scripts/tile_ncc.py` | The zcorrect step's slice-to-slice NCC, kept per tile (the step keeps the median over tiles), so the spacing can be solved from each tile alone. |
| `scripts/step_stats.py` | Distribution of the estimated spacing, near-duplicate and thick steps, the most uneven stretches, the cuts across breaks, and the agreement between tiles. |
