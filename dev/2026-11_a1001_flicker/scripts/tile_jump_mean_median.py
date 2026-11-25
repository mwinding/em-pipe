"""Is the flicker left after the display windows the difference between the mean and the median of the
tiles' jumps?

    python dev/2026-11_a1001_flicker/scripts/tile_jump_mean_median.py ZFLICKER_CSV [OUT_PNG]

ZFLICKER_CSV: tools/zflicker.py output made with --config (series "raw (fixed window)" and "raw - windows").
The tile balance puts each slice's tiles on a common scale with mean offset 0 (intensity.solve), so the
scale follows the MEAN of the tiles' jumps; intensity.per_slice: slice shifts every window by the MEDIAN of
the tiles' jumps. If that is where the left-over flicker comes from, the per-slice residual of "raw -
windows" should track mean - median of the raw tiles' jumps. Prints their correlation, the regression
slope and both robust SDs; draws the two against z and against each other.
"""
import sys

import numpy as np
import pandas as pd


def robust_sd(a):
    a = np.asarray(a, float)
    return 1.4826 * np.median(np.abs(a - np.median(a)))


def main():
    df = pd.read_csv(sys.argv[1], dtype={"box": str})
    raw = df[df["series"] == "raw (fixed window)"]
    left = df[df["series"] == "raw - windows"].groupby("z")["resid"].median()
    by = raw.groupby("z")["resid"]
    d = (by.mean() - by.median()).reindex(left.index)
    ok = d.notna() & left.notna()
    r = float(np.corrcoef(d[ok], left[ok])[0, 1])
    slope = float(np.polyfit(d[ok], left[ok], 1)[0])
    print(f"slices {int(ok.sum())}; corr(left-over common, mean - median of tile jumps) = {r:.3f}; slope {slope:.2f}")
    print(f"robust SD: left-over common {robust_sd(left[ok]):.2f} greys, mean - median {robust_sd(d[ok]):.2f} greys")
    # Other candidates: the raw tiles' common jump (under- or over-corrected) and the windows' own common shift
    for name, s in (("raw common (median over tiles)", by.median()),
                     ("raw common (mean over tiles)", by.mean()),
                     ("windows common", df[df["series"] == "windows (correction)"].groupby("z")["resid"].median())):
        s = s.reindex(left.index)
        k = s.notna() & left.notna()
        print(f"corr(left-over common, {name}) = {np.corrcoef(s[k], left[k])[0, 1]:.3f}; "
              f"slope {np.polyfit(s[k], left[k], 1)[0]:.2f}")
    # intensity._tissue_brightness takes the slice's common window as the MEDIAN over tiles of the balanced
    # starting windows (smooth lo plus the slice's balance offset, which carries minus the tile's jump), i.e.
    # one tile's; every tile's window then carries that tile's jump. Candidate: the tiles' median jump minus
    # the jump of the tile whose (smooth level - jump) is the median that slice.
    w = df[df["series"] == "windows (correction)"].pivot(index="z", columns="box", values="value")
    j = raw.pivot(index="z", columns="box", values="resid").reindex(index=w.index, columns=w.columns)
    trend = w.rolling(51, center=True, min_periods=1).median()
    key = (trend - j).to_numpy()
    mid = np.argsort(key, axis=1)[:, key.shape[1] // 2]
    cand = pd.Series(np.median(j.to_numpy(), axis=1) - j.to_numpy()[np.arange(len(mid)), mid], index=w.index)
    k = cand.notna() & left.reindex(cand.index).notna()
    lc = left.reindex(cand.index)[k]
    print(f"corr(left-over common, median jump - middle tile's jump) = {np.corrcoef(cand[k], lc)[0, 1]:.3f}; "
          f"slope {np.polyfit(cand[k], lc, 1)[0]:.2f}; its robust SD {robust_sd(cand[k]):.2f}")
    if len(sys.argv) > 2:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a, b) = plt.subplots(1, 2, figsize=(15, 4), gridspec_kw={"width_ratios": [3, 1]}, layout="constrained")
        a.plot(left.index, left.to_numpy(), color="black", lw=1, label="left after the windows (common)")
        a.plot(d.index, d.to_numpy(), color="tab:orange", lw=1, label="mean - median of the tiles' jumps")
        a.set_xlabel("z")
        a.set_ylabel("greys")
        a.legend(loc="upper right")
        b.scatter(d[ok], left[ok], s=6, color="0.3")
        b.set_xlabel("mean - median of the tiles' jumps (greys)")
        b.set_ylabel("left after the windows (greys)")
        b.set_title(f"r = {r:.2f}, slope {slope:.2f}")
        fig.savefig(sys.argv[2], dpi=110)


if __name__ == "__main__":
    main()
