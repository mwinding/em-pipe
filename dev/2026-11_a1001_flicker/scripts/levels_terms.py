"""Which term of the display windows carries the slice-to-slice flicker left after them?

    python dev/2026-11_a1001_flicker/scripts/levels_terms.py CONFIG [OUT_CSV]

Recomputes the run's levels with em-pipe's own functions, keeping the parts apart (as
intensity.compute_levels combines them for window: reference, brightness: tissue):

  lo = (L - o) / g + jump
  L     the slice's common-scale window, from M (the median over tiles of the balanced starting windows,
        intensity.common_window) shifted by a slowly following amount (reference tissue)
  g, o  the tile's balance gain and offset that slice (overlaps)
  jump  the per-slice jump (intensity.per_slice)

and the raw tile's tissue median t_p50. All in greys of the tile's window (255 / (hi - lo)), minus their
running mean over 9 slices; then per slice the median over tiles ("common"). Prints each term's common
robust SD and its correlation with the left-over (t_p50 - lo, the brightness the render shows), and
writes the per-slice commons to OUT_CSV.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pipeline import intensity  # noqa: E402
from pipeline.config import load_config, step_defaults  # noqa: E402


def robust_sd(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return 1.4826 * np.median(np.abs(a - np.median(a)))


def main():
    cfg = load_config(sys.argv[1], step_defaults=step_defaults("intensity"))
    c = cfg["intensity"]
    df = intensity.balanced_scale(cfg)
    if c["per_slice"] == "slice":
        rel = df["jump"] / (df["hi"] - df["lo"])
        df["jump"] = rel.groupby(df["z"]).transform("median") * (df["hi"] - df["lo"])
    w0 = intensity.common_window(df)
    tm = float(c["target_median"])
    M = (w0["Lc"] + tm / 255.0 * (w0["Hc"] - w0["Lc"]))
    df = intensity.reference_windows(cfg, df)
    g, o = df["g"].fillna(1.0), df["o"].fillna(0.0)
    lo = (df["L"] - o) / g + (df["jump"] if c["per_slice"] else 0.0)
    hi = (df["H"] - o) / g + (df["jump"] if c["per_slice"] else 0.0)
    k = 255.0 / (hi - lo)
    terms = pd.DataFrame({"z": df["z"], "tile": df["tile"],
                          "t_p50": df["t_p50"] * k,
                          "L": df["L"] / g * k,
                          "M": df["z"].map(M) / g * k,
                          "L - M": (df["L"] - df["z"].map(M)) / g * k,
                          "-o/g": -o / g * k,
                          "jump": (df["jump"] if c["per_slice"] else 0.0) * k})
    terms["lo"] = terms["L"] + terms["-o/g"] + terms["jump"]
    terms["left (t_p50 - lo)"] = terms["t_p50"] - terms["lo"]
    names = [n for n in terms.columns if n not in ("z", "tile")]
    terms = terms.sort_values(["tile", "z"])
    hp = terms[names] - terms.groupby("tile")[names].transform(
        lambda s: s.rolling(9, center=True, min_periods=1).mean())
    hp[["z", "tile"]] = terms[["z", "tile"]]
    common = hp.groupby("z")[names].median()
    own = pd.DataFrame(hp[names].to_numpy() - common.reindex(hp["z"]).to_numpy(), columns=names)
    left = common["left (t_p50 - lo)"]
    for n in names:
        r = np.corrcoef(common[n], left)[0, 1] if np.std(common[n]) > 0 else float("nan")
        print(f"{n:>18}: common {robust_sd(common[n]):6.2f}, own {robust_sd(own[n]):6.2f} greys; "
              f"corr with left-over common {r:+.2f}")
    if len(sys.argv) > 2:
        common.to_csv(sys.argv[2], float_format="%.4f")


if __name__ == "__main__":
    main()
