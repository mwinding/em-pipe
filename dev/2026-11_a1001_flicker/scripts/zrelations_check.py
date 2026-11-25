"""How the zrelations step's slice-to-slice relations compare: on the corrected values u (flat-field and patch
shape, what intensity uses) against the same on the raw thumbnails, and against the change of the tile's
tissue median in the preview statistics (raw, each slice's own tissue mask).

    python dev/2026-11_a1001_flicker/scripts/zrelations_check.py CONFIG

Per relation of distance 1 (tile t, z -> z + 1): its change at its centre as a fraction of the centre (u and raw),
and the preview t_p50's change as a fraction of t_p50; all turned into greys of the tile's display window at z
(the config's levels: 255 / (hi - lo), times the u centre). Prints the robust SD of each, of their differences,
and their correlations; per tile too.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pipeline import intensity  # noqa: E402
from pipeline.config import load_config, step_defaults, step_path  # noqa: E402


def robust_sd(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(1.4826 * np.median(np.abs(a - np.median(a)))) if a.size else float("nan")


def main():
    cfg = load_config(sys.argv[1], step_defaults=step_defaults("intensity"))
    rel = pd.read_csv(step_path(cfg, "zrelations", "relations.csv"), dtype={"tile": str})
    rel = rel[rel["distance"] == 1]
    stats = pd.read_csv(step_path(cfg, "preview", "stats.csv"), dtype={"tile": str})[["z", "tile", "t_p50"]]
    lv = pd.read_csv(intensity.levels_path(cfg), dtype={"tile": str})
    d = rel.merge(lv.rename(columns={"z": "z1"}), on=["z1", "tile"])
    d = d.merge(stats.rename(columns={"z": "z1", "t_p50": "t1"}), on=["z1", "tile"])
    d = d.merge(stats.rename(columns={"z": "z2", "t_p50": "t2"}), on=["z2", "tile"])
    k = 255.0 / (d["hi"] - d["lo"]) * d["centre"]          # greys per unit fraction of the tile's level
    d["u"] = ((d["gain"] - 1) * d["centre"] + d["offset"]) / d["centre"] * k
    d["raw"] = ((d["gain_raw"] - 1) * d["centre_raw"] + d["offset_raw"]) / d["centre_raw"] * k
    d["t_p50"] = (d["t2"] - d["t1"]) / d["t1"] * k
    print(f"{len(d)} relations (distance 1), {d['tile'].nunique()} tiles, levels {intensity.levels_path(cfg)}")
    print("slice-to-slice change at the centre, greys of the window (robust SD):")
    for c in ("u", "raw", "t_p50"):
        print(f"  {c:>6}: {robust_sd(d[c]):6.2f}")
    for a, b in (("u", "raw"), ("u", "t_p50"), ("raw", "t_p50")):
        print(f"  {a} - {b}: {robust_sd(d[a] - d[b]):6.3f} greys (robust SD), median {np.median(d[a] - d[b]):+.3f}; "
              f"r {np.corrcoef(d[a], d[b])[0, 1]:.4f}")
    print("per tile, robust SD of u - raw and of u - t_p50 (greys); median tissue pixels:")
    for t, g in d.groupby("tile"):
        print(f"  {t}: {robust_sd(g['u'] - g['raw']):.3f}  {robust_sd(g['u'] - g['t_p50']):.3f}  n {int(g['n'].median())}")


if __name__ == "__main__":
    main()
