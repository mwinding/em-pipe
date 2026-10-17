"""Parameter sweep of the Janelia-style solve on the predicted seam steps (no rendering).

    python tools/experiments/janelia_tune.py BINS_DIR TILE_WINDOWS LEVELS SLICES ZS [--shape-zs ZS]

For each setting: the predicted step at every bin relation's centre after the maps (display greys,
a minus b), summarised like the seam metric (per seam the median over its bins, then the median |.|
and the count > 10 over seams, corner contacts left out), next to the same for the current
pipeline's per-tile windows (levels.csv) and the largest within-tile range of the maps.
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import janelia_solve as js  # noqa: E402


def interior_range(maps, hw, v):
    out = []
    for t, (g, c) in maps.items():
        h, w = hw[t]
        xs, ys = np.meshgrid(np.linspace(0, w - 1, 17), np.linspace(0, h - 1, 17))
        val = js.interp_grid(g, xs, ys, w, h) * (v - js.R) + js.interp_grid(c, xs, ys, w, h)
        out.append(np.ptp(val))
    return max(out)


def summarise(res):
    res = res.copy()
    ra, rb = res["a"].str.split("-", expand=True), res["b"].str.split("-", expand=True)
    res = res[(ra[0] == rb[0]) | (ra[1] == rb[1])]
    per = res.groupby(["z", "a", "b"])[[c for c in ("before", "after") if c in res]].median()
    seg = pd.cut(per.index.get_level_values("z"), [0, 828, 3279, 12529, 13771, 15378, 1e9], right=False)
    out = {f"{c}_med": per[c].abs().median() for c in per}
    out.update({f"{c}_n10": int((per[c].abs() > 10).sum()) for c in per})
    out.update({f"{c}_n30": int((per[c].abs() > 30).sum()) for c in per})
    out["seams"] = len(per)
    out["seg_after"] = ",".join(f"{v:.1f}" for v in per["after"].abs().groupby(seg, observed=False).median())
    if "before" in per:
        out["seg_before"] = ",".join(f"{v:.1f}" for v in per["before"].abs().groupby(seg, observed=False).median())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bins")
    p.add_argument("windows")
    p.add_argument("levels")
    p.add_argument("slices")
    p.add_argument("zs")
    p.add_argument("--shape-zs")
    p.add_argument("--grid", default="8")
    p.add_argument("--smooth", default="0.1,0.3,1")
    p.add_argument("--smooth-c", default="")
    p.add_argument("--prior", default="0.02")
    p.add_argument("--gain-power", default="1,2")
    p.add_argument("--patch-prior", default="0")
    p.add_argument("--variant", default="A")
    args = p.parse_args()
    zs = js.read_zs(args.zs)
    wins = pd.read_csv(args.windows, dtype={"tile": str})
    sizes = pd.read_csv(args.slices, dtype={"tile": str}, usecols=["z", "tile", "height", "width"])
    sizes = sizes[sizes["z"].isin(zs)]
    levels = pd.read_csv(args.levels, dtype={"tile": str})
    levels = levels[levels["z"].isin(zs)]
    fl = lambda s: [float(v) for v in s.split(",") if v != ""]   # noqa: E731
    rows = []
    for grid, sm, smc, pr, gp, pp, var in itertools.product(fl(args.grid), fl(args.smooth), fl(args.smooth_c) or [None],
                                                            fl(args.prior), fl(args.gain_power), fl(args.patch_prior),
                                                            args.variant.split(",")):
        ns = argparse.Namespace(bins=args.bins, grid=int(grid), smooth=sm, smooth_c=smc, prior=pr, gain_power=gp,
                                patch_prior=pp, prior_c=0.0, variant=var, shape_zs=args.shape_zs)
        maps_by_z = js.solve_variant(ns, zs, wins)
        res, rng = [], []
        for z, maps in maps_by_z.items():
            C = js.common_window(maps, wins[wins["z"] == z])
            hw = {r.tile: (r.height, r.width) for r in sizes[sizes["z"] == z].itertuples()}
            lz = {r.tile: (r.lo, r.hi) for r in levels[levels["z"] == z].itertuples()}
            res.append(js.predicted_steps(js.load_bins(args.bins, z), maps, C, hw, lz))
            rng.append(interior_range(maps, hw, 0.5 * (C[0] + C[1])) / (C[1] - C[0]) * 255)
        res = pd.concat(res)
        out = {"variant": var, "grid": int(grid), "smooth": sm, "smooth_c": smc, "prior": pr, "gain_power": gp,
               "patch_prior": pp, **summarise(res), "range_med": float(np.median(rng)), "range_max": float(np.max(rng))}
        rows.append(out)
        print(pd.DataFrame([out]).to_string(index=False, float_format="%.2f"), flush=True)
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.2f"))


if __name__ == "__main__":
    main()
