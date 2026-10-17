"""Does the tile brightness relation change across the width of an overlap (tile-edge effects)?

    python tools/experiments/janelia_across.py CONFIG LEVELS_CSV Z[,Z...] [--band 64]

For each pair of side-by-side tiles of a slice: the overlap's pixel pairs (as janelia_bins.py), split
into bands by distance from the seam line (the middle of the overlap, where the cut render switches
tiles), and per band one relation va ~ gain vb + offset; printed is the step a - b at the whole
overlap's centre value in display greys of tile a's window in LEVELS_CSV (the current per-tile
windows), so a constant across bands means the overlap average is what the seam shows.
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import overlaps, shading, transforms  # noqa: E402
from pipeline.config import deep_merge, load_config  # noqa: E402
from pipeline.slices import StackCache, load_slices  # noqa: E402
import janelia_bins as jb  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("levels")
    p.add_argument("zs")
    p.add_argument("--band", type=float, default=64)
    args = p.parse_args()
    cfg = load_config(args.config, step_defaults=deep_merge(overlaps.DEFAULTS, shading.DEFAULTS))
    o = cfg["overlaps"]
    slices = load_slices(cfg)
    stitch = transforms.read_csv(Path(cfg["output_dir"]) / "work" / "stitch" / "tiles.csv", ["z", "tile"])
    fields = shading.load(cfg)
    cache = StackCache(cfg["raw_dir"])
    lv = pd.read_csv(args.levels, dtype={"tile": str}).set_index(["z", "tile"])
    rows = []
    for z in [int(v) for v in args.zs.split(",")]:
        recs = slices[slices["z"] == z].to_dict("records")
        for r in recs:
            r["key"] = (int(r["segment"]), r["tile"])
        for ra, rb in itertools.combinations(sorted(recs, key=lambda r: r["tile"]), 2):
            (ar, ac), (br, bc) = ra["tile"].split("-"), rb["tile"].split("-")
            if ar != br and ac != bc:
                continue   # corner contact
            got = jb.pair_values_pos(cache, ra, rb, stitch[(z, ra["tile"])], stitch[(z, rb["tile"])], fields, o)
            if got is None:
                continue
            va, vb, xa, ya, xb, yb = got
            vertical = ar != br     # a above b: across = y
            # signed distance from the seam line (the overlap's middle), positive towards b's side
            pa_full = ya if vertical else xa
            across = pa_full - 0.5 * (pa_full.min() + pa_full.max())
            whole = overlaps.fit_relation(va, vb)
            if whole is None:
                continue
            m = float(np.median(vb))
            lo, hi = lv.loc[(z, ra["tile"]), ["lo", "hi"]]
            span = hi - lo
            for b0 in np.arange(np.floor(across.min() / args.band) * args.band, across.max(), args.band):
                sel = (across >= b0) & (across < b0 + args.band)
                if sel.sum() < 500:
                    continue
                rel = overlaps.fit_relation(va[sel], vb[sel])
                if rel is None:
                    continue
                g, off = rel[:2]
                rows.append({"z": z, "seam": f"{ra['tile']}|{rb['tile']}", "band": b0 + args.band / 2,
                             "n": int(sel.sum()), "step_ab": (g * m + off - m) / span * 255})
    df = pd.DataFrame(rows)
    wide = df.pivot_table(index=["z", "seam"], columns="band", values="step_ab")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(wide.round(1).to_string())


if __name__ == "__main__":
    main()
