import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import sys

import numpy as np
import pandas as pd

from pipeline import patchgrid
from pipeline.config import load_config, deep_merge
from pipeline.intensity import solve, connected

cfg = load_config(os.path.expandvars("configs/lab/${DEV_PREFIX}37i.yaml"))
p = deep_merge(patchgrid.DEFAULTS, {"patchgrid": cfg.get("patchgrid") or {}})["patchgrid"]
shapes, R = patchgrid.load(cfg)
raw = patchgrid.read_bins(cfg)
print("bin columns:", list(raw.columns), file=sys.stderr)
for z in (9600, 14000, 15100, 16000, 16530, 17500):
    b = patchgrid.shape_corrected(raw[raw.z == z].copy(), shapes, R)
    b["gw"] = patchgrid.bin_weights(b, p["gain_power"], p["gain_cnr_min"])[1]
    out = []
    for (a, bb), g in b.groupby(["a", "b"]):
        used = g[g.gw > 0]
        wmed = (np.average(used.gain, weights=used.gw * np.sqrt(used.n)) if len(used) else float("nan"))
        out.append(f"{a}-{bb}: {len(used)}/{len(g)} bins used, gain {wmed:.3f}")
    rels = [(r.a, r.b, r.gain, r.offset, r.n, r.gw, r.centre) for r in b.itertuples()]
    full = solve(connected(rels))
    within = [r for r in rels if (r[0][-1] == r[1][-1])]   # same column only (x-y: tile row-col, col = last char)
    print(f"z {z}: " + " | ".join(out), file=sys.stderr)
    print("    solved gains: " + ", ".join(f"{t} {g:.3f}" for t, (g, o) in full.items()), file=sys.stderr)
    if z in (16530, 17500):
        cross = b[(b.a.str[-1] != b.b.str[-1]) & (b.gw > 0)]
        print(cross[["a", "b", "gain", "offset", "n", "cnr", "centre", "gw"] + [c for c in ("pa", "pb", "patch_a", "patch_b") if c in b]]
              .sort_values("gw", ascending=False).head(12).to_string(), file=sys.stderr)
