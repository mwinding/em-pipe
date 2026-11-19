import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import sys

import numpy as np

from pipeline import patchgrid
from pipeline.config import load_config, deep_merge
from pipeline.intensity import solve, connected

cfg = load_config(os.path.expandvars("configs/lab/${DEV_PREFIX}37i.yaml"))
p = deep_merge(patchgrid.DEFAULTS, {"patchgrid": cfg.get("patchgrid") or {}})["patchgrid"]
shapes, R = patchgrid.load(cfg)
raw = patchgrid.read_bins(cfg)
z = 16530
b0 = raw[raw.z == z].copy()
b = patchgrid.shape_corrected(b0, shapes, R)
b["gw"] = patchgrid.bin_weights(b, p["gain_power"], p["gain_cnr_min"])[1]
print("gw stats: nan", int(b.gw.isna().sum()), "pos", int((b.gw > 0).sum()), "zero", int((b.gw == 0).sum()), file=sys.stderr)
print(b[["a", "b", "pa", "pb", "gain", "n", "cnr", "gw"]].assign(raw_gain=b0["gain"].values).to_string(), file=sys.stderr)
rels = [(r.a, r.b, r.gain, r.offset, r.n, r.gw, r.centre) for r in b.itertuples()]
print("solve all:", solve(connected(rels)), file=sys.stderr)
pos = [r for r in rels if r[5] > 0]
print("solve gw>0 only:", solve(connected(pos)), file=sys.stderr)
for (seg, t), (sg, sc) in sorted(shapes.items()):
    if seg in (11, 12):
        print(f"shape seg {seg} {t}: sg mean {np.mean(sg):.3f} [{np.min(sg):.3f}-{np.max(sg):.3f}], sc mean {np.mean(sc):.0f}",
              file=sys.stderr)
print("R", R, file=sys.stderr)
