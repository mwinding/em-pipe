"""Mismatch across a layout change after the tile-level solve, over ALL matched points (none
dropped as outliers), per (old tile, new tile) and per slice pair."""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expandvars("${EM_PIPE_ROOT}"))
from pipeline import align as al, transforms  # noqa: E402
from pipeline.config import load_config  # noqa: E402

cfg = load_config(sys.argv[1], step_defaults=al.DEFAULTS)
z_old = int(sys.argv[2])
df = al.load_slices(cfg)
zs = np.array(al.z_values(df))
m = al._tile_matches(cfg, df, zs, cfg["align"], z_old - 8, z_old + 8)
local = transforms.read_csv(al.step_path(cfg, "align", "local_tiles.csv"), ["z", "tile"])
stitch = al._stitch_transforms(cfg, df)
T = pd.read_csv(al.step_path(cfg, "align", "transforms.csv"))
T = dict(zip(T["z"], T[transforms.COLUMNS].to_numpy().reshape(-1, 2, 3)))


def place(z, t, q, use_local=True):
    A = local[(z, t)] if use_local and (z, t) in local else transforms.compose(T[z], stitch[(z, t)])
    return transforms.apply(A, q)[0]


rows = []
for za, ta, qa, zb, tb, qb in zip(m["za"], m["ta"], m["qa"], m["zb"], m["tb"], m["qb"]):
    if not (za <= z_old < zb):
        continue
    after = np.linalg.norm(place(za, ta, qa) - place(zb, tb, qb))
    before = np.linalg.norm(place(za, ta, qa, False) - place(zb, tb, qb, False))
    rows.append((za, zb, ta, tb, before, after))
d = pd.DataFrame(rows, columns=["za", "zb", "ta", "tb", "before", "after"])
q = lambda s: f"median {s.median():5.1f}  p90 {s.quantile(0.9):5.1f}  max {s.max():5.1f}"
print(f"{len(d)} points across z {z_old} -> {z_old + 1} (none excluded)")
print(f"  before: {q(d.before)}")
print(f"  after:  {q(d.after)}")
print("adjacent pair only (z_old, z_old + 1):")
a = d[(d.za == z_old) & (d.zb == z_old + 1)]
print(f"  before: {q(a.before)}\n  after:  {q(a.after)}")
print("per (old tile -> new tile), after:  median / p90 / points")
g = d.groupby(["ta", "tb"])["after"]
print(pd.DataFrame({"median": g.median().round(1), "p90": g.quantile(0.9).round(1), "n": g.size()}).to_string())
