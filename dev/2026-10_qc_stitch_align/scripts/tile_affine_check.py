"""How far per-tile affines could reduce the mismatch at a tile layout change.

Matches across the change (existing align chunks) are mapped through stitch and the solved align
transforms into the aligned frame; then the mismatch is measured as is, and after fitting a
translation or affine per new tile, and an affine per (old tile, new tile) pair.
"""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import glob
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expandvars("${EM_PIPE_ROOT}"))
from pipeline import align as al, transforms  # noqa: E402
from pipeline.config import load_config  # noqa: E402

cfg = load_config(sys.argv[1], step_defaults=al.DEFAULTS)
z_last_old = int(sys.argv[2])   # last slice of the old layout
df = al.load_slices(cfg)
st = al._stitch_transforms(cfg, df)
T = pd.read_csv(al.step_path(cfg, "align", "transforms.csv"))
T = dict(zip(T["z"], T[transforms.COLUMNS].to_numpy().reshape(-1, 2, 3)))

P = {k: [] for k in ("pa", "pb", "ta", "tb", "za", "zb")}
for f in sorted(glob.glob(str(al.step_path(cfg, "align", "matches")) + "/*.npz")):
    d = np.load(f)
    x = (d["z_a"] <= z_last_old) & (d["z_b"] > z_last_old)
    if not x.any():
        continue
    tiles = d["tiles"]
    pa = al._to_montage(st, d["z_a"][x], d["ta"][x], tiles, d["qa"][x])
    pb = al._to_montage(st, d["z_b"][x], d["tb"][x], tiles, d["qb"][x])
    P["pa"].append(np.array([transforms.apply(T[z], p)[0] for z, p in zip(d["z_a"][x], pa)]))
    P["pb"].append(np.array([transforms.apply(T[z], p)[0] for z, p in zip(d["z_b"][x], pb)]))
    for k, v in (("ta", tiles[d["ta"][x]]), ("tb", tiles[d["tb"][x]]), ("za", d["z_a"][x]), ("zb", d["z_b"][x])):
        P[k].append(v)
P = {k: np.concatenate(v) for k, v in P.items()}
pa, pb = P["pa"], P["pb"]
print(f"{len(pa)} matched points across z {z_last_old} -> {z_last_old + 1}, "
      f"slice pairs {sorted(set(zip(P['za'].tolist(), P['zb'].tolist())))[:3]}...")


def fit(groups, model):
    r = np.empty(len(pa))
    for g in np.unique(groups):
        s = groups == g
        if model == "translation":
            r[s] = np.linalg.norm(pa[s] - pb[s] - (pa[s] - pb[s]).mean(0), axis=1)
        else:
            X = np.column_stack([pb[s], np.ones(s.sum())])
            A, *_ = np.linalg.lstsq(X, pa[s], rcond=None)
            r[s] = np.linalg.norm(pa[s] - X @ A, axis=1)
    return r


def show(name, r):
    print(f"  {name:48s} median {np.median(r):5.1f} px   p90 {np.percentile(r, 90):5.1f} px   max {r.max():5.1f} px")


show("as now (one affine for the whole field)", np.linalg.norm(pa - pb, axis=1))
show("+ a translation per new tile", fit(P["tb"], "translation"))
show("+ an affine per new tile", fit(P["tb"], "affine"))
pair = np.char.add(np.char.add(P["ta"].astype(str), ">"), P["tb"].astype(str))
show("+ an affine per (old tile, new tile) pair", fit(pair, "affine"))
print("per new tile, mismatch as now (median px) and mean offset (x, y):")
for tb in np.unique(P["tb"]):
    s = P["tb"] == tb
    m = (pa[s] - pb[s]).mean(0)
    print(f"  {tb}: {np.median(np.linalg.norm(pa[s] - pb[s], axis=1)):5.1f} px, offset ({m[0]:6.1f}, {m[1]:6.1f}), {s.sum()} points")
