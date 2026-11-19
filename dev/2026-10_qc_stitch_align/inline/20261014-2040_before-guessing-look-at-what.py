# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-14T20:40:01.980Z
# Ran: remote, via ssh
# Heredoc line: ssh -o BatchMode=yes -o ConnectTimeout=20 -o LogLevel=QUIET ${CLUSTER_LOGIN} 'source ${EM_PIPE_ROOT}/slurm/env.sh >/dev/null 2>&1; cd ${DEV_LAB_AREA}
# Assistant note before it: Before guessing, I'll look at what the alignment and stitching actually did at the 3×3 → 2×2 change. This only reads the run's output files.

import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import numpy as np, glob
from pipeline.config import load_config
from pipeline import align as al
cfg = load_config(os.path.expandvars("configs/${DEV_PROJECT}_test_3x3.yaml"), step_defaults=al.DEFAULTS)
df = al.load_slices(cfg); st = al._stitch_transforms(cfg, df)
P = {k: [] for k in ("pa", "pb", "ta", "tb", "qa", "qb")}
for f in sorted(glob.glob(str(al.step_path(cfg, "align", "matches")) + "/*.npz")):
    d = np.load(f); x = (d["z_a"] <= 3278) & (d["z_b"] >= 3279)
    if not x.any(): continue
    tiles = d["tiles"]
    P["pa"].append(al._to_montage(st, d["z_a"][x], d["ta"][x], tiles, d["qa"][x])); P["pb"].append(al._to_montage(st, d["z_b"][x], d["tb"][x], tiles, d["qb"][x]))
    P["ta"].append(tiles[d["ta"][x]]); P["tb"].append(tiles[d["tb"][x]]); P["qa"].append(d["qa"][x]); P["qb"].append(d["qb"][x])
P = {k: np.concatenate(v) for k, v in P.items()}
pa, pb = P["pa"], P["pb"]
X = np.column_stack([pb, np.ones(len(pb))]); A, *_ = np.linalg.lstsq(X, pa, rcond=None); r = pa - X @ A
print("3x3 tile -> 2x2 tile: points, mean residual (x, y) px after one affine, spread")
for ta, tb in sorted(set(zip(P["ta"], P["tb"]))):
    s = (P["ta"] == ta) & (P["tb"] == tb)
    print(f"  {ta} -> {tb}: {s.sum():5d}  mean ({r[s,0].mean():6.1f}, {r[s,1].mean():6.1f})  sd ({r[s,0].std():4.1f}, {r[s,1].std():4.1f})")
# within one tile pair, does x residual vary with x position in the tile (scan nonlinearity)?
for ta, tb in [("1-1", "0-0"), ("1-1", "1-1")]:
    s = (P["ta"] == ta) & (P["tb"] == tb)
    if s.sum() < 50: continue
    qx = P["qb"][s][:, 0]; bins = np.linspace(0, 13875, 8)
    print(ta, tb, "x residual by x position in the 2x2 tile:", [round(float(r[s][(qx >= lo) & (qx < hi), 0].mean()), 1) if ((qx >= lo) & (qx < hi)).any() else None for lo, hi in zip(bins[:-1], bins[1:])])
