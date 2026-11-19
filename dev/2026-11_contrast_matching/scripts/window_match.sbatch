#!/bin/bash
#SBATCH --job-name=em-window-match
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
set -eo pipefail
cd ${EM_PIPE_ROOT}
source slurm/env.sh
python -u - <<'PY'
"""Would intensity.window: median (tissue median -> target_median grey, tissue p5-p95 over target_span
greys) make 37i's plain neuropil match 35h's, and CLAHE 1.5 on it match 35h's CLAHE 1.5?
Simulated on the rendered 37i plain crops: per slice, u' = tm + ts / S * (u - M), with M and S the slice's
tissue median and p5-p95 span in the current greys (preview tissue percentiles through the current levels)."""
import json
import math
import os

import cv2
import numpy as np
import pandas as pd
import tensorstore as ts
from scipy import ndimage

S35 = os.path.expandvars("${DEV_35H}")
S37 = os.path.expandvars("${DEV_37I}")
OUT_ROOT = __import__('os').environ.get('DEV_OUT', 'dev_out')  # outputs: a directory you choose, never the dataset folder
OUT = f"{OUT_ROOT}/qc/clahe_match"
TILE, K, CELL = 256, 10, 2560


def tissue_greys(work):
    """Per z: tissue p5, p50, p95 in display greys (median over tiles) and the span p95 - p5."""
    lv = pd.read_csv(f"{work}/intensity/levels.csv", dtype={"tile": str})
    st = pd.read_csv(f"{work}/preview/stats.csv", dtype={"tile": str}, usecols=["z", "tile", "t_p5", "t_p50", "t_p95"])
    d = st.merge(lv, on=["z", "tile"])
    for q in ("t_p5", "t_p50", "t_p95"):
        d[q] = 255 * (d[q] - d.lo) / (d.hi - d.lo)
    g = d.groupby("z")[["t_p5", "t_p50", "t_p95"]].median()
    g["span"] = g.t_p95 - g.t_p5
    return g


g35, g37 = tissue_greys(f"{S35}/work"), tissue_greys(f"{S37}/work")
for name, g in (("35h", g35), ("37i", g37)):
    q = g.quantile([0.1, 0.5, 0.9])
    print(f"{name} whole-slice tissue in its plain volume: median grey p10/50/90 "
          f"{q.t_p50.round(1).tolist()}, p5-p95 span {q.span.round(1).tolist()}", flush=True)


def opn(root, s):
    return ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": f"{root}/s{s}"}}, read=True, open=True).result()


def grid(n):
    g = math.ceil(n / TILE)
    return g, (n if n % g == 0 else n + g - n % g) // g


def metrics(x):
    x = x.astype(np.float32)
    m = ndimage.uniform_filter(x, 32)
    lc8 = np.sqrt(np.maximum(ndimage.uniform_filter(x * x, 32) - m * m, 0)).mean()
    h, w = (s // 4 * 4 for s in x.shape)
    y = x[:h, :w].reshape(h // 4, 4, w // 4, 4).mean(axis=(1, 3))
    m = ndimage.uniform_filter(y, 8)
    lc32 = np.sqrt(np.maximum(ndimage.uniform_filter(y * y, 8) - m * m, 0)).mean()
    p5, p50, p95 = np.percentile(x, [5, 50, 95])
    return {"lc8": float(lc8), "lc32": float(lc32), "std": float(x.std()), "p5": float(p5), "p50": float(p50), "p95": float(p95)}


CELLS = {(os.path.expandvars(f"{S37}/${{DEV_PREFIX}}37i_test_z1000_8nm.ome.zarr"), 1000): ["B2", "B4", "C1", "C2", "C4"],
         (os.path.expandvars(f"{S37}/${{DEV_PREFIX}}37i_test_z9600_8nm.ome.zarr"), 9600): ["C1", "C2", "D1", "D2", "E2", "D7", "D8", "E8", "H4", "I4"]}
med35 = float(g35.t_p50.median())
span35 = float(g35.span.median())
# settings to try: (label, target_median, target_span as a function of the slice) -- tm/ts are what the config would hold
A = [0.65, 0.7, 0.75, 0.8, 0.85]
rows = []
for (v, z0), cells in CELLS.items():
    P = opn(v, 0)
    H, W = P.shape[1:]
    (gx, tx), (gy, ty) = grid(W), grid(H)
    for plane in (16, 48):
        z = z0 + plane
        M, S = float(g37.at[z, "t_p50"]), float(g37.at[z, "span"])
        for cell in cells:
            i, j = ord(cell[0]) - ord("A"), int(cell[1:])
            yu, xu = H - 1 - (i + 0.5) * CELL, W - 1 - (j + 0.5) * CELL
            iy = min(max(0, round(yu / ty - K / 2)), gy - K - 1)
            ix = min(max(0, round(xu / tx - K / 2)), gx - K - 1)
            ys, xs = slice(H - (iy + K) * ty, H - iy * ty), slice(W - (ix + K) * tx, W - ix * tx)
            u = np.asarray(P[plane, ys, xs].read().result())[::-1, ::-1].astype(np.float32)
            inner = (slice(ty, -ty), slice(tx, -tx))
            rec = {"z": z, "cell": cell, "M": M, "S": S}
            variants = {"current": u}
            # 35h's whole-slice tissue mapping: median grey and p5-p95 span as 35h's plain volume has them
            variants["35h_tissue_map"] = med35 + span35 / S * (u - M)
            for a in A:   # neuropil-matched candidates: contrast x a, neuropil median moved to 35h's (150)
                variants[f"x{a}"] = 150.0 + a * (u - M) - a * (128.5 - M)
            for key, x in variants.items():
                x8 = np.clip(np.rint(x), 0, 255).astype(np.uint8)
                rec[key] = {"plain": metrics(x8[inner])}
                for clip in (1.0, 1.2, 1.5):
                    rec[key][f"clahe{clip}"] = metrics(cv2.createCLAHE(clipLimit=clip, tileGridSize=(K, K)).apply(x8)[inner])
            rows.append(rec)
    print(f"{os.path.basename(v)}: {len(cells)} cells x 2 planes", flush=True)

json.dump(rows, open(f"{OUT}/window_match.json", "w"), indent=1)
print("\n35h neuropil target (from the neuropil run): plain lc8 29.67 lc32 24.21 std 39.15 p50 150.0 | "
      "CLAHE 1.5 lc8 46.88 lc32 38.03 std 58.06 p50 143.0")
print(f"35h whole-slice tissue: median grey {med35:.1f}, p5-p95 span {span35:.1f}; 37i median over slices: "
      f"median grey {g37.t_p50.median():.1f}, span {g37.span.median():.1f}")
print("\n37i neuropil (median over crops)        lc8    lc32     std     p50   ||  CLAHE 1.0 lc8 | 1.2 lc8 | 1.5: lc8  lc32   std   p50")
for key in ["current", "35h_tissue_map"] + [f"x{a}" for a in A]:
    def m(stage, k):
        return float(np.median([r[key][stage][k] for r in rows]))
    print(f"  {key:<36}" + "".join(f"{m('plain', k):8.2f}" for k in ("lc8", "lc32", "std", "p50")) +
          f"   || {m('clahe1.0', 'lc8'):8.2f} | {m('clahe1.2', 'lc8'):7.2f} | " +
          "".join(f"{m('clahe1.5', k):7.2f}" for k in ("lc8", "lc32", "std", "p50")))
print("\nconfig values per candidate x a (window: median): target_span = a * 37i span "
      f"({g37.span.median():.1f}) = " + ", ".join(f"x{a}: {a * g37.span.median():.0f}" for a in A))
print("target_median = 150 - a * (128.5 - 37i median grey " + f"{g37.t_p50.median():.1f}) = " +
      ", ".join(f"x{a}: {150 - a * (128.5 - g37.t_p50.median()):.0f}" for a in A))
PY
