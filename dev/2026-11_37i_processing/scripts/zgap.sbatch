#!/bin/bash
#SBATCH --job-name=em-zgap
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
set -eo pipefail
cd ${EM_PIPE_ROOT}
source slurm/env.sh
python -u - <<'PY'
"""How thick is the step across the 18-19 Oct pause (z 2082 | 2083)? Similarity of planes z and z+k
(normalised cross-correlation in 128 px windows at 16 nm, after each window's best shift) against k on
both sides of the pause, and across it; the distance whose similarity matches the pause's is the step."""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import cv2
import numpy as np
import tensorstore as ts

T = os.path.expandvars("${DEV_DATA_ROOT}/37_i/${DEV_PREFIX}37i_test2083")
V = os.path.expandvars(f"{T}/${{DEV_PREFIX}}37i_test_z2083_8nm.ome.zarr")
Z0 = 2051
X0, X1, Y0, Y1 = 4000, 18000, 4000, 22000      # display px: most of the tissue
W = 128
arr = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": f"{V}/s0"}}, read=True, open=True).result()
planes = {}
for z0 in range(0, 64, 8):   # read in z blocks to bound memory; keep 16 nm planes
    blk = np.asarray(arr[z0:z0 + 8, Y0:Y1, X0:X1].read().result()).astype(np.float32)
    for i in range(blk.shape[0]):
        p = blk[i]
        h, w = p.shape[0] // 2 * 2, p.shape[1] // 2 * 2
        planes[Z0 + z0 + i] = p[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
H, Wd = planes[Z0].shape
han = cv2.createHanningWindow((W, W), cv2.CV_32F)
# windows with tissue in the reference plane z 2084 (not resin, not no-data)
ref = planes[2084]
wins = [(i, j) for i in range(0, H - W, W) for j in range(0, Wd - W, W)
        if (ref[i:i + W, j:j + W] == 0).mean() < 0.01 and ref[i:i + W, j:j + W].std() > 10
        and (ref[i:i + W, j:j + W] > 220).mean() < 0.3]
rng = np.random.default_rng(0)
wins = [wins[k] for k in rng.choice(len(wins), min(400, len(wins)), replace=False)]
print(f"{len(wins)} tissue windows", flush=True)


def sim(za, zb):
    a, b = planes[za], planes[zb]
    out = []
    for i, j in wins:
        wa, wb = a[i:i + W, j:j + W], b[i:i + W, j:j + W]
        if (wa == 0).mean() > 0.01 or (wb == 0).mean() > 0.01:
            continue
        (dx, dy), _ = cv2.phaseCorrelate(wa, wb, han)
        dx, dy = int(round(dx)), int(round(dy))
        if abs(dx) > 16 or abs(dy) > 16:
            dx = dy = 0
        ya, yb = max(0, -dy), max(0, dy)
        xa, xb = max(0, -dx), max(0, dx)
        h, w = W - abs(dy), W - abs(dx)
        pa = cv2.GaussianBlur(wa[ya:ya + h, xa:xa + w], (0, 0), 1.0).ravel()
        pb = cv2.GaussianBlur(wb[yb:yb + h, xb:xb + w], (0, 0), 1.0).ravel()
        out.append(np.corrcoef(pa, pb)[0, 1])
    return float(np.median(out)) if out else np.nan


curve = {}
for side, bases, top in (("before", range(2051, 2069), 2082), ("after", range(2084, 2101), 2114)):
    for k in (1, 2, 3, 4, 6, 8, 10, 14):
        v = [sim(z, z + k) for z in bases if z + k <= top][::3]
        curve[(side, k)] = float(np.median(v))
    print(side, {k: round(curve[(side, k)], 3) for s, k in curve if s == side}, flush=True)
junction = {p: sim(*p) for p in [(2082, 2083), (2083, 2084), (2082, 2084), (2081, 2083), (2081, 2084), (2082, 2085), (2080, 2084)]}
print("across the pause and around z 2083:", {f"{a}->{b}": round(v, 3) for (a, b), v in junction.items()}, flush=True)
ks = np.array([1, 2, 3, 4, 6, 8, 10, 14], float)
for side in ("before", "after"):
    r = np.array([curve[(side, k)] for k in ks])
    for p in [(2082, 2083), (2082, 2084)]:
        v = junction[p]
        k_eff = float(np.interp(-v, -r, ks)) if r.min() <= v <= r.max() else (np.inf if v < r.min() else 0.0)
        print(f"  {p[0]}->{p[1]} ({junction[p]:.3f}) on the {side} curve: like {k_eff:.1f} slices apart "
              f"(nominal {p[1] - p[0]}){' (beyond the curve: more than 14)' if np.isinf(k_eff) else ''}", flush=True)
PY
