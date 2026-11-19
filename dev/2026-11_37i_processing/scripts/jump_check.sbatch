#!/bin/bash
#SBATCH --job-name=em-jump-check
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -eo pipefail
cd ${EM_PIPE_ROOT}
source slurm/env.sh
python -u - <<'PY'
"""Local shifts between consecutive planes of the z 2083 test slab along the 0-0 / 1-0 seam (display
coordinates: right half, around y 13740): phase correlation in 256 px windows with tissue."""
import os
import sys

import cv2
import numpy as np
import tensorstore as ts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S37 = os.path.expandvars("${DEV_37I}")
OUT_ROOT = __import__('os').environ.get('DEV_OUT', 'dev_out')  # outputs: a directory you choose, never the dataset folder
V = os.path.expandvars(f"{S37}/${{DEV_PREFIX}}37i_test_z2083_8nm.ome.zarr")
TAG = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TAG", "now")
Z0 = 2051
X0, X1, Y0, Y1 = 10500, 20500, 11000, 16500
WIN = 256
arr = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": f"{V}/s0"}}, read=True, open=True).result()
planes = {z: None for z in range(2080, 2087)}
block = np.asarray(arr[2080 - Z0:2087 - Z0, Y0:Y1, X0:X1].read().result())
for i, z in enumerate(planes):
    planes[z] = block[i].astype(np.float32)
han = cv2.createHanningWindow((WIN, WIN), cv2.CV_32F)
pairs = [(z, z + 1) for z in range(2080, 2086)]
ny, nx = (Y1 - Y0) // WIN, (X1 - X0) // WIN
maps = {}
for za, zb in pairs:
    m = np.full((ny, nx), np.nan)
    a, b = planes[za], planes[zb]
    for i in range(ny):
        for j in range(nx):
            wa = a[i * WIN:(i + 1) * WIN, j * WIN:(j + 1) * WIN]
            wb = b[i * WIN:(i + 1) * WIN, j * WIN:(j + 1) * WIN]
            if (wa == 0).mean() > 0.05 or wa.std() < 8 or (wa > 230).mean() > 0.5:   # no data or resin
                continue
            (dx, dy), resp = cv2.phaseCorrelate(wa, wb, han)
            if resp > 0.05:
                m[i, j] = np.hypot(dx, dy)
    maps[(za, zb)] = m
    v = m[np.isfinite(m)]
    print(f"z {za}->{zb}: {len(v)} windows, shift median {np.median(v):.2f} px, p90 {np.percentile(v, 90):.2f}, "
          f"p99 {np.percentile(v, 99):.2f}, max {v.max():.2f}; windows > 3 px: {(v > 3).sum()}", flush=True)
fig, axes = plt.subplots(len(pairs), 1, figsize=(10, 2.6 * len(pairs)), layout="constrained")
for ax, (k, m) in zip(axes, maps.items()):
    im = ax.imshow(m, vmin=0, vmax=6, cmap="magma", interpolation="nearest")
    ax.set_title(f"|shift| z {k[0]} -> {k[1]} (256 px windows; x {X0}-{X1}, y {Y0}-{Y1}, display)", fontsize=9)
    ax.axis("off")
fig.colorbar(im, ax=axes, shrink=0.6, label="px")
fig.savefig(f"{OUT_ROOT}/qc/z2083_shifts_{TAG}.png", dpi=90)
# full-res crops at the worst window of 2082->2083 / 2083->2084 for a look
m = np.nan_to_num(maps[(2082, 2083)]) + np.nan_to_num(maps[(2083, 2084)])
i, j = np.unravel_index(np.argmax(m), m.shape)
cy, cx = i * WIN + WIN // 2, j * WIN + WIN // 2
r0, c0 = max(0, cy - 400), max(0, cx - 400)
fig, axes = plt.subplots(1, 3, figsize=(15, 5.3), layout="constrained")
for ax, z in zip(axes, (2082, 2083, 2084)):
    ax.imshow(planes[z][r0:r0 + 800, c0:c0 + 800], cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"z {z}  (display x {X0 + c0}, y {Y0 + r0}, 800 px)", fontsize=9)
    ax.axis("off")
fig.savefig(f"{OUT_ROOT}/qc/z2083_worst_{TAG}.png", dpi=90)
print("worst window at display x", X0 + cx, "y", Y0 + cy, "sum of shifts", round(float(m.max()), 2))
PY
