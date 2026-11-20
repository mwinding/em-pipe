#!/bin/bash
#SBATCH --job-name=em-zstep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
set -eo pipefail
cd ${EM_PIPE_ROOT}
source slurm/env.sh
python -u - <<'PY'
"""Similarity of planes z and z+k (median phase-correlation peak in 256 px tissue windows) against k, in
the test slabs at different depths: a faster fall means thicker slices (or blurrier images)."""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import cv2
import numpy as np
import tensorstore as ts

S = os.path.expandvars("${DEV_37I}")
T = os.path.expandvars("${DEV_DATA_ROOT}/37_i/${DEV_PREFIX}37i_test2083")
SLABS = [("z1000 (16 Oct, single tile)", os.path.expandvars(f"{S}/${{DEV_PREFIX}}37i_test_z1000_8nm.ome.zarr"), 1000),
         ("z2051-2082 (17 Oct, before the pause)", os.path.expandvars(f"{T}/${{DEV_PREFIX}}37i_test_z2083_8nm.ome.zarr"), 2051),
         ("z2084-2114 (19 Oct, after the pause)", os.path.expandvars(f"{T}/${{DEV_PREFIX}}37i_test_z2083_8nm.ome.zarr"), 2051),
         ("z9600 (~30 Oct)", os.path.expandvars(f"{S}/${{DEV_PREFIX}}37i_test_z9600_8nm.ome.zarr"), 9600),
         ("z16500 (~9 Nov)", os.path.expandvars(f"{S}/${{DEV_PREFIX}}37i_test_z16500_8nm.ome.zarr"), 16500)]
KS = [1, 2, 3, 4, 5, 6, 8]
WIN = 256
han = cv2.createHanningWindow((WIN, WIN), cv2.CV_32F)


def tissue(w):
    return (w == 0).mean() < 0.02 and w.std() > 12 and (w > 230).mean() < 0.4


for label, v, z0 in SLABS:
    arr = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": f"{v}/s0"}}, read=True, open=True).result()
    n, H, W = arr.shape
    cy, cx = H // 2, W // 2
    ys, xs = slice(max(0, cy - 3000), min(H, cy + 3000)), slice(max(0, cx - 4000), min(W, cx + 4000))
    if "before" in label:
        bases = [4, 12]          # planes 4-20 (+k <= 31: z <= 2082)
    elif "after" in label:
        bases = [33, 45]         # z 2084, 2096
    else:
        bases = [10, 30, 45]
    need = sorted({b + k for b in bases for k in [0, *KS] if b + k < n})
    block = {p: np.asarray(arr[p, ys, xs].read().result()).astype(np.float32) for p in need}
    curve = {}
    for k in KS:
        vals = []
        for b in bases:
            if b + k >= n or ("before" in label and b + k > 31):
                continue
            a, c = block[b], block[b + k]
            for i in range(0, a.shape[0] - WIN, WIN):
                for j in range(0, a.shape[1] - WIN, WIN):
                    wa, wc = a[i:i + WIN, j:j + WIN], c[i:i + WIN, j:j + WIN]
                    if tissue(wa) and tissue(wc):
                        vals.append(cv2.phaseCorrelate(wa, wc, han)[1])
        curve[k] = (float(np.median(vals)), len(vals)) if vals else (np.nan, 0)
    print(f"{label}: " + ", ".join(f"k{k} {m:.3f}" for k, (m, _) in curve.items())
          + f"  ({curve[1][1]} windows at k1)", flush=True)
PY
