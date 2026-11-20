#!/bin/bash
#SBATCH --job-name=em-zprofile
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-19
#SBATCH --output=%x-%A_%a.out
set -eo pipefail
cd ${EM_PIPE_ROOT}
source slurm/env.sh
python -u - <<'PY'
"""z-step profile of the final 37i volume. Similarity of planes z and z+k (median phase-correlation peak in
256 px tissue windows) against k, every 32 slices along the stack, and around each acquisition pause
(before, after, and across it). A faster fall with k means thicker slices (or blurrier images: 'detail'
is each base plane's fine-detail fraction, std of a 1-2 px band-pass over std, to tell the two apart)."""
import json
import os

import cv2
import numpy as np
import tensorstore as ts

S = os.path.expandvars("${DEV_37I}")
OUT_ROOT = __import__('os').environ.get('DEV_OUT', 'dev_out')  # outputs: a directory you choose, never the dataset folder
V = os.path.expandvars(f"{S}/${{DEV_PREFIX}}37i_8nm.ome.zarr/s0")
OUT = f"{OUT_ROOT}/qc/zstep_profile"
os.makedirs(OUT, exist_ok=True)
task, ntask = int(os.environ["SLURM_ARRAY_TASK_ID"]), 20
KS = [1, 2, 3, 4, 6, 8]
WIN = 256
YS, XS = slice(11264, 17408), slice(9728, 17920)      # central 6144 x 8192 px, shard-aligned
PAUSES = [2083, 5559, 5782, 11014, 17333]             # 5782: clock-change control, no real pause
han = cv2.createHanningWindow((WIN, WIN), cv2.CV_32F)
arr = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": V}}, read=True, open=True).result()
n = arr.shape[0]


def tissue(w):
    return (w == 0).mean() < 0.02 and w.std() > 12 and (w > 230).mean() < 0.4


def peak(a, c):
    vals = []
    for i in range(0, a.shape[0] - WIN + 1, WIN):
        for j in range(0, a.shape[1] - WIN + 1, WIN):
            wa, wc = a[i:i + WIN, j:j + WIN], c[i:i + WIN, j:j + WIN]
            if tissue(wa) and tissue(wc):
                vals.append(cv2.phaseCorrelate(wa, wc, han)[1])
    return (float(np.median(vals)), len(vals)) if len(vals) >= 20 else (None, len(vals))


def detail(a):
    vals = []
    b1, b2 = cv2.GaussianBlur(a, (0, 0), 1.0), cv2.GaussianBlur(a, (0, 0), 2.0)
    for i in range(0, a.shape[0] - WIN + 1, WIN):
        for j in range(0, a.shape[1] - WIN + 1, WIN):
            w = a[i:i + WIN, j:j + WIN]
            if tissue(w):
                vals.append((b1 - b2)[i:i + WIN, j:j + WIN].std() / w.std())
    return float(np.median(vals)) if len(vals) >= 20 else None


def read(z0, z1):
    blk = np.asarray(arr[z0:z1, YS, XS].read().result())
    return {z0 + i: blk[i].astype(np.float32) for i in range(blk.shape[0])}


with open(f"{OUT}/task_{task:02d}.jsonl", "w") as f:
    for b in range(task, (n + 63) // 64, ntask):
        z0 = 64 * b
        bases = [z for z in (z0 + 4, z0 + 36) if z + max(KS) < n]
        if not bases:
            continue
        pl = read(bases[0], bases[-1] + max(KS) + 1)
        for z in bases:
            curve = {k: peak(pl[z], pl[z + k]) for k in KS}
            f.write(json.dumps({"z": z, "curve": curve, "detail": detail(pl[z])}) + "\n")
            f.flush()
        print(f"block {b} (z {z0}) done", flush=True)
    for j, zp in enumerate(PAUSES):
        if j % ntask != task:
            continue
        pl = read(zp - 9, zp + 10)
        rec = {"pause": zp,
               "before": {k: peak(pl[zp - 9], pl[zp - 9 + k]) for k in range(1, 9)},
               "after": {k: peak(pl[zp], pl[zp + k]) for k in range(1, 9)},
               "across": {f"{a}->{c}": peak(pl[a], pl[c])
                          for a, c in [(zp - 1, zp), (zp - 2, zp), (zp - 1, zp + 1), (zp - 2, zp + 1), (zp - 3, zp)]},
               "detail_before": detail(pl[zp - 1]), "detail_after": detail(pl[zp])}
        f.write(json.dumps(rec) + "\n")
        print(f"pause {zp} done", flush=True)
PY
