#!/bin/bash
#SBATCH --job-name=em-zprofile
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-19
#SBATCH --output=%x-%A_%a.out
set -eo pipefail
cd ${EM_PIPE_ROOT}
source slurm/env.sh
python -u - <<'PY'
"""z-step profile of the final 37i volume (v2). Similarity of planes z and z+k (median phase-correlation
peak in 256 px windows) against k, every 32 slices, and before / after / across each acquisition pause.
Windows: up to 400 tissue windows anywhere in the slice, found on the 16x level (s4) and read alone at
full resolution (whole 64-plane chunks, so both bases of a block come from one read); each pair keeps the
windows that are tissue in both planes (mean < 195 and SD > 18: resin is ~218 / SD 9-13). A faster fall
with k means thicker slices, or blurrier images: 'detail' (1-2 px band-pass SD over SD) tells them apart."""
import json
import os

import cv2
import numpy as np
import tensorstore as ts

S = os.path.expandvars("${DEV_37I}")
OUT_ROOT = __import__('os').environ.get('DEV_OUT', 'dev_out')  # outputs: a directory you choose, never the dataset folder
V = os.path.expandvars(f"{S}/${{DEV_PREFIX}}37i_8nm.ome.zarr")
OUT = f"{OUT_ROOT}/qc/zstep_profile"
os.makedirs(OUT, exist_ok=True)
task, ntask = int(os.environ["SLURM_ARRAY_TASK_ID"]), int(os.environ["SLURM_ARRAY_TASK_COUNT"])
KS = [1, 2, 3, 4, 6, 8]
WIN, NWIN = 256, 400
PAUSES = [2083, 5559, 5782, 11014, 17333]             # 5782: clock-change control, no real pause
han = cv2.createHanningWindow((WIN, WIN), cv2.CV_32F)


def open_(level):
    return ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": f"{V}/{level}"}},
                   read=True, open=True).result()


s0, s4 = open_("s0"), open_("s4")
n, H, W = s0.shape


def tissue(w):
    return (w == 0).mean() < 0.02 and w.mean() < 195 and w.std() > 18 and (w > 225).mean() < 0.1


def windows(z):
    """Up to NWIN tissue windows (y, x) at s0 for plane z, chosen on s4 (16 px there = 256 at s0)."""
    p = np.asarray(s4[min(z // 16, s4.shape[0] - 1)].read().result()).astype(np.float32)
    c = WIN // 16
    out = []
    for i in range(0, p.shape[0] - c + 1, c):
        for j in range(0, p.shape[1] - c + 1, c):
            b = p[i:i + c, j:j + c]
            if b.min() > 0 and b.mean() < 190 and (b > 210).mean() < 0.1 and (i + c) * 16 <= H and (j + c) * 16 <= W:
                out.append((i * 16, j * 16))
    rng = np.random.default_rng(z)
    if len(out) > NWIN:
        out = [out[k] for k in sorted(rng.choice(len(out), NWIN, replace=False))]
    return out


def read(wins, z0, z1):
    futs = [s0[z0:z1, y:y + WIN, x:x + WIN].read() for y, x in wins]
    return [np.asarray(f.result()) for f in futs]       # each (z1 - z0, WIN, WIN) uint8


def peak(stack, a, c):
    vals = []
    for blk in stack:
        wa, wc = blk[a].astype(np.float32), blk[c].astype(np.float32)
        if tissue(wa) and tissue(wc):
            vals.append(cv2.phaseCorrelate(wa, wc, han)[1])
    return (float(np.median(vals)), len(vals)) if len(vals) >= 20 else (None, len(vals))


def detail(stack, a):
    vals = []
    for blk in stack:
        w = blk[a].astype(np.float32)
        if tissue(w):
            vals.append((cv2.GaussianBlur(w, (0, 0), 1.0) - cv2.GaussianBlur(w, (0, 0), 2.0))[8:-8, 8:-8].std() / w.std())
    return float(np.median(vals)) if len(vals) >= 20 else None


with open(f"{OUT}/v2_{task:02d}.jsonl", "w") as f:
    for b in range(task, (n + 63) // 64, ntask):
        z0 = 64 * b
        bases = [z for z in (z0 + 4, z0 + 36) if z + max(KS) < n]
        if not bases:
            continue
        wins = windows(z0 + 20)
        st = read(wins, z0, min(n, z0 + 64))
        for z in bases:
            curve = {k: peak(st, z - z0, z - z0 + k) for k in KS}
            f.write(json.dumps({"z": z, "curve": curve, "detail": detail(st, z - z0), "cand": len(wins)}) + "\n")
            f.flush()
        print(f"block {b} (z {z0}): {len(wins)} windows, k1 {curve[1]}", flush=True)
    for j, zp in enumerate(PAUSES):
        if j % ntask != task:
            continue
        wins = windows(zp)
        st = read(wins, zp - 9, zp + 10)
        o = lambda z: z - (zp - 9)
        rec = {"pause": zp,
               "before": {k: peak(st, o(zp - 9), o(zp - 9 + k)) for k in range(1, 9)},
               "after": {k: peak(st, o(zp), o(zp + k)) for k in range(1, 9)},
               "across": {f"{a}->{c}": peak(st, o(a), o(c))
                          for a, c in [(zp - 1, zp), (zp - 2, zp), (zp - 1, zp + 1), (zp - 2, zp + 1), (zp - 3, zp)]},
               "detail_before": detail(st, o(zp - 1)), "detail_after": detail(st, o(zp))}
        f.write(json.dumps(rec) + "\n")
        print(f"pause {zp} done", flush=True)
PY
