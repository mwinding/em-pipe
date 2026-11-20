#!/bin/bash
#SBATCH --job-name=em-zfine
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
set -eo pipefail
cd ${EM_PIPE_ROOT}
source slurm/env.sh
python -u - <<'PY'
"""Slice-by-slice z-step around the 18-19 Oct pause (z 2082 | 2083) in the final volume. Similarity of z and
z+k (median phase-correlation peak, 300 tissue windows as in the profile) for every z in 2052-2165; each
pair's distance in 'reference slices' read off a reference curve (before the pause: z 2052-2075; after it:
z 2130-2160), from z->z+3 (well inside the measured range); summed over the stretch = its real depth."""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import json

import cv2
import numpy as np
import tensorstore as ts

S = os.path.expandvars("${DEV_37I}")
OUT_ROOT = __import__('os').environ.get('DEV_OUT', 'dev_out')  # outputs: a directory you choose, never the dataset folder
V = os.path.expandvars(f"{S}/${{DEV_PREFIX}}37i_8nm.ome.zarr")
WIN, NWIN, Z0, Z1, ZP = 256, 300, 2048, 2176, 2083
han = cv2.createHanningWindow((WIN, WIN), cv2.CV_32F)
op = lambda l: ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": f"{V}/{l}"}}, read=True, open=True).result()
s0, s4 = op("s0"), op("s4")
p = np.asarray(s4[2100 // 16].read().result()).astype(np.float32)
c = WIN // 16
cand = [(i * 16, j * 16) for i in range(0, p.shape[0] - c + 1, c) for j in range(0, p.shape[1] - c + 1, c)
        if (b := p[i:i + c, j:j + c]).min() > 0 and b.mean() < 190 and (b > 210).mean() < 0.1]
rng = np.random.default_rng(0)
wins = [cand[k] for k in sorted(rng.choice(len(cand), min(NWIN, len(cand)), replace=False))]
st = []
for i in range(0, len(wins), 50):
    futs = [s0[Z0:Z1, y:y + WIN, x:x + WIN].read() for y, x in wins[i:i + 50]]
    st += [np.asarray(f.result()) for f in futs]
print(f"{len(wins)} windows read", flush=True)


def tissue(w):
    return (w == 0).mean() < 0.02 and w.mean() < 195 and w.std() > 18 and (w > 225).mean() < 0.1


def sim(a, b):
    v = []
    for blk in st:
        wa, wb = blk[a - Z0].astype(np.float32), blk[b - Z0].astype(np.float32)
        if tissue(wa) and tissue(wb):
            v.append(cv2.phaseCorrelate(wa, wb, han)[1])
    return float(np.median(v)) if len(v) >= 20 else np.nan


KS = range(1, 9)
S_ = {(z, k): sim(z, z + k) for z in range(2052, 2166) for k in KS if z + k < Z1}
ref = {}
for name, zs in (("before", range(2052, 2076)), ("after", range(2130, 2161))):
    ref[name] = np.array([np.nanmedian([S_[(z, k)] for z in zs if (z, k) in S_ and (name == "after" or z + k < ZP)]) for k in KS])
    print(f"reference {name}: " + " ".join(f"{v:.3f}" for v in ref[name]), flush=True)
ks = np.arange(1, 9, dtype=float)


def dist(v, r):
    """Distance in reference slices for similarity v (linear between the reference's samples)."""
    if not np.isfinite(v):
        return np.nan
    if v >= r[0]:
        return 1.0 * (1 - (v - r[0]) / (1 - r[0]))     # closer than one slice: shrink toward 0 at similarity 1
    if v <= r[-1]:
        return np.inf
    return float(np.interp(-v, -r, ks))


rows = []
for z in range(2052, 2166):
    rec = {"z": z, "s1": S_.get((z, 1)), "s2": S_.get((z, 2)), "s3": S_.get((z, 3))}
    for name in ("before", "after"):
        rec[f"step_{name}"] = dist(S_.get((z, 3), np.nan), ref[name]) / 3     # mean step of z..z+3
    rows.append(rec)
print("\n  z    s1    s2    s3   step(before ref)  step(after ref)   [step = mean of z->z+3 in reference slices]")
for r in rows:
    if 2070 <= r["z"] <= 2160:
        print(f"{r['z']}  {r['s1']:.3f} {r['s2']:.3f} {r['s3']:.3f}   {r['step_before']:.2f}   {r['step_after']:.2f}"
              + ("   <- crosses the pause" if r["z"] < ZP <= r["z"] + 3 else ""), flush=True)
print("\nacross the pause: " + ", ".join(f"{a}->{b} {sim(a, b):.3f}" for a, b in
      [(2082, 2083), (2081, 2083), (2082, 2084), (2083, 2084), (2083, 2085), (2083, 2086), (2084, 2087)]))
json.dump({"rows": rows, "ref": {k: v.tolist() for k, v in ref.items()}},
          open(f"{OUT_ROOT}/qc/zstep_profile/fine_2083.json", "w"), default=float)
PY
