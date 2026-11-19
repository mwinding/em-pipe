#!/bin/bash
#SBATCH --job-name=em-clahe-match
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
set -eo pipefail
cd ${EM_PIPE_ROOT}
source slurm/env.sh
python -u - <<'PY'
"""Match 37i's CLAHE strength to 35h's: CLAHE re-applied offline to tile-aligned crops of the plain
volumes (exactly as render does: cv2 CLAHE on the unrotated uint8 plane, 256 px tiles), metrics on the
crop interiors (one CLAHE tile of margin, so they equal the full-plane result)."""
import json
import math
import os

import cv2
import numpy as np
import tensorstore as ts
from scipy import ndimage

from pipeline import background

S35 = os.path.expandvars("${DEV_35H}")
S37 = os.path.expandvars("${DEV_37I}")
OUT_ROOT = __import__('os').environ.get('DEV_OUT', 'dev_out')  # outputs: a directory you choose, never the dataset folder
OUT = f"{OUT_ROOT}/qc/clahe_match"
os.makedirs(OUT, exist_ok=True)
TILE, K = 256, 10                 # CLAHE tile px; crop = K x K CLAHE tiles, metrics on the inner K-2
CLIPS = [0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0]
NCROP = 4
rng = np.random.default_rng(0)


def opn(root, s):
    return ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": f"{root}/s{s}"}},
                   read=True, open=True).result()


def grid(n):
    g = math.ceil(n / TILE)
    return g, (n if n % g == 0 else n + g - n % g) // g    # tiles, tile px (cv2 pads to a multiple)


def metrics(x):
    x = x.astype(np.float32)
    m = ndimage.uniform_filter(x, 32)
    lc8 = np.sqrt(np.maximum(ndimage.uniform_filter(x * x, 32) - m * m, 0)).mean()
    h, w = (s // 4 * 4 for s in x.shape)
    y = x[:h, :w].reshape(h // 4, 4, w // 4, 4).mean(axis=(1, 3))
    m = ndimage.uniform_filter(y, 8)
    lc32 = np.sqrt(np.maximum(ndimage.uniform_filter(y * y, 8) - m * m, 0)).mean()
    d = np.diff(x, axis=1).ravel()
    noise = 1.4826 * np.median(np.abs(d - np.median(d))) / math.sqrt(2)
    p5, p50, p95 = np.percentile(x, [5, 50, 95])
    return {"lc8": float(lc8), "lc32": float(lc32), "noise8": float(noise), "std": float(x.std()),
            "p5": float(p5), "p50": float(p50), "p95": float(p95)}


def crops_for(plain, z, n):
    """Up to n tile-aligned K x K crops (unrotated tile indices) whose inner part is >= 97 % tissue."""
    H, W = plain.shape[1:]
    (gx, tx), (gy, ty) = grid(W), grid(H)
    lo = np.asarray(opn(plain_root[plain], 4)[z // 16].read().result())[::-1, ::-1]   # unrotated, 128 nm
    bg = background.mask(lo) | ndimage.binary_dilation(lo == 0, iterations=4)
    f = W / lo.shape[1]
    ok = []
    for iy in range(0, gy - K):
        for ix in range(0, gx - K):
            y0, y1 = int((iy + 1) * ty / f), int((iy + K - 1) * ty / f)
            x0, x1 = int((ix + 1) * tx / f), int((ix + K - 1) * tx / f)
            if 1 - bg[y0:y1, x0:x1].mean() >= 0.97:
                ok.append((iy, ix))
    picked = []
    for i in rng.permutation(len(ok)):
        iy, ix = ok[i]
        if all(abs(iy - a) >= K or abs(ix - b) >= K for a, b in picked):
            picked.append((iy, ix))
        if len(picked) == n:
            break
    return picked, (tx, ty), (H, W)


def read_crop(arr, z, iy, ix, tx, ty, H, W):
    """Unrotated crop of K x K CLAHE tiles at tile (iy, ix) from a volume stored rotated 180 degrees."""
    ys, xs = slice(H - (iy + K) * ty, H - iy * ty), slice(W - (ix + K) * tx, W - ix * tx)
    return np.asarray(arr[z, ys, xs].read().result())[::-1, ::-1]


plain_root = {}
sets = {"35h": [], "37i": []}
v35p, v35c = os.path.expandvars(f"{S35}/${{DEV_PREFIX}}35h_8nm.ome.zarr"), os.path.expandvars(f"{S35}/${{DEV_PREFIX}}35h_8nm_clahe.ome.zarr")
n35 = opn(v35p, 0).shape[0]
for frac in (0.2, 0.5, 0.8):
    z0 = 64 * round(n35 * frac / 64)
    sets["35h"] += [(v35p, v35c, z0 + 16), (v35p, v35c, z0 + 48)]
for d in (1000, 9600, 16500):
    p, c = os.path.expandvars(f"{S37}/${{DEV_PREFIX}}37i_test_z{d}_8nm.ome.zarr"), os.path.expandvars(f"{S37}/${{DEV_PREFIX}}37i_test_z{d}_8nm_clahe.ome.zarr")
    sets["37i"] += [(p, c, 16), (p, c, 48)]

rows, panels = [], {}
for name, items in sets.items():
    for p, c, z in items:
        P, C = opn(p, 0), opn(c, 0)
        plain_root[P] = p
        picked, (tx, ty), (H, W) = crops_for(P, z, NCROP)
        print(f"{name} {os.path.basename(p)} z {z}: {len(picked)} crops, tile {tx}x{ty}", flush=True)
        for iy, ix in picked:
            plain = read_crop(P, z, iy, ix, tx, ty, H, W)
            vol = read_crop(C, z, iy, ix, tx, ty, H, W)
            inner = (slice(ty, -ty), slice(tx, -tx))
            rec = {"set": name, "vol": os.path.basename(p), "z": z, "iy": iy, "ix": ix,
                   "plain": metrics(plain[inner]), "volume_clahe": metrics(vol[inner])}
            for clip in CLIPS:
                out = cv2.createCLAHE(clipLimit=clip, tileGridSize=(K, K)).apply(plain)
                if clip == 1.5:
                    diff = np.abs(out[inner].astype(int) - vol[inner])
                    rec["repro_exact_frac"], rec["repro_max_diff"] = float((diff == 0).mean()), int(diff.max())
                rec[f"clip{clip}"] = metrics(out[inner])
                if len(panels.get(name, [])) < 2 and z in (48, ) or (name == "35h" and len(panels.get(name, [])) < 2):
                    pass
            rows.append(rec)
            key = (name, len([r for r in rows if r["set"] == name]))
            if key[1] <= 3:   # keep a few crops for the picture: 1024 px from the inner part
                c0y, c0x = ty + (plain.shape[0] - 2 * ty - 1024) // 2, tx + (plain.shape[1] - 2 * tx - 1024) // 2
                sl = (slice(c0y, c0y + 1024), slice(c0x, c0x + 1024))
                panels[key] = {"plain": plain[sl], **{f"clip{cl}": cv2.createCLAHE(clipLimit=cl, tileGridSize=(K, K))
                                                      .apply(plain)[sl] for cl in CLIPS}, "label": f"{rec['vol']} z{z}"}

json.dump(rows, open(f"{OUT}/metrics.json", "w"), indent=1)


def agg(name, key, m):
    return float(np.median([r[key][m] for r in rows if r["set"] == name]))


print("\nreproduction (clip 1.5 offline vs the CLAHE volume, crop interiors): exact fraction min",
      min(r["repro_exact_frac"] for r in rows), "max diff", max(r["repro_max_diff"] for r in rows))
ms = ["lc8", "lc32", "noise8", "std", "p5", "p50", "p95"]
print("\nmedian over crops        " + "".join(f"{m:>9}" for m in ms))
for name in sets:
    for key in ["plain", "volume_clahe"] + [f"clip{c}" for c in CLIPS]:
        print(f"{name} {key:<20}" + "".join(f"{agg(name, key, m):9.2f}" for m in ms))
print("\nclip for 37i matching 35h at clip 1.5 (linear interpolation over the sweep):")
for m in ("lc8", "lc32", "std", "noise8"):
    target = agg("35h", "clip1.5", m)
    ys = [agg("37i", f"clip{c}", m) for c in CLIPS]
    best = float(np.interp(target, ys, CLIPS)) if all(np.diff(ys) > 0) else None
    print(f"  {m}: 35h {target:.2f}; 37i {['%.2f' % y for y in ys]} -> clip {best}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
show = ["plain", "clip0.8", "clip1.0", "clip1.2", "clip1.5"]
keys = sorted(panels)
fig, axes = plt.subplots(len(keys), len(show), figsize=(3.2 * len(show), 3.4 * len(keys)), layout="constrained")
for i, k in enumerate(keys):
    for j, s in enumerate(show):
        ax = axes[i, j]
        ax.imshow(panels[k][s], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(f"{k[0]} {s}" + (f"\n{panels[k]['label']}" if j == 0 else ""), fontsize=8)
        ax.axis("off")
fig.savefig(f"{OUT}/crops_8nm.png", dpi=90)
print("wrote", OUT, flush=True)
PY
