#!/bin/bash
#SBATCH --job-name=em-window-pics
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -eo pipefail
cd ${EM_PIPE_ROOT}
source slurm/env.sh
python -u - <<'PY'
"""Side by side at 8 nm: 35h neuropil vs 37i neuropil, plain and CLAHE 1.5, 37i now (tissue 1-99 %) and with
the median window matched on neuropil (u' = 150 + 0.75 (u - 128.5), i.e. window: median, target_median 151,
target_span 123)."""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import math

import cv2
import numpy as np
import tensorstore as ts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S35 = os.path.expandvars("${DEV_35H}")
S37 = os.path.expandvars("${DEV_37I}")
OUT_ROOT = __import__('os').environ.get('DEV_OUT', 'dev_out')  # outputs: a directory you choose, never the dataset folder
OUT = f"{OUT_ROOT}/qc/clahe_match"
TILE, K, CELL, SHOW = 256, 10, 2560, 640


def opn(root):
    return ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": f"{root}/s0"}}, read=True, open=True).result()


def grid(n):
    g = math.ceil(n / TILE)
    return g, (n if n % g == 0 else n + g - n % g) // g


def crop(v, z, cell):
    """Tile-aligned unrotated crop around a display cell: (plain crop, slice of its centre SHOW px)."""
    P = opn(v)
    H, W = P.shape[1:]
    (gx, tx), (gy, ty) = grid(W), grid(H)
    i, j = ord(cell[0]) - ord("A"), int(cell[1:])
    yu, xu = H - 1 - (i + 0.5) * CELL, W - 1 - (j + 0.5) * CELL
    iy = min(max(0, round(yu / ty - K / 2)), gy - K - 1)
    ix = min(max(0, round(xu / tx - K / 2)), gx - K - 1)
    ys, xs = slice(H - (iy + K) * ty, H - iy * ty), slice(W - (ix + K) * tx, W - ix * tx)
    u = np.asarray(P[z, ys, xs].read().result())[::-1, ::-1]
    c0y, c0x = (u.shape[0] - SHOW) // 2, (u.shape[1] - SHOW) // 2
    return u, (slice(c0y, c0y + SHOW), slice(c0x, c0x + SHOW))


def clahe(u):
    return cv2.createCLAHE(clipLimit=1.5, tileGridSize=(K, K)).apply(u)


def show(u):   # back to display orientation for the picture
    return u[::-1, ::-1]


PAIRS = [((os.path.expandvars(f"{S35}/${{DEV_PREFIX}}35h_8nm.ome.zarr"), 8848, "D8"), (os.path.expandvars(f"{S37}/${{DEV_PREFIX}}37i_test_z9600_8nm.ome.zarr"), 16, "D2")),
         ((os.path.expandvars(f"{S35}/${{DEV_PREFIX}}35h_8nm.ome.zarr"), 12368, "E3"), (os.path.expandvars(f"{S37}/${{DEV_PREFIX}}37i_test_z9600_8nm.ome.zarr"), 16, "E2")),
         ((os.path.expandvars(f"{S35}/${{DEV_PREFIX}}35h_8nm.ome.zarr"), 5328, "H6"), (os.path.expandvars(f"{S37}/${{DEV_PREFIX}}37i_test_z9600_8nm.ome.zarr"), 16, "D7")),
         ((os.path.expandvars(f"{S35}/${{DEV_PREFIX}}35h_8nm.ome.zarr"), 8848, "C2"), (os.path.expandvars(f"{S37}/${{DEV_PREFIX}}37i_test_z1000_8nm.ome.zarr"), 16, "B2"))]
cols = ["35h plain", "37i plain now", "37i plain new window", "35h CLAHE 1.5", "37i CLAHE 1.5 now", "37i CLAHE 1.5 new window"]
fig, axes = plt.subplots(len(PAIRS), len(cols), figsize=(len(cols) * 3.4, len(PAIRS) * 3.6), layout="constrained")
for r, ((v35, z35, c35), (v37, z37, c37)) in enumerate(PAIRS):
    a, sa = crop(v35, z35, c35)
    b, sb = crop(v37, z37, c37)
    bn = np.clip(np.rint(150 + 0.75 * (b.astype(np.float32) - 128.5)), 0, 255).astype(np.uint8)
    imgs = [a[sa], b[sb], bn[sb], clahe(a)[sa], clahe(b)[sb], clahe(bn)[sb]]
    names = [f"35h z{z35} {c35}", f"37i {v37.split('test_')[1].split('_')[0]} {c37}"]
    for c, (img, title) in enumerate(zip(imgs, cols)):
        ax = axes[r, c]
        ax.imshow(show(img), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(f"{title}\n{names[0] if c in (0, 3) else names[1]}", fontsize=8)
        ax.axis("off")
fig.suptitle("Neuropil at 8 nm (5 µm across): 37i now (tissue 1-99 %) vs 37i with the median window matched to 35h "
             "(target_median 151, target_span 123); CLAHE 1.5 as 35h", fontsize=10)
fig.savefig(f"{OUT}/window_match_8nm.png", dpi=110)
print("wrote", f"{OUT}/window_match_8nm.png")
PY
