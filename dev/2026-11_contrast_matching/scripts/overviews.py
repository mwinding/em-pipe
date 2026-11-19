"""s4 (128 nm) overviews in display orientation with a lettered grid of 2560 px cells (s0), to pick neuropil."""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import string
import sys

import numpy as np
import tensorstore as ts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S35 = os.path.expandvars("${DEV_35H}")
S37 = os.path.expandvars("${DEV_37I}")
OUT_ROOT = __import__('os').environ.get('DEV_OUT', 'dev_out')  # outputs: a directory you choose, never the dataset folder
OUT = f"{OUT_ROOT}/qc/clahe_match"
CELL = 2560


def opn(root, s):
    return ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": f"{root}/s{s}"}}, read=True, open=True).result()


v35 = os.path.expandvars(f"{S35}/${{DEV_PREFIX}}35h_8nm.ome.zarr")
n35 = opn(v35, 0).shape[0]
items = [("35h", v35, 64 * round(n35 * f / 64) + 16) for f in (0.15, 0.3, 0.5, 0.7, 0.85)]
items += [("37i", os.path.expandvars(f"{S37}/${{DEV_PREFIX}}37i_test_z{d}_8nm.ome.zarr"), 16) for d in (1000, 9600, 16500)]
for name, v, z in items:
    H, W = opn(v, 0).shape[1:]
    img = np.asarray(opn(v, 4)[z // 16].read().result())
    f = W / img.shape[1]
    fig, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    for i, y in enumerate(range(0, H, CELL)):
        ax.axhline(y / f, color="yellow", lw=0.4)
        for j, x in enumerate(range(0, W, CELL)):
            ax.text((x + CELL / 2) / f, (y + CELL / 2) / f, f"{string.ascii_uppercase[i]}{j}", color="yellow",
                    fontsize=6, ha="center", va="center")
    for x in range(0, W, CELL):
        ax.axvline(x / f, color="yellow", lw=0.4)
    ax.set_title(f"{name} {v.rsplit('/', 1)[-1]} z {z}  (display orientation, cells {CELL} px)", fontsize=8)
    ax.axis("off")
    tag = f"{name}_{v.rsplit('/', 1)[-1].replace('.ome.zarr', '')}_z{z}"
    fig.savefig(f"{OUT}/overview_{tag}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(tag, H, W, file=sys.stderr)
