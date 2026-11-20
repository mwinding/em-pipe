#!/bin/bash
#SBATCH --job-name=em-neuropil-sheet
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -eo pipefail
cd ${EM_PIPE_ROOT}
source slurm/env.sh
python -u - <<'PY'
"""Every neuropil cell of the CLAHE comparison: the plain volume, 1024 px (8 nm) at the cell centre shown at 16 nm."""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import numpy as np
import tensorstore as ts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S35 = os.path.expandvars("${DEV_35H}")
S37 = os.path.expandvars("${DEV_37I}")
OUT_ROOT = __import__('os').environ.get('DEV_OUT', 'dev_out')  # outputs: a directory you choose, never the dataset folder
OUT = f"{OUT_ROOT}/qc/clahe_match"
CELL, HALF = 2560, 512
CELLS = [("35h", os.path.expandvars(f"{S35}/${{DEV_PREFIX}}35h_8nm.ome.zarr"), 5328, ["H4", "H6", "H7", "I4", "I6", "I7"]),
         ("35h", os.path.expandvars(f"{S35}/${{DEV_PREFIX}}35h_8nm.ome.zarr"), 8848, ["C2", "D2", "E2", "E3", "D8", "E8", "E9", "H5"]),
         ("35h", os.path.expandvars(f"{S35}/${{DEV_PREFIX}}35h_8nm.ome.zarr"), 12368, ["D2", "E3", "F3", "G4", "H5", "C8", "E8", "F8"]),
         ("37i", os.path.expandvars(f"{S37}/${{DEV_PREFIX}}37i_test_z1000_8nm.ome.zarr"), 16, ["B2", "B4", "C1", "C2", "C4"]),
         ("37i", os.path.expandvars(f"{S37}/${{DEV_PREFIX}}37i_test_z9600_8nm.ome.zarr"), 16, ["C1", "C2", "D1", "D2", "E2", "D7", "D8", "E8", "H4", "I4"])]
panels = []
for name, v, z, cells in CELLS:
    arr = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": f"{v}/s0"}}, read=True, open=True).result()
    H, W = arr.shape[1:]
    for cell in cells:
        i, j = ord(cell[0]) - ord("A"), int(cell[1:])
        yc, xc = min(int((i + 0.5) * CELL), H - HALF), min(int((j + 0.5) * CELL), W - HALF)
        a = np.asarray(arr[z, yc - HALF:yc + HALF, xc - HALF:xc + HALF].read().result()).astype(np.float32)
        a = a.reshape(HALF, 2, HALF, 2).mean(axis=(1, 3))
        label = "z1000 slab" if "z1000" in v else "z9600 slab" if "z9600" in v else f"z {z}"
        panels.append((name, f"{name} {label} {cell}", a))
    print(name, v.rsplit("/", 1)[-1], z, len(cells), flush=True)

ncol = 6
for name in ("35h", "37i"):
    ps = [p for p in panels if p[0] == name]
    nrow = -(-len(ps) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.3, nrow * 3.5), layout="constrained", squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, title, a) in zip(axes.flat, ps):
        ax.imshow(a, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(title, fontsize=9)
    fig.suptitle(f"{name} plain volume: the neuropil patches of the CLAHE comparison (8 µm across, shown at 16 nm)")
    fig.savefig(f"{OUT}/neuropil_patches_{name}.png", dpi=110)
    plt.close(fig)
print("wrote", OUT)
PY
