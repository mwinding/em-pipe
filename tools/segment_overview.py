"""One overview slice per acquisition segment of a rendered volume, with the tiles outlined.

A segment (check step, work/check/slices.csv) is a run of slices acquired with one tile layout (grid
and tile size); a new one starts where the tiling changed. For each segment the middle slice is drawn
from a low-resolution scale of the volume, with every tile's outline from the render plan, and labelled
with its z range, dates and layout.

  python tools/segment_overview.py --config CONFIG --volume VOLUME.ome.zarr OUT.png [--scale 5] [--output-dir DIR]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from refmask_check import Dataset  # noqa: E402
from pipeline import omezarr, preview, refmask  # noqa: E402
from pipeline.config import deep_merge, load_config, step_path  # noqa: E402

COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#008300", "#e34948")


def tile_outline(ds, z, tile, f):
    """Corners (5 x 2, closed) of a tile slice in the stored volume's plane grid at downsample f."""
    p = ds.plan.loc[(z, tile)]
    h, w = float(p["height"]), float(p["width"])
    c = np.array([[0, 0], [w, 0], [w, h], [0, h], [0, 0]], float)
    A = np.array([[p.a, p.b, p.tx], [p.c, p.d, p.ty]], float)
    v = c @ A[:, :2].T + A[:, 2] - np.asarray(ds.meta["origin_xy"], float)
    if ds.rotated:
        W, H = ds.meta["canvas_size_xy"]
        v = np.c_[W - 1 - v[:, 0], H - 1 - v[:, 1]]
    return v / f


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--volume", required=True)
    p.add_argument("out_png")
    p.add_argument("--scale", type=int, default=5, help="pyramid scale to draw (5: 256 nm for 8 nm data)")
    p.add_argument("--output-dir", help="the dataset's output_dir as mounted here, if not the config's")
    a = p.parse_args(argv)
    cfg = load_config(a.config, output_dir=a.output_dir, step_defaults=deep_merge(refmask.DEFAULTS, preview.DEFAULTS))
    ds = Dataset(cfg, a.volume)
    sl = pd.read_csv(step_path(cfg, "check", "slices.csv"), dtype={"tile": str})
    sl = sl[~sl["excluded"].astype(bool)]
    segs = []
    for seg, g in sl.groupby("segment"):
        zs = np.sort(g["z"].unique())
        segs.append((int(seg), int(zs[0]), int(zs[-1]), int(zs[len(zs) // 2]), g))
    f = 2 ** a.scale
    arr = omezarr.open_scale(ds.root, a.scale)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = min(3, len(segs))
    nrow = int(np.ceil(len(segs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 5.6 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (seg, z0, z1, zm, g) in zip(axes.ravel(), segs):
        img = np.asarray(arr[ds.planes[zm] // f].read().result())
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        rows = g[g["z"] == zm]
        for k, t in enumerate(sorted(rows["tile"])):
            v = tile_outline(ds, zm, t, f)
            ax.plot(v[:, 0], v[:, 1], color=COLOURS[k % len(COLOURS)], lw=1.4)
            cx, cy = v[:4].mean(0)
            ax.text(cx, cy, t, color=COLOURS[k % len(COLOURS)], fontsize=9, ha="center", va="center",
                    bbox=dict(facecolor="white", alpha=0.6, lw=0, pad=1))
        t0, t1 = g["timestamp"].min()[5:16].replace("T", " "), g["timestamp"].max()[5:16].replace("T", " ")
        nr, nc = g["tile_row"].nunique(), g["tile_col"].nunique()
        h, w = int(g["height"].median()), int(g["width"].median())
        ax.set_title(f"segment {seg}: z {z0}-{z1} ({t0} to {t1})\n{nr} x {nc} tiles of {w} x {h} px; "
                     f"shown: z {zm}", fontsize=10, loc="left")
    fig.suptitle(f"Acquisition segments (one tile layout each), {8 * f} nm", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(a.out_png, dpi=90)
    return 0


if __name__ == "__main__":
    sys.exit(main())
