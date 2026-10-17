"""Brightness steps across tile seams, in tissue and in plain resin, for before/after slice stacks.

    python tools/seam_steps.py CONFIG OLD_VOLUME Z[,Z...] NAME=DIR [NAME=DIR ...]

DIRs hold z<z>.tif stacks from tools/shading_before_after.py (frame 0: before, frame 1: after); the
before column comes from the first DIR. Which tile owns each output pixel is found by rendering the
slice with every tile replaced by a constant (so elastic warps and cut seams are exactly as in the
real render). For each pair of tiles that share a seam, pixels 8-30 px (32 nm) either side of it
are compared in 150 px stretches along the seam: step = median(B side) - median(A side), in display
grey levels, for tissue and for bright resin separately (background.mask on the before image).
A real step is systematic along the seam; the anatomy differs from stretch to stretch, so the
median over stretches is reported. Writes <output_dir>/qc/seam_steps.csv and prints a summary.
Runs on NEMO (reads the render plan): sbatch --cpus-per-task=8 --mem=64G --wrap "...".
"""
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import background, render, shading  # noqa: E402
from pipeline.config import deep_merge, load_config  # noqa: E402

NEAR, FAR, STRETCH, MIN_PX = 8, 30, 150, 150


def ownership(cfg, old, z):
    """Tile label (1..n, 0 = no data) of every output pixel of slice z, and the tile names."""
    out = Path(cfg["output_dir"])
    meta = json.loads((out / "work" / "render" / old.removesuffix(".zarr").removesuffix(".ome") / "render.json").read_text())
    f = 4
    w, h = meta["canvas_size_xy"]
    meta = {**meta, "downsample": f, "shape": [1, math.ceil(h / f), math.ceil(w / f)],
            "settings": {**meta["settings"], "downsample": f, "clahe": {"enabled": False}}, "shading": None}
    plan = render._tile_plan(cfg)
    plan = plan[plan["z"] == z]
    tiles = sorted(plan["tile"])
    code = {t: (k + 1) * 20 / 255 for k, t in enumerate(tiles)}
    keep = render._normalise
    render._normalise = lambda img, lo, hi: img
    try:
        with ThreadPoolExecutor(8) as pool:
            r = render._Renderer(cfg, meta, plan, pool)
            r.shading, r.destreak = {}, None
            r.cache.read = lambda t, rows, cols: np.full((rows.stop - rows.start, cols.stop - cols.start),
                                                         code[t["tile"]], np.float32)
            img = r.plane([(z, 1.0)])
    finally:
        render._normalise = keep
    return np.rint(img / 20).astype(np.uint8), tiles


def classes(before):
    """(tissue, resin) masks from the before image: background.mask at 128 nm, scaled to the valid pixels."""
    valid = before > 0
    small = cv2.resize(before, (before.shape[1] // 4, before.shape[0] // 4), interpolation=cv2.INTER_AREA)
    lo, hi = np.percentile(small[small > 0], (0.5, 99.5))
    s8 = np.clip((small.astype(float) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    s8[small == 0] = 0
    bg = background.mask(s8)
    bg = cv2.resize(bg.astype(np.uint8), (before.shape[1], before.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    bright = cv2.resize((s8 > 128).astype(np.uint8), (before.shape[1], before.shape[0]),
                        interpolation=cv2.INTER_NEAREST) > 0
    return ~bg & valid, bg & bright & valid


def steps(lab, tiles, images, tissue, resin, z):
    rows = []
    dist = {k: ndimage.distance_transform_edt(lab != k) for k in range(1, len(tiles) + 1)}
    for a in range(1, len(tiles) + 1):
        for b in range(a + 1, len(tiles) + 1):
            sa = (lab == a) & (dist[b] >= NEAR) & (dist[b] <= FAR)
            sb = (lab == b) & (dist[a] >= NEAR) & (dist[a] <= FAR)
            if sa.sum() < 2000 or sb.sum() < 2000:
                continue
            ys, xs = np.nonzero(sa | sb)
            along = ys if np.ptp(ys) > np.ptp(xs) else xs
            axis = 0 if np.ptp(ys) > np.ptp(xs) else 1
            for cls, m in (("tissue", tissue), ("resin", resin)):
                diffs = {name: [] for name in images}
                for s0 in range(along.min(), along.max() + 1, STRETCH):
                    band = np.zeros(lab.shape, bool)
                    if axis == 0:
                        band[s0:s0 + STRETCH] = True
                    else:
                        band[:, s0:s0 + STRETCH] = True
                    pa, pb = sa & band & m, sb & band & m
                    if pa.sum() < MIN_PX or pb.sum() < MIN_PX:
                        continue
                    for name, img in images.items():
                        diffs[name].append(float(np.median(img[pb])) - float(np.median(img[pa])))
                if diffs[next(iter(images))]:
                    rows.append({"z": z, "seam": f"{tiles[a - 1]}|{tiles[b - 1]}", "class": cls,
                                 "stretches": len(diffs[next(iter(images))]),
                                 **{name: round(float(np.median(d)), 1) for name, d in diffs.items()}})
    return rows


def main():
    cfg = load_config(sys.argv[1], step_defaults=deep_merge(render.DEFAULTS, shading.DEFAULTS))
    old, zs = sys.argv[2], [int(v) for v in sys.argv[3].split(",")]
    dirs = dict(a.split("=", 1) for a in sys.argv[4:])
    all_rows = []
    for z in zs:
        stacks = {name: tifffile.imread(Path(d) / f"z{z}.tif") for name, d in dirs.items()}
        before = stacks[next(iter(dirs))][0]
        images = {"before": before, **{name: s[1] for name, s in stacks.items()}}
        lab, tiles = ownership(cfg, old, z)
        lab = lab[:before.shape[0], :before.shape[1]]
        tissue, resin = classes(before)
        rows = steps(lab, tiles, images, tissue, resin, z)
        all_rows += rows
        print(pd.DataFrame(rows).to_string(index=False), flush=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(Path(cfg["output_dir"]) / "qc" / "seam_steps.csv", index=False)
    names = ["before", *dirs]
    print("\nmedian |step| over seams (grey levels of 255):")
    print(df.groupby("class")[names].agg(lambda s: round(float(s.abs().median()), 1)).to_string())
    print("\nseams with |step| > 10:")
    print(df.groupby("class")[names].agg(lambda s: int((s.abs() > 10).sum())).to_string(), f"(of {len(df) // 2} per class)")


if __name__ == "__main__":
    main()
