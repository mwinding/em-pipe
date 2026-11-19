"""Display-window options side by side at 8 nm (35h), as one ImageJ TIFF stack.

Each option's levels come from the pipeline's own intensity step (smoothing, tile balancing,
per-slice correction); render uses the run's own plan otherwise, without CLAHE. Slices: either side
of the tile layout changes and the middle of each layout; rows: two fixed crops in the aligned frame;
columns: the options. Writes qc/normalisation_compare.tif and prints the crops' median grey.
"""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import copy
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, os.path.expandvars("${EM_PIPE_ROOT}"))
from pipeline import config as C, elastic, intensity, render, transforms  # noqa: E402
from pipeline.slices import load_slices  # noqa: E402

CROP, GAP = 700, 16
OPTIONS = {
    "1 current: all-pixel 0.5-99.5% -> 0-255": {"window": "percentile", "percentile_of": "all",
                                                "lo_percentile": 0.5, "hi_percentile": 99.5},
    "2 tissue 1-99% -> 0-255": {"window": "percentile", "percentile_of": "tissue", "lo_percentile": 1, "hi_percentile": 99},
    "3 tissue 5-95% -> 0-255": {"window": "percentile", "percentile_of": "tissue", "lo_percentile": 5, "hi_percentile": 95},
    "4 median-anchored: tissue median -> 140, p5-p95 over 160": {"window": "median", "target_median": 140,
                                                                 "target_span": 160},
}

cfg = C.load_config(sys.argv[1], step_defaults=C.step_defaults())
out = Path(cfg["output_dir"]) / "qc" / "normalisation_compare.tif"
plan = render._tile_plan(cfg)
levels = {}
for name, over in OPTIONS.items():
    c = copy.deepcopy(cfg)
    c["intensity"].update(over)
    levels[name] = intensity.compute_levels(c)[["z", "tile", "lo", "hi"]]
    print("levels:", name, flush=True)

sl = load_slices(cfg)
stamp = sl.groupby("z")["timestamp"].first()
changes = pd.read_csv(Path(cfg["output_dir"]) / "work" / "align" / "segment_changes.csv")
big = changes[changes["z_after"].isin([3279, 12529])]["z_after"].tolist()
zs = sorted({1500, 8000, 15000, 16500} | {z + d for z in big for d in (-20, -1, 0, 20)})
zs = [z for z in zs if z in set(plan["z"])]
seg_of = sl.drop_duplicates("z").set_index("z")["segment"]
here = plan[plan["z"] == 8000].set_index("tile")
crops = {}
for t in ("0-0", "1-0"):
    r = here.loc[t]
    crops[t] = transforms.apply(transforms.from_row(r), [[r["width"] / 2, r["height"] / 2]])[0]

settings = render._settings(cfg)
settings["clahe"] = {**settings["clahe"], "enabled": False}
fields = elastic.load(cfg)
sub = plan[plan["z"].isin(zs)].drop(columns=["lo", "hi"])
H = len(crops) * CROP + (len(crops) - 1) * GAP
Wd = len(OPTIONS) * CROP + (len(OPTIONS) - 1) * GAP
stack = np.full((len(zs), H, Wd), 128, np.uint8)
labels, med = [], []
with ThreadPoolExecutor(8) as pool:
    rend = {}
    for j, name in enumerate(OPTIONS):
        p = sub.merge(levels[name], on=["z", "tile"], how="left")
        for i, (t, xy) in enumerate(crops.items()):
            meta = {"downsample": 1, "origin_xy": [float(np.floor(xy[0] - CROP / 2)), float(np.floor(xy[1] - CROP / 2))],
                    "shape": [1, CROP, CROP], "settings": settings, "destreak": None, "elastic": fields}
            rend[(i, j)] = render._Renderer(cfg, meta, p, pool)
    for k, z in enumerate(zs):
        row = {"z": z}
        for (i, j), rr in rend.items():
            img = rr.plane([(int(z), 1.0)])
            stack[k, i * (CROP + GAP):i * (CROP + GAP) + CROP, j * (CROP + GAP):j * (CROP + GAP) + CROP] = img
            row[(j, i)] = float(np.median(img[img > 0])) if (img > 0).any() else np.nan
        med.append(row)
        labels.append(f"z {z} {stamp.loc[z]:%d %b %H:%M} (layout {int(seg_of.loc[z])}) | columns: "
                      + " | ".join(OPTIONS) + " | rows: crop at tile 0-0, 1-0 centres (z 8000)")
        print("slice", z, flush=True)
tifffile.imwrite(out, stack, imagej=True, resolution=(125.0, 125.0),
                 metadata={"spacing": 0.008, "unit": "micron", "axes": "ZYX", "Labels": labels})
m = pd.DataFrame(med).set_index("z")
print("median grey of the two crops (mean of both), per option:")
for j, name in enumerate(OPTIONS):
    v = m[[(j, 0), (j, 1)]].mean(axis=1)
    print(f"  {name}: " + "  ".join(f"z{z}:{x:.0f}" for z, x in v.items()))
print("->", out)
