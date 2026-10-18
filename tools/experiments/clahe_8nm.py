"""CLAHE computed the way the real display copy would be: on the full 8 nm plane, then downsampled
to 32 nm like the pyramid (experiment, Oct 2026).

    python tools/experiments/clahe_8nm.py CONFIG OLD_VOLUME SRC_DIR QC_DIR WORK_DIR Z[,Z...]

For each slice z: renders the whole 8 nm plane from the raw tiles with the current levels and
corrections (flat-field, patchgrid shapes when enabled; render.plane_meta on the canvas of
OLD_VOLUME, as tools/shading_before_after.py does at 32 nm), applies
render._clahe at 8 nm for every clip limit in CLIPS and tile size in TILES (full-resolution px, as
in render.clahe), and brings each result to 32 nm by two 2x2 means (pyramid.downsample's rounding,
in x-y only: the real pyramid also averages 4 slices in z). Writes
  WORK_DIR/clahe8_<clip>_<tile>/z<z>.tif   single 32 nm plane;
  WORK_DIR/v5_8nm/z<z>.tif                 the 8 nm plane without CLAHE, downsampled the same way;
  QC_DIR/clahe8nm_2.0_512/z<z>.tif         [frame 0 of SRC_DIR (original volume), clip 2.0 tile 512];
  WORK_DIR/check/z<z>.json                 v5_8nm vs frame 1 of SRC_DIR (the 32 nm render), timings,
                                           the 8 nm crop centre;
  WORK_DIR/crops8/z<z>.tif                 for z in CROP_Z: a CROP x CROP px 8 nm crop centred on a
                                           tile junction, [no CLAHE, every variant].
In an array job SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT split the z list.
Runs on the cluster (reads the raw tiles): sbatch --cpus-per-task=8 --mem=128G.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline import patchgrid, render, shading, transforms  # noqa: E402
from pipeline.config import deep_merge, load_config  # noqa: E402

CLIPS = (1.5, 2.0, 3.0)
TILES = (256, 512, 1024)
CROP_Z = {1000, 6000, 9000, 13000, 17688}
CROP = 2048


def ds2(a):
    """2x2 mean of a uint8 plane, rounded halves up; an odd last row/column is edge-padded (as pyramid)."""
    pad = [(0, n % 2) for n in a.shape]
    if any(p for _, p in pad):
        a = np.pad(a, pad, mode="edge")
    s = np.add(a[0::2], a[1::2], dtype=np.uint16)
    s = s[:, 0::2] + s[:, 1::2]
    return ((s + 2) >> 2).astype(np.uint8)


def junction(tiles, origin):
    """8 nm canvas (x, y) where four tiles meet: rows 0/1 and columns 0/1 of a 2x2 layout, rows 1/2
    and columns 0/1 of a 3x3 one (the seam 1-0|2-0)."""
    box = {t["tile"]: transforms.bbox(transforms.from_row(t), t["width"], t["height"]) for t in tiles}
    r = 1 if "2-0" in box else 0
    a, right, below = box[f"{r}-0"], box[f"{r}-1"], box[f"{r + 1}-0"]
    cx = (max(a[0], right[0]) + min(a[2], right[2])) / 2
    cy = (max(a[1], below[1]) + min(a[3], below[3])) / 2
    return int(round(cx - origin[0])), int(round(cy - origin[1]))


def write(path, frames, labels=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.tif")
    md = {"spacing": 0.032, "unit": "micron", "axes": "ZYX" if len(frames) > 1 else "YX"}
    if labels:
        md["Labels"] = labels
    tifffile.imwrite(tmp, np.stack(frames) if len(frames) > 1 else frames[0], imagej=True,
                     resolution=(1 / 0.032, 1 / 0.032), metadata=md)
    os.replace(tmp, path)


def main():
    cfg = load_config(sys.argv[1], step_defaults=deep_merge(deep_merge(render.DEFAULTS, shading.DEFAULTS),
                                                            patchgrid.DEFAULTS))
    old, src, qc, work = sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
    zs = [int(v) for v in sys.argv[6].split(",")]
    k, n = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)), int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    zs = zs[k::n]
    out = Path(cfg["output_dir"])
    old_meta = json.loads((out / "work" / "render" / old.removesuffix(".zarr").removesuffix(".ome") / "render.json").read_text())
    meta = render.plane_meta(cfg, old_meta, 1, clahe={"enabled": False})
    plan = render._tile_plan(cfg)
    plan = plan[plan["z"].isin(zs)]
    with ThreadPoolExecutor(8) as pool:
        r = render._Renderer(cfg, meta, plan, pool)
        for z in zs:
            t0 = time.time()
            val, valid = r._warp(z, *r._slice([pool.submit(r._tile, t) for t in r.tiles[z]]))
            plane = cv2.convertScaleAbs(val, alpha=255.0)
            del val
            t_render = time.time() - t0
            before, v5 = tifffile.imread(src / f"z{z}.tif")[:2]
            small = ds2(ds2(plane))
            hh, ww = min(small.shape[0], v5.shape[0]), min(small.shape[1], v5.shape[1])
            both = (small[:hh, :ww] > 0) & (v5[:hh, :ww] > 0)
            diff = np.abs(small[:hh, :ww].astype(np.int16) - v5[:hh, :ww])[both]
            write(work / "v5_8nm" / f"z{z}.tif", [small])
            cx, cy = junction(r.tiles[z], r.origin)
            y0, x0 = max(0, cy - CROP // 2), max(0, cx - CROP // 2)
            crop = np.s_[y0:y0 + CROP, x0:x0 + CROP]
            crops, names = [plane[crop].copy()], ["no CLAHE"]
            t1 = time.time()
            for clip in CLIPS:
                for tile in TILES:
                    c = render._clahe(plane.copy(), valid, clip, tile)
                    if z in CROP_Z:
                        crops.append(c[crop].copy())
                        names.append(f"CLAHE clip {clip} tile {tile} (8 nm)")
                    cs = ds2(ds2(c))
                    del c
                    write(work / f"clahe8_{clip}_{tile}" / f"z{z}.tif", [cs])
                    if (clip, tile) == (2.0, 512):
                        write(qc / "clahe8nm_2.0_512" / f"z{z}.tif", [before, cs[:before.shape[0], :before.shape[1]]],
                              [f"z {z}: original volume", f"z {z}: current pipeline, CLAHE clip 2.0 tile 512 at 8 nm, 2x2 means to 32 nm"])
            if z in CROP_Z:
                d = work / "crops8"
                d.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(d / f"z{z}.tif", np.stack(crops), imagej=True, resolution=(1 / 0.008, 1 / 0.008),
                                 metadata={"spacing": 0.008, "unit": "micron", "axes": "ZYX", "Labels": names})
            info = {"z": z, "render_s": round(t_render, 1), "clahe_s": round(time.time() - t1, 1),
                    "shape_8nm": list(plane.shape), "junction_xy_8nm": [cx, cy], "crop_x0y0_8nm": [x0, y0],
                    "v5_vs_8nm_px": int(both.sum()), "mean_abs_diff": round(float(diff.mean()), 3),
                    "p99_abs_diff": float(np.percentile(diff, 99)), "max_abs_diff": int(diff.max()),
                    "frac_gt_2": round(float((diff > 2).mean()), 5)}
            (work / "check").mkdir(parents=True, exist_ok=True)
            (work / "check" / f"z{z}.json").write_text(json.dumps(info))
            print(json.dumps(info), flush=True)
            del plane, valid


if __name__ == "__main__":
    main()
