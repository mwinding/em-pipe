"""Janelia-style intensity matching, step 1: brightness relations per patch pair of every tile overlap.

    python tools/experiments/janelia_bins.py CONFIG OUT_DIR Z[,Z...]|@ZFILE [--grid 8] [--min-pairs 2000]

As saalfeldlab/render's intensity-adjust client (AdjustBlock / IntensityMatcher), each tile is split
into a grid x grid array of patches and every pixel pair of an overlap is a match between the
patch it falls in on tile a and the patch on tile b. Pixel pairs are those of pipeline.overlaps
(32 nm, flat-fielded, registered with the stitch transforms, lightly smoothed), here with their
full-resolution positions in both tiles; per (patch_a, patch_b) one robust relation
va ~ gain * vb + offset (overlaps.fit_relation: MAD outlier rejection plays the part of Janelia's
RANSAC). A row with patch -1 is the whole overlap's relation (as in work/overlaps/relations.csv).

Writes OUT_DIR/z<z>.csv (columns COLUMNS). Array tasks (SLURM_ARRAY_TASK_ID) take every n-th slice;
slices whose CSV exists are skipped.
"""
import argparse
import itertools
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline import overlaps, shading, transforms  # noqa: E402
from pipeline.config import deep_merge, load_config  # noqa: E402
from pipeline.slices import StackCache, load_slices  # noqa: E402

COLUMNS = ["z", "a", "b", "pa", "pb", "gain", "offset", "n", "n_total", "rms", "cnr", "centre", "median_a",
           "xa", "ya", "xb", "yb"]


def patch_index(x, y, w, h, grid):
    """Patch (row * grid + col) of full-resolution tile pixels (x, y) of a w x h tile."""
    c = np.clip(np.floor((np.asarray(x) + 0.5) / w * grid), 0, grid - 1).astype(int)
    r = np.clip(np.floor((np.asarray(y) + 0.5) / h * grid), 0, grid - 1).astype(int)
    return r * grid + c


def pair_values_pos(cache, rec_a, rec_b, T_a, T_b, fields, o):
    """overlaps.pair_values with positions: (va, vb, xa, ya, xb, yb) on a's downsampled grid, or None.

    (xa, ya): each pair's tile-a full-resolution pixel; (xb, yb): the same point in tile b.
    """
    f, trim, sigma = int(o["factor"]), int(o["trim_px"]), float(o["smooth_px"])
    shape_a = (int(rec_a["height"]), int(rec_a["width"]))
    shape_b = (int(rec_b["height"]), int(rec_b["width"]))
    box_a = overlaps.overlap_box(T_a, shape_a, T_b, shape_b, trim, f)
    if box_a is None:
        return None
    a_in_b = transforms.compose(transforms.invert(T_b), T_a)
    pts = transforms.apply(a_in_b, [[box_a[0], box_a[1]], [box_a[2], box_a[1]], [box_a[0], box_a[3]],
                                    [box_a[2], box_a[3]]])
    pad = 2 * f
    bx0, by0 = max(0, math.floor(pts[:, 0].min()) - pad), max(0, math.floor(pts[:, 1].min()) - pad)
    bx1, by1 = min(shape_b[1], math.ceil(pts[:, 0].max()) + pad), min(shape_b[0], math.ceil(pts[:, 1].max()) + pad)
    bx1, by1 = bx0 + (bx1 - bx0) // f * f, by0 + (by1 - by0) // f * f
    if bx1 - bx0 < 4 * f or by1 - by0 < 4 * f:
        return None
    va, ok_a = overlaps._read(cache, rec_a, fields.get(rec_a["key"]), box_a, f)
    vb, ok_b = overlaps._read(cache, rec_b, fields.get(rec_b["key"]), (bx0, by0, bx1, by1), f)
    if sigma > 0:
        va, vb = cv2.GaussianBlur(va, (0, 0), sigma), cv2.GaussianBlur(vb, (0, 0), sigma)
        k = 2 * math.ceil(2 * sigma) + 1
        ok_a = cv2.erode(ok_a.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        ok_b = cv2.erode(ok_b.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    to_full_a = overlaps._ds_to_full(box_a[0], box_a[1], f)
    M = transforms.compose(transforms.invert(overlaps._ds_to_full(bx0, by0, f)), transforms.compose(a_in_b, to_full_a))
    size = (va.shape[1], va.shape[0])
    vb_on_a = cv2.warpAffine(vb, M, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ok_on_a = cv2.warpAffine(ok_b.astype(np.float32), M, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.999
    ok = ok_a & ok_on_a
    ii, jj = np.nonzero(ok)
    pa = transforms.apply(to_full_a, np.c_[jj, ii])
    pb = transforms.apply(a_in_b, pa)
    return va[ok], vb_on_a[ok], pa[:, 0], pa[:, 1], pb[:, 0], pb[:, 1]


def measure_slice(cache, recs, stitch, fields, o, grid, min_pairs):
    z = int(recs[0]["z"])
    rows = []
    for ra, rb in itertools.combinations(sorted(recs, key=lambda r: r["tile"]), 2):
        if (z, ra["tile"]) not in stitch or (z, rb["tile"]) not in stitch:
            continue
        got = pair_values_pos(cache, ra, rb, stitch[(z, ra["tile"])], stitch[(z, rb["tile"])], fields, o)
        if got is None:
            continue
        va, vb, xa, ya, xb, yb = got
        pa = patch_index(xa, ya, int(ra["width"]), int(ra["height"]), grid)
        pb = patch_index(xb, yb, int(rb["width"]), int(rb["height"]), grid)
        groups = [(-1, -1, np.ones(len(va), bool))]
        key = pa * grid * grid + pb
        for k in np.unique(key):
            groups.append((int(k // (grid * grid)), int(k % (grid * grid)), key == k))
        for p_a, p_b, sel in groups:
            if sel.sum() < (int(o["min_pixels"]) if p_a < 0 else min_pairs):
                continue
            rel = overlaps.fit_relation(va[sel], vb[sel], float(o["reject"]),
                                        min_pixels=int(o["min_pixels"]) if p_a < 0 else min_pairs // 2)
            if rel is None:
                continue
            gain, offset, n, rms, cnr = rel
            rows.append({"z": z, "a": ra["tile"], "b": rb["tile"], "pa": p_a, "pb": p_b, "gain": gain,
                         "offset": offset, "n": n, "n_total": int(sel.sum()), "rms": rms, "cnr": cnr,
                         "centre": float(np.median(vb[sel])), "median_a": float(np.median(va[sel])),
                         "xa": float(xa[sel].mean()), "ya": float(ya[sel].mean()),
                         "xb": float(xb[sel].mean()), "yb": float(yb[sel].mean())})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("out", type=Path)
    p.add_argument("zs")
    p.add_argument("--grid", type=int, default=8)
    p.add_argument("--min-pairs", type=int, default=2000)
    args = p.parse_args()
    text = Path(args.zs[1:]).read_text() if args.zs.startswith("@") else args.zs
    zs = sorted({int(v) for v in text.replace("\n", ",").split(",") if v.strip()})
    tid = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)) - int(os.environ.get("SLURM_ARRAY_TASK_MIN", 0))
    ntask = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    zs = zs[tid::ntask]
    cfg = load_config(args.config, step_defaults=deep_merge(overlaps.DEFAULTS, shading.DEFAULTS))
    o = cfg["overlaps"]
    slices = load_slices(cfg)
    stitch = transforms.read_csv(Path(cfg["output_dir"]) / "work" / "stitch" / "tiles.csv", ["z", "tile"])
    fields = shading.load(cfg)
    cache = StackCache(cfg["raw_dir"])
    args.out.mkdir(parents=True, exist_ok=True)
    for z in zs:
        path = args.out / f"z{z}.csv"
        if path.exists():
            continue
        t0 = time.monotonic()
        recs = slices[slices["z"] == z].to_dict("records")
        for r in recs:
            r["key"] = (int(r["segment"]), r["tile"])
        rows = measure_slice(cache, recs, stitch, fields, o, args.grid, args.min_pairs)
        tmp = path.with_suffix(".tmp")
        pd.DataFrame(rows, columns=COLUMNS).to_csv(tmp, index=False, float_format="%.6g")
        tmp.rename(path)
        print(f"z {z}: {len(rows)} relations in {time.monotonic() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
