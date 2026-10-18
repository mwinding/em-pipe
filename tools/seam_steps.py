"""Brightness steps across tile seams, in tissue and in plain resin, for before/after slice stacks.

    python tools/seam_steps.py [--out CSV] CONFIG OLD_VOLUME Z[,Z...]|@ZFILE NAME=DIR [NAME=DIR ...]
    python tools/seam_steps.py --merge [--out CSV] CONFIG

DIRs hold z<z>.tif stacks of 32 nm planes as the renderer makes them (unrotated), e.g. from
tools/shading_before_after.py (frame 0: before, frame 1: after); the before column comes from the
first DIR. ZFILE lists z separated by commas, spaces or newlines.

Which tile owns each output pixel is found by rendering the slice from OLD_VOLUME's plan (its
render.json: canvas, elastic fields) with every tile replaced by a constant code, so elastic warps
and seams are where the real render puts them (to within a pixel). That render always blends cut
(in a seam or feather render the owner is then the tile with the most weight) and samples the
codes nearest wherever it resamples them: the elastic remap of slices with an elastic field (e.g.
all 3x3 slices of 35h) and the affine warp of non-integer tile transforms. Bilinear sampling, as in
the real render, mixes the codes of two tiles (or of a tile and no data) along every tile boundary,
and the rounded mixture is another tile's label: 1-2 px lines of stray labels (~26,000 px per 3x3
slice of 35h) that put pixels far from a seam into its bands (1-0|2-0 read +25..+45 grey levels
instead of ~+1.5) and stretch a seam's extent across it, so it was cut into stretches the wrong
way (many 3x3 vertical seams kept 1 stretch). As a safety net, label pixels that a 5 x 5 opening
of their own label removes become no data; the count is printed per slice (a few px, trimmed where
three tiles meet; hundreds would mean stray labels again).

For each pair of tiles that share a seam, pixels 8-30 px (32 nm) either side of it are compared in
150 px stretches along the seam: step = median(B side) - median(A side), in display grey levels,
for tissue and for bright resin separately (background.mask on the before image). A real step is
systematic along the seam; the anatomy differs from stretch to stretch, so the median over
stretches is reported. Writes CSV (default <output_dir>/qc/seam_steps.csv; columns z, seam, class,
stretches, before, one per NAME) and prints a summary.

Runs on the cluster (reads the render plan): sbatch --cpus-per-task=8 --mem=64G --wrap "...". In parallel:
sbatch --array=0-<n-1> ...; task k of n takes every n-th z from the k-th and writes
<CSV stem>.<run>.part<k>of<n>.csv, run = a short hash of OLD_VOLUME, the whole z list and the
NAME=DIR list. Then --merge (e.g. sbatch --dependency=afterok:<array job>) checks that the parts
are all n of one run (a part left over from another run, e.g. with other z, is refused, not
merged), writes CSV with their rows in z order and deletes the parts.
"""
import argparse
import hashlib
import json
import math
import os
import re
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
from pipeline.cli import atomic_write  # noqa: E402
from pipeline.config import deep_merge, load_config  # noqa: E402

NEAR, FAR, STRETCH, MIN_PX = 8, 30, 150, 150
OPEN = 5   # clean-up: label pixels that an OPEN x OPEN opening of their own label removes become 0
COLUMNS = ["z", "seam", "class", "stretches"]   # then one step column per image: before, NAME...


class _NearestCV2:
    """cv2 as render sees it during the ownership render: remap and warpAffine sample nearest."""

    def __getattr__(self, name):
        return getattr(cv2, name)

    @staticmethod
    def remap(src, map1, map2, interpolation, **kw):
        return cv2.remap(src, map1, map2, cv2.INTER_NEAREST, **kw)

    @staticmethod
    def warpAffine(src, M, dsize, **kw):
        return cv2.warpAffine(src, M, dsize, **{**kw, "flags": cv2.INTER_NEAREST})


def ownership(cfg, old, z, plan=None):
    """Tile label (1..n, 0 = no data) of every 32 nm output pixel of slice z, and the tile names.

    plan: render._tile_plan(cfg), made here if not given (pass it in for many slices).
    """
    out = Path(cfg["output_dir"])
    meta = json.loads((out / "work" / "render" / old.removesuffix(".zarr").removesuffix(".ome") / "render.json").read_text())
    f = 4
    w, h = meta["canvas_size_xy"]
    # No grey-level corrections: the codes must reach the plane unchanged.
    meta = {**meta, "downsample": f, "shape": [1, math.ceil(h / f), math.ceil(w / f)], "shading": None,
            "patchgrid": None, "destreak": None,
            "settings": {**meta["settings"], "downsample": f, "blend": "cut", "clahe": {"enabled": False}}}
    plan = render._tile_plan(cfg) if plan is None else plan
    plan = plan[plan["z"] == z].assign(blend="cut", lo=0.0, hi=1.0)   # levels 0..1: the codes pass unchanged
    tiles = sorted(plan["tile"])
    step = 255 // max(len(tiles), 1)   # label k renders as k * step grey levels
    code = {t: (k + 1) * step / 255 for k, t in enumerate(tiles)}
    keep, render.cv2 = render.cv2, _NearestCV2()
    try:
        with ThreadPoolExecutor(8) as pool:
            r = render._Renderer(cfg, meta, plan, pool)
            r.cache.read = lambda t, rows, cols: np.full((rows.stop - rows.start, cols.stop - cols.start),
                                                         code[t["tile"]], np.float32)
            img = r.plane([(z, 1.0)])
    finally:
        render.cv2 = keep
    return np.rint(img / step).astype(np.uint8), tiles


def clean(lab):
    """Labels with pixels that an OPEN x OPEN opening of their own label removes set to 0 (no data)."""
    out = lab.copy()
    kernel = np.ones((OPEN, OPEN), np.uint8)
    for k in np.unique(lab[lab > 0]):
        m = (lab == k).view(np.uint8)
        out[(m > cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel))] = 0
    return out


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


def z_list(arg):
    """Sorted distinct z of 'Z[,Z...]' or '@ZFILE' (z separated by commas, spaces or newlines)."""
    text = Path(arg[1:]).read_text() if arg.startswith("@") else arg
    return sorted({int(v) for v in re.split(r"[,\s]+", text) if v})


def array_task():
    """(k, n): this Slurm array task's index from 0 and the array's task count; None outside an array."""
    if "SLURM_ARRAY_TASK_ID" not in os.environ:
        return None
    k = int(os.environ["SLURM_ARRAY_TASK_ID"]) - int(os.environ.get("SLURM_ARRAY_TASK_MIN", 0))
    return k, int(os.environ["SLURM_ARRAY_TASK_COUNT"])


def run_id(old, zs, dirs):
    """Identifies an array run: a short hash of OLD_VOLUME, the whole z list and the NAME=DIR list."""
    return hashlib.sha256(json.dumps([old, [int(z) for z in zs], list(dirs)]).encode()).hexdigest()[:10]


def part_path(out, run, k, n):
    return out.with_name(f"{out.stem}.{run}.part{k}of{n}.csv")


def merge(out):
    """Rows of all n array-task parts of ``out`` (one run), written to ``out`` in z order; the parts are deleted."""
    pattern = re.compile(rf"{re.escape(out.stem)}\.([0-9a-f]+)\.part(\d+)of(\d+)\.csv")
    parts = {(m[1], int(m[2]), int(m[3])): p for p in out.parent.glob("*.csv") if (m := pattern.fullmatch(p.name))}
    runs = {(run, n) for run, _, n in parts}
    if len(runs) != 1:
        raise SystemExit(f"expected the parts of one array run next to {out}, found "
                         f"{sorted(p.name for p in parts.values()) or 'none'}")
    run, n = runs.pop()
    missing = sorted(set(range(n)) - {k for _, k, _ in parts})
    if missing:
        raise SystemExit(f"{out}: the parts of array tasks {missing} (of {n}) are missing")
    df = pd.concat([pd.read_csv(parts[run, k, n]) for k in range(n)], ignore_index=True)   # one run: same columns
    df = df.sort_values("z", kind="stable", ignore_index=True)   # each z's rows come from one part, in order
    atomic_write(out, lambda p: df.to_csv(p, index=False))
    for p in parts.values():
        p.unlink()
    return df


def summary(df):
    names = [c for c in df.columns if c not in COLUMNS]
    if df.empty:
        print("\nno seam measured")
        return
    print("\nmedian |step| over seams (grey levels of 255):")
    print(df.groupby("class")[names].agg(lambda s: round(float(s.abs().median()), 1)).to_string())
    print("\nseams with |step| > 10, of n:")
    big = df.groupby("class")[names].agg(lambda s: int((s.abs() > 10).sum()))
    print(big.assign(n=df.groupby("class").size()).to_string())


def main(argv=None):
    p = argparse.ArgumentParser(description="Brightness steps across tile seams (see the module docstring).")
    p.add_argument("config")
    p.add_argument("old_volume", nargs="?", help="the render whose plan (render.json) the stacks share")
    p.add_argument("z", nargs="?", help="Z[,Z...] or @ZFILE")
    p.add_argument("dirs", nargs="*", metavar="NAME=DIR")
    p.add_argument("--out", type=Path, help="output CSV (default <output_dir>/qc/seam_steps.csv)")
    p.add_argument("--merge", action="store_true", help="merge the array tasks' parts of the output CSV")
    a = p.parse_args(argv)
    if not a.merge and not a.dirs:
        p.error("OLD_VOLUME, Z and at least one NAME=DIR are required (or --merge)")
    cfg = load_config(a.config, step_defaults=deep_merge(render.DEFAULTS, shading.DEFAULTS))
    out = a.out or Path(cfg["output_dir"]) / "qc" / "seam_steps.csv"
    if a.merge:
        summary(merge(out))
        return
    zs = z_list(a.z)
    task = array_task()
    if task:
        k, n = task
        out = part_path(out, run_id(a.old_volume, zs, a.dirs), k, n)   # the whole z list: the same in every task
        zs = zs[k::n]
    dirs = dict(d.split("=", 1) for d in a.dirs)
    plan = render._tile_plan(cfg)
    all_rows = []
    for z in zs:
        stacks = {name: tifffile.imread(Path(d) / f"z{z}.tif") for name, d in dirs.items()}
        before = stacks[next(iter(dirs))][0]
        images = {"before": before, **{name: s[1] for name, s in stacks.items()}}
        lab, tiles = ownership(cfg, a.old_volume, z, plan)
        lab = lab[:before.shape[0], :before.shape[1]]
        tidy = clean(lab)
        print(f"z {z}: {int(np.count_nonzero(tidy != lab))} stray ownership px dropped", flush=True)
        tissue, resin = classes(before)
        rows = steps(tidy, tiles, images, tissue, resin, z)
        all_rows += rows
        print(pd.DataFrame(rows).to_string(index=False), flush=True)
    df = pd.DataFrame(all_rows, columns=[*COLUMNS, "before", *dirs])
    atomic_write(out, lambda path: df.to_csv(path, index=False))
    print(f"\nwrote {out}")
    summary(df)


if __name__ == "__main__":
    main()
