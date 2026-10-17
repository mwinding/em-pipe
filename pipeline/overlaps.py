"""Tile overlaps: the brightness relation between every pair of overlapping tiles of a slice,
measured on the raw tiles, for intensity's tile balance (``intensity.balance_source: overlaps``).

Per selected slice (every ``every``-th) and pair of tiles whose stitched footprints overlap: the
overlap is read from both raw tiles (minus ``trim_px`` at each tile's border), divided by the tiles'
shading fields when ``shading.enabled``, area-downsampled by ``factor`` (4: 32 nm, the overlap
then ~25-30 px wide), lightly smoothed (``smooth_px``, the same for both tiles, so noise and
sub-pixel misplacement stay symmetric) and tile b resampled onto tile a's grid with the stitch
transforms. On those pixel pairs (tissue and resin alike: resin is a clean reference for the
bright end) ``fit_relation`` finds v_a = gain * v_b + offset robustly: pairs off the line by more
than ``reject`` robust sigmas (a streak in one tile, a misplaced edge) are dropped, and the line is
the reduced major axis of the rest (symmetric in a and b, so noise in both doesn't flatten the
gain as an ordinary regression would). Overlaps with little contrast (e.g. only resin) pin the
offset but hardly the gain; ``cnr`` (inlier p10-p90 spread over the fit's rms) is kept so the
balance can weight the gain accordingly.

With ``patch_grid`` G > 0 the same pixel pairs are also split by where they lie in each tile, as in
saalfeldlab/render's intensity-adjust client: every tile is cut into G x G patches and each pair
belongs to (patch of a, patch of b) by its full-resolution positions in the two tiles; every
(patch_a, patch_b) bin with at least ``patch_min_pairs`` pairs gets its own ``fit_relation``
(MAD rejection plays the part of Janelia's RANSAC). These bin relations, which show how the
relation changes along a seam, are what ``patchgrid`` fits smooth within-tile maps to.

    run   (array over z-chunks of ``chunk_slices``): work/overlaps/chunks/chunk_<z0>-<z1>.csv (and
          chunk_<z0>-<z1>_patches.csv with patch_grid > 0), each with a .json of what it was
          measured from (settings, slice times, stitch, shading fields); a chunk whose inputs
          changed is measured again
    merge: work/overlaps/relations.csv (z, a, b, gain, offset, n, n_total, rms, cnr,
           median_a, median_b) and qc/overlaps.png; with patch_grid > 0 also
           work/overlaps/patches.csv (PATCH_COLUMNS: pa, pb the patches (row * G + col) of a and b,
           centre the bin's median of v_b, xa, ya / xb, yb its mean full-resolution position in
           tiles a / b) and patches.json (the grid)

The relative gain of the tiles (only their brightness differences matter) is solved by
``intensity``; this step only measures.
"""

import hashlib
import itertools
import json
import logging
import math
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import shading, transforms  # noqa: E402
from .cli import atomic_write, base_parser, my_chunks, setup, task_info  # noqa: E402
from .config import deep_merge, qc_path, step_dir, step_path  # noqa: E402
from .preview import chunk_name, z_chunks  # noqa: E402
from .slices import StackCache, load_slices  # noqa: E402

log = logging.getLogger(__name__)

DEFAULTS = {
    "overlaps": {
        "enabled": False,      # run_pipeline.sh: measure tile overlaps after align (for balance_source: overlaps)
        "every": 1,            # measure every N-th selected slice
        "factor": 4,           # area-downsampling of the overlaps before matching (4: 32 nm at 8 nm pixels)
        "trim_px": 16,         # full-resolution px left out at each tile's border (scan edge effects)
        "smooth_px": 1.0,      # Gaussian sigma (downsampled px) applied to both tiles' overlap
        "min_pixels": 500,     # pixel pairs (downsampled) a relation needs
        "reject": 3.0,         # pairs further than this many robust sigmas off the line are dropped
        "chunk_slices": 100,   # selected z per array-task chunk (fixed blocks of global z)
        "patch_grid": 0,       # > 0: also a relation per (patch of a, patch of b) on a patch_grid x patch_grid
                               # grid over each tile (for patchgrid); 0: off
        "patch_min_pairs": 2000,   # pixel pairs (downsampled) a patch bin needs
    },
}

SATURATED = 65535
COLUMNS = ["z", "a", "b", "gain", "offset", "n", "n_total", "rms", "cnr", "median_a", "median_b"]
PATCH_COLUMNS = ["z", "a", "b", "pa", "pb", "gain", "offset", "n", "n_total", "rms", "cnr", "centre", "median_a",
                 "xa", "ya", "xb", "yb"]
PATCH_KEYS = ("patch_grid", "patch_min_pairs")


def fit_relation(va, vb, reject=3.0, iterations=4, min_pixels=500):
    """(gain, offset, n_inliers, rms, cnr) with va ~ gain * vb + offset, or None.

    cnr: the inliers' p10-p90 spread in a over the fit's rms (contrast to noise): how well the
    pairs pin the gain (only resin: low; tissue: high).

    Starts from the medians and inter-quartile ranges, then alternates dropping pairs further than
    ``reject`` robust sigmas from the line and refitting the reduced major axis (slope = ratio of
    standard deviations) of the rest.
    """
    va, vb = np.asarray(va, np.float64), np.asarray(vb, np.float64)
    if len(va) < min_pixels:
        return None
    qa, qb = np.percentile(va, [25, 50, 75]), np.percentile(vb, [25, 50, 75])
    if qa[2] <= qa[0] or qb[2] <= qb[0]:
        return None
    gain = (qa[2] - qa[0]) / (qb[2] - qb[0])
    offset = qa[1] - gain * qb[1]
    keep = np.ones(len(va), bool)
    for _ in range(iterations):
        r = va - (gain * vb + offset)
        mid = np.median(r)
        sigma = 1.4826 * np.median(np.abs(r - mid))
        keep = np.abs(r - mid) <= reject * max(sigma, 1e-6)
        if keep.sum() < min_pixels:
            return None
        a, b = va[keep], vb[keep]
        sa, sb = a.std(), b.std()
        if sa <= 0 or sb <= 0:
            return None
        gain = sa / sb
        offset = a.mean() - gain * b.mean()
    a, b = va[keep], vb[keep]
    rms = float(np.sqrt(np.mean((a - (gain * b + offset)) ** 2)))
    p10, p90 = np.percentile(a, [10, 90])
    return float(gain), float(offset), int(keep.sum()), rms, float((p90 - p10) / max(rms, 1e-6))


def _ds_to_full(x0, y0, f):
    """Downsampled crop pixel (centre) -> full-resolution tile pixel."""
    return np.array([[f, 0.0, x0 + (f - 1) / 2], [0.0, f, y0 + (f - 1) / 2]])


def overlap_box(T_a, shape_a, T_b, shape_b, trim, f):
    """Tile a's pixel box (x0, y0, x1, y1), sides multiples of f, covered by tile b (both trimmed); or None."""
    (ha, wa), (hb, wb) = shape_a, shape_b
    b_in_a = transforms.compose(transforms.invert(T_a), T_b)
    pts = transforms.apply(b_in_a, [[trim, trim], [wb - 1 - trim, trim], [trim, hb - 1 - trim],
                                    [wb - 1 - trim, hb - 1 - trim]])
    x0, y0 = max(trim, math.ceil(pts[:, 0].min())), max(trim, math.ceil(pts[:, 1].min()))
    x1, y1 = min(wa - trim, math.floor(pts[:, 0].max()) + 1), min(ha - trim, math.floor(pts[:, 1].max()) + 1)
    x1, y1 = x0 + (x1 - x0) // f * f, y0 + (y1 - y0) // f * f
    if x1 - x0 < 4 * f or y1 - y0 < 4 * f:
        return None
    return x0, y0, x1, y1


def _read(cache, rec, field, box, f):
    """(downsampled values, valid mask) of a tile's pixel box, flat-fielded when a field is given."""
    x0, y0, x1, y1 = box
    img = cache.read(rec, slice(y0, y1), slice(x0, x1)).astype(np.float32)
    ok = ((img > 0) & (img < SATURATED)).astype(np.float32)
    if field is not None:
        img /= shading.gain_window(field, (int(rec["height"]), int(rec["width"])), slice(y0, y1), slice(x0, x1))
    size = ((x1 - x0) // f, (y1 - y0) // f)
    return (cv2.resize(img, size, interpolation=cv2.INTER_AREA),
            cv2.resize(ok, size, interpolation=cv2.INTER_AREA) > 0.999)


def pair_values(cache, rec_a, rec_b, T_a, T_b, fields, o, positions=False):
    """Pixel pairs (va, vb) of the overlap of tiles a and b, on a's downsampled grid; None if no overlap.

    positions: also each pair's full-resolution pixel in tile a (xa, ya) and the same point in
    tile b (xb, yb): (va, vb, xa, ya, xb, yb).
    """
    f, trim, sigma = int(o["factor"]), int(o["trim_px"]), float(o["smooth_px"])
    shape_a = (int(rec_a["height"]), int(rec_a["width"]))
    shape_b = (int(rec_b["height"]), int(rec_b["width"]))
    box_a = overlap_box(T_a, shape_a, T_b, shape_b, trim, f)
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
    va, ok_a = _read(cache, rec_a, fields.get(rec_a["key"]), box_a, f)
    vb, ok_b = _read(cache, rec_b, fields.get(rec_b["key"]), (bx0, by0, bx1, by1), f)
    if sigma > 0:
        va, vb = cv2.GaussianBlur(va, (0, 0), sigma), cv2.GaussianBlur(vb, (0, 0), sigma)
        k = 2 * math.ceil(2 * sigma) + 1
        ok_a = cv2.erode(ok_a.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        ok_b = cv2.erode(ok_b.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    # a's downsampled pixel -> a full res -> b full res -> b's downsampled pixel
    to_full_a = _ds_to_full(box_a[0], box_a[1], f)
    M = transforms.compose(transforms.invert(_ds_to_full(bx0, by0, f)), transforms.compose(a_in_b, to_full_a))
    size = (va.shape[1], va.shape[0])
    vb_on_a = cv2.warpAffine(vb, M, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ok_on_a = cv2.warpAffine(ok_b.astype(np.float32), M, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.999
    ok = ok_a & ok_on_a
    if not positions:
        return va[ok], vb_on_a[ok]
    ii, jj = np.nonzero(ok)   # the same (row-major) order as va[ok]
    pa = transforms.apply(to_full_a, np.c_[jj, ii])
    pb = transforms.apply(a_in_b, pa)
    return va[ok], vb_on_a[ok], pa[:, 0], pa[:, 1], pb[:, 0], pb[:, 1]


def patch_index(x, y, w, h, grid):
    """Patch (row * grid + col) of full-resolution pixels (x, y) of a w x h tile on a grid x grid grid."""
    c = np.clip(np.floor((np.asarray(x) + 0.5) / w * grid), 0, grid - 1).astype(int)
    r = np.clip(np.floor((np.asarray(y) + 0.5) / h * grid), 0, grid - 1).astype(int)
    return r * grid + c


def patch_relations(va, vb, xa, ya, xb, yb, shape_a, shape_b, o):
    """Rows (dicts, PATCH_COLUMNS without z, a, b) of one overlap's (patch_a, patch_b) bins."""
    grid, min_pairs = int(o["patch_grid"]), int(o["patch_min_pairs"])
    pa = patch_index(xa, ya, shape_a[1], shape_a[0], grid)
    pb = patch_index(xb, yb, shape_b[1], shape_b[0], grid)
    key = pa * grid * grid + pb
    order = np.argsort(key, kind="stable")
    keys, starts = np.unique(key[order], return_index=True)
    rows = []
    for k, sel in zip(keys, np.split(order, starts[1:])):
        if len(sel) < min_pairs:
            continue
        rel = fit_relation(va[sel], vb[sel], float(o["reject"]), min_pixels=min_pairs // 2)
        if rel is None:
            continue
        gain, offset, n, rms, cnr = rel
        rows.append({"pa": int(k // (grid * grid)), "pb": int(k % (grid * grid)), "gain": gain, "offset": offset,
                     "n": n, "n_total": len(sel), "rms": rms, "cnr": cnr, "centre": float(np.median(vb[sel])),
                     "median_a": float(np.median(va[sel])), "xa": float(xa[sel].mean()), "ya": float(ya[sel].mean()),
                     "xb": float(xb[sel].mean()), "yb": float(yb[sel].mean())})
    return rows


def measure_slice(cache, recs, stitch, fields, o):
    """(rows (dicts, COLUMNS), patch rows (dicts, PATCH_COLUMNS)) for one slice's overlapping tile
    pairs; patch rows only with ``patch_grid`` > 0."""
    z = int(recs[0]["z"])
    patches = int(o["patch_grid"]) > 0
    rows, patch_rows = [], []
    for ra, rb in itertools.combinations(sorted(recs, key=lambda r: r["tile"]), 2):
        if (z, ra["tile"]) not in stitch or (z, rb["tile"]) not in stitch:
            continue
        got = pair_values(cache, ra, rb, stitch[(z, ra["tile"])], stitch[(z, rb["tile"])], fields, o, positions=patches)
        if got is None:
            continue
        if patches:
            shape_a, shape_b = (int(ra["height"]), int(ra["width"])), (int(rb["height"]), int(rb["width"]))
            patch_rows += [{"z": z, "a": ra["tile"], "b": rb["tile"], **p}
                           for p in patch_relations(*got, shape_a, shape_b, o)]
        va, vb = got[:2]
        rel = fit_relation(va, vb, float(o["reject"]), min_pixels=int(o["min_pixels"]))
        if rel is None:
            continue
        gain, offset, n, rms, cnr = rel
        rows.append({"z": z, "a": ra["tile"], "b": rb["tile"], "gain": gain, "offset": offset, "n": n,
                     "n_total": len(va), "rms": rms, "cnr": cnr,
                     "median_a": float(np.median(va)), "median_b": float(np.median(vb))})
    return rows, patch_rows


def _selected(slices, every):
    zs = sorted(slices["z"].unique())
    return set(zs[::max(1, int(every))])


def _identity(cfg, zs, slices, stitch_rows):
    """What a chunk is measured from: settings, slice times, stitch transforms, shading fields."""
    times = slices[slices["z"].isin(zs)].groupby("z")["timestamp"].first()
    settings = dict(cfg["overlaps"])
    if not int(settings.get("patch_grid") or 0):
        # Without patches: the same identity as chunks measured before the patch settings existed.
        settings = {k: v for k, v in settings.items() if k not in PATCH_KEYS}
    return {"settings": settings, "stamps": [f"{z}:{pd.Timestamp(t).isoformat()}" for z, t in times.items()],
            "stitch": hashlib.sha256(stitch_rows.to_csv(index=False).encode()).hexdigest(),
            "shading": shading.digest(cfg)}


def run(cfg, args):
    o = cfg["overlaps"]
    slices = load_slices(cfg)
    stitch_df = pd.read_csv(step_path(cfg, "stitch", "tiles.csv"), dtype={"tile": str})
    stitch = transforms.read_csv(step_path(cfg, "stitch", "tiles.csv"), ["z", "tile"])
    fields = shading.load(cfg)
    keep = _selected(slices, o["every"])
    cache = StackCache(cfg["raw_dir"])
    mine = my_chunks(z_chunks(slices, o["chunk_slices"]), *task_info(args))
    log.info("%d chunks for this task", len(mine))
    for zs in mine:
        zs = [z for z in zs if z in keep]
        if not zs:
            continue
        name = chunk_name(zs)
        out = step_dir(cfg, "overlaps", "chunks", f"chunk_{name}.csv")
        ident = _identity(cfg, zs, slices, stitch_df[stitch_df["z"].isin(zs)])
        meta = out.with_suffix(".json")
        if (out.exists() and meta.exists() and not args.overwrite and json.loads(meta.read_text()) == ident
                and (int(o["patch_grid"]) <= 0 or _patch_path(out).exists())):   # as merge checks it
            log.info("chunk %s is current, skipping", name)
            continue
        out.unlink(missing_ok=True)
        rows, patch_rows = [], []
        for z in zs:
            recs = slices[slices["z"] == z].to_dict("records")
            for r in recs:
                r["key"] = (int(r["segment"]), r["tile"])
            got = measure_slice(cache, recs, stitch, fields, o)
            rows += got[0]
            patch_rows += got[1]
        atomic_write(meta, lambda p: Path(p).write_text(json.dumps(ident)))
        if int(o["patch_grid"]) > 0:
            atomic_write(_patch_path(out), lambda p: pd.DataFrame(patch_rows, columns=PATCH_COLUMNS).to_csv(
                p, index=False, float_format="%.8g"))
        atomic_write(out, lambda p: pd.DataFrame(rows, columns=COLUMNS).to_csv(p, index=False))   # last: marks done
        log.info("chunk %s: %d relations (%d patch bins) over %d slices", name, len(rows), len(patch_rows), len(zs))
    return 0


def _patch_path(chunk):
    return chunk.with_name(chunk.stem + "_patches.csv")


def merge(cfg, args):
    o = cfg["overlaps"]
    slices = load_slices(cfg)
    keep = _selected(slices, o["every"])
    stitch_df = pd.read_csv(step_path(cfg, "stitch", "tiles.csv"), dtype={"tile": str})
    grid = int(o["patch_grid"])
    parts, patch_parts, bad = [], [], []
    for zs in z_chunks(slices, o["chunk_slices"]):
        zs = [z for z in zs if z in keep]
        if not zs:
            continue
        name = chunk_name(zs)
        path = step_path(cfg, "overlaps", "chunks", f"chunk_{name}.csv")
        meta = path.with_suffix(".json")
        ident = _identity(cfg, zs, slices, stitch_df[stitch_df["z"].isin(zs)])
        if (not path.exists() or not meta.exists() or json.loads(meta.read_text()) != ident
                or (grid > 0 and not _patch_path(path).exists())):
            bad.append(name)
            continue
        parts.append(pd.read_csv(path, dtype={"a": str, "b": str}))
        if grid > 0:
            patch_parts.append(pd.read_csv(_patch_path(path), dtype={"a": str, "b": str}))
    if bad:
        raise RuntimeError(f"{len(bad)} overlap chunks missing or out of date (first: {bad[0]}); run 'overlaps run'")
    rel = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLUMNS)
    atomic_write(step_dir(cfg, "overlaps", "relations.csv"), lambda p: rel.to_csv(p, index=False, float_format="%.6g"))
    plot(rel, qc_path(cfg, "overlaps.png"))
    log.info("%d relations over %d slices", len(rel), rel["z"].nunique() if len(rel) else 0)
    if grid > 0:
        bins = pd.concat(patch_parts, ignore_index=True) if patch_parts else pd.DataFrame(columns=PATCH_COLUMNS)
        atomic_write(step_dir(cfg, "overlaps", "patches.json"),
                     lambda p: Path(p).write_text(json.dumps({k: o[k] for k in PATCH_KEYS})))
        # 8 digits: a gain's rounding error times the data's distance from 0 (~37000) stays far below a grey.
        atomic_write(step_dir(cfg, "overlaps", "patches.csv"),
                     lambda p: bins.to_csv(p, index=False, float_format="%.8g"))
        log.info("%d patch bins (grid %d) over %d slices", len(bins), grid, bins["z"].nunique() if len(bins) else 0)
    return 0


def plot(rel, path):
    if rel.empty:
        return
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 9), layout="constrained")
    plotted = False
    for (a, b), g in rel.groupby(["a", "b"]):
        if len(g) < 10:
            continue
        lab = f"{a}|{b}"
        axes[0].plot(g["z"], g["gain"], ".", ms=1, label=lab)
        axes[1].plot(g["z"], 100 * (g["median_a"] - g["median_b"]) / g["median_b"], ".", ms=1, label=lab)
        axes[2].plot(g["z"], 100 * g["rms"] / g["median_a"], ".", ms=1, label=lab)
        plotted = True
    axes[0].set_ylabel("gain a / b")
    axes[1].set_ylabel("median a vs b in overlap (%)")
    axes[2].set_ylabel("fit rms (% of median)")
    axes[2].set_xlabel("z")
    if plotted:
        axes[0].legend(fontsize=6, ncol=6, markerscale=8)
    fig.suptitle("Tile overlaps: brightness relation of each overlapping tile pair (raw, flat-fielded)")
    atomic_write(path, lambda p: fig.savefig(p, dpi=90))
    plt.close(fig)


def main(argv=None):
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("command", choices=["run", "merge"])
    args = p.parse_args(argv)
    cfg = setup(args, deep_merge(DEFAULTS, shading.DEFAULTS))
    return run(cfg, args) if args.command == "run" else merge(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
