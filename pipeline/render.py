"""Render: apply every correction once and write the volume's scale 0 as OME-Zarr.

``init`` (single job) freezes the output: per selected (z, tile) the composite transform
tile pixel -> aligned (``align ∘ stitch``) and the intensity levels go to ``render/tiles.csv``
(columns z, tile, file, index, height, width, a, b, tx, c, d, ty, lo, hi); the canvas, voxel
size, source slices of every output plane and the pixel-affecting settings go to
``render/render.json``; and the empty multiscale volume is created. ``run`` (array over
z-slabs of ``render.slab`` planes = one shard) renders slabs from that frozen plan only, so a
config edit between init and run cannot mix geometries; re-running init detects changed
inputs or settings via the digest in render.json. A finished slab leaves
``render/done/slab_{k:06d}`` holding that digest; markers of any other plan do not count.

Geometry: output pixel (row i, col j) at downsample f is the mean of the aligned pixels
``origin_xy + (f*j, f*i) + [0, f)``. Pixel centres sit at integer coordinates (as in OpenCV).
"""

import hashlib
import json
import logging
import math
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import omezarr, transforms
from .cli import atomic_write, base_parser, my_chunks, setup, task_info
from .config import deep_merge, get, step_dir
from .slices import StackCache, load_slices, z_values

log = logging.getLogger(__name__)

DEFAULTS = {
    "render": {
        "name": "volume.ome.zarr",  # output folder under output_dir/render/
        "downsample": 1,            # integer binning factor f in x, y and z (1 = full resolution)
        "bbox": None,               # [xmin, ymin, xmax, ymax] in aligned px to render a region; null = all
        "integer_shifts": True,     # round translation-only transforms to whole px (no interpolation blur)
        "blend_px": 256,            # feathering ramp width from each tile edge, full-resolution px
        "slab": 64,                 # planes per run work unit = shard z size
        "chunk": [64, 64, 64],      # inner chunk (z, y, x) that neuroglancer fetches
        "shard_xy": 1024,           # shard size in y and x
        "num_scales": 7,            # pyramid levels s0..s{n-1}; render writes s0, pyramid the rest
        "compression": "gzip",      # none | gzip | zstd | blosc; gzip is safe for neuroglancer
        "clahe": {
            "enabled": True,        # local contrast equalisation of each output plane
            "clip_limit": 2.0,      # cv2 CLAHE clip limit
            "tile_px": 512,         # CLAHE tile size in full-resolution px
        },
        "threads": 8,               # tiles rendered concurrently (does not change the output)
    }
}


# ----- init -----------------------------------------------------------------------------

def _settings(cfg):
    """Validated render settings that shape the output (all but ``threads``)."""
    r = {k: v for k, v in cfg["render"].items() if k != "threads"}
    name = str(r["name"])
    if Path(name).name != name or not name.endswith(".zarr"):
        raise ValueError(f"render.name must be a folder name ending in .zarr, got {name!r}")
    f = r["downsample"]
    if isinstance(f, bool) or not isinstance(f, int) or f < 1:
        raise ValueError(f"render.downsample must be an integer >= 1, got {f!r}")
    if r["compression"] not in omezarr.COMPRESSORS:
        raise ValueError(f"render.compression must be one of {sorted(omezarr.COMPRESSORS)}")
    b = r["bbox"]
    if b is not None and (len(b) != 4 or b[2] <= b[0] or b[3] <= b[1]):
        raise ValueError(f"render.bbox must be [xmin, ymin, xmax, ymax] with max > min, got {b}")
    return r


def _require(path, step):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run the {step} step first")
    return path


# Absolute tolerances only: np.allclose's default rtol=1e-5 (also in transforms.is_translation)
# accepts a 0.1 px offset at x = 10^4 px, or a 1e-5 scale (0.14 px across a real tile).
def _whole(v, tol=1e-6):
    """True if every value is an integer to within ``tol``."""
    v = np.asarray(v, float)
    return bool(np.all(np.abs(v - np.round(v)) <= tol))


def _is_shift(T):
    """True if the 2x3 transform is a pure translation."""
    return bool(np.all(np.abs(np.asarray(T, float)[:, :2] - np.eye(2)) <= 1e-9))


def _tile_plan(cfg):
    """Per selected (z, tile): file, index, size, transform tile -> aligned, levels lo/hi."""
    sl = load_slices(cfg)
    if sl.empty:
        raise ValueError("no slices selected")
    out = Path(cfg["output_dir"])
    stitch = transforms.read_csv(_require(out / "stitch" / "tiles.csv", "stitch"), ["z", "tile"])
    align = transforms.read_csv(_require(out / "align" / "transforms.csv", "align"), ["z"])
    keys = list(zip(sl["z"], sl["tile"]))
    missing = [k for k in keys if k not in stitch or k[0] not in align]
    if missing:
        raise ValueError(f"{len(missing)} selected (z, tile) lack a stitch or align transform, "
                         f"e.g. {missing[:3]}: re-run stitch merge / align solve")
    mats = []
    for z, tile in keys:
        T = transforms.compose(align[z], stitch[(z, tile)])
        if cfg["render"]["integer_shifts"] and _is_shift(T):
            # floor(x + 0.5), not rint: half-pixel shifts must round the same way for every z.
            T = transforms.translation(*np.floor(T[:, 2] + 0.5))
        mats.append(T)
    plan = sl[["z", "tile", "file", "index", "height", "width"]].copy()
    plan[transforms.COLUMNS] = np.array(mats).reshape(-1, 6)
    levels = out / "intensity" / "levels.csv"
    if levels.exists():
        lv = pd.read_csv(levels, dtype={"tile": str})[["z", "tile", "lo", "hi"]]
        plan = plan.merge(lv, on=["z", "tile"], how="left", validate="many_to_one")
    else:
        plan["lo"] = plan["hi"] = np.nan
    n = int(plan["lo"].isna().sum())
    if n:
        log.warning("%s has no levels for %d of %d (z, tile): those tile slices are scaled to their "
                    "own p0.5/p99.5", levels, n, len(plan))
    return plan


def _canvas(plan, bbox):
    """(x0, y0, width, height) of the full-resolution canvas in aligned px."""
    if bbox is None:
        boxes = np.array([transforms.bbox(transforms.from_row(r), r["width"], r["height"])
                          for r in plan.to_dict("records")])
        bbox = (*boxes[:, :2].min(axis=0), *boxes[:, 2:].max(axis=0))
        # One stray transform (e.g. a failed alignment) would silently inflate every plane.
        by_z = pd.DataFrame(boxes, columns=["x0", "y0", "x1", "y1"]).groupby(plan["z"].to_numpy())
        extent = np.c_[by_z["x1"].max() - by_z["x0"].min(), by_z["y1"].max() - by_z["y0"].min()]
        typical = np.median(extent, axis=0)
        size = np.array(bbox[2:]) - bbox[:2]
        if np.any(size > 1.5 * typical):
            log.warning("canvas %s px (x, y) is over 1.5x the median slice extent %s: check "
                        "align/transforms.csv for outliers, or set render.bbox", size.round(), typical.round())
    x0, y0 = math.floor(bbox[0]), math.floor(bbox[1])
    return x0, y0, math.ceil(bbox[2]) - x0, math.ceil(bbox[3]) - y0


def _voxel_nm(cfg, files):
    """Raw (z, y, x) voxel size: median over the used files in check/files.csv, else 8 nm."""
    path = Path(cfg["output_dir"]) / "check" / "files.csv"
    df = pd.read_csv(path, dtype={"file": str}) if path.exists() else pd.DataFrame({"file": []})
    df = df[df["file"].isin(set(files))]
    vox, unknown = [], []
    for col in ("voxel_z_nm", "voxel_y_nm", "voxel_x_nm"):
        v = pd.to_numeric(df[col], errors="coerce").median() if col in df else np.nan
        if not (np.isfinite(v) and v > 0):
            unknown.append(col)
            v = 8.0
        vox.append(float(v))
    if unknown:
        log.warning("no usable %s in %s: assuming 8 nm", ", ".join(unknown), path)
    return vox


def _positions(cfg, zs):
    """zcorrect positions (nm) of the selected z, or None when z-correction is off or not run."""
    if not get(cfg, "zcorrect.enabled", False):
        return None
    path = Path(cfg["output_dir"]) / "zcorrect" / "positions.csv"
    if not path.exists():
        log.warning("zcorrect.enabled but %s not found: rendering without z-correction", path)
        return None
    pos = pd.read_csv(path).set_index("z")["position_nm"].dropna()
    missing = [z for z in zs if z not in pos.index]
    if missing:
        raise ValueError(f"{path} has no position for {len(missing)} selected slices "
                         f"(e.g. z={missing[:3]}): re-run zcorrect")
    return pos.loc[zs].to_numpy(float)


def plane_sources(zs, f, positions=None, step_nm=None):
    """Source slices [(z, weight), ...] of every output plane.

    Without positions, plane k is the mean of the k-th group of f consecutive selected slices
    (neighbours by position in ``zs``, which may skip z). With positions (nm, one per z), planes
    are ``step_nm`` apart over the position range and interpolate linearly between the two
    slices that bracket them.
    """
    if positions is None:
        return [[(z, 1.0 / len(g)) for z in g] for g in (zs[i:i + f] for i in range(0, len(zs), f))]
    order = np.argsort(positions, kind="stable")
    p, zz = np.asarray(positions, float)[order], np.asarray(zs)[order]
    planes = []
    for k in range(int(np.floor((p[-1] - p[0]) / step_nm + 1e-6)) + 1):
        q = p[0] + k * step_nm
        i = int(np.searchsorted(p, q, side="right")) - 1   # p[i] <= q < p[i + 1]
        if i >= len(p) - 1:
            planes.append([(int(zz[-1]), 1.0)])
            continue
        t = (q - p[i]) / (p[i + 1] - p[i])
        planes.append([(int(z), float(w)) for z, w in ((zz[i], 1 - t), (zz[i + 1], t)) if w > 1e-6])
    return planes


def _destreak_settings(cfg):
    """Full destreak section (module defaults + config) when enabled, else None."""
    if not get(cfg, "destreak.enabled", False):
        return None
    from . import destreak
    return deep_merge(getattr(destreak, "DEFAULTS", {}), {"destreak": cfg["destreak"]})["destreak"]


def init(cfg, overwrite=False):
    """Freeze the render plan and create the empty volume. Returns an exit code."""
    s = _settings(cfg)
    f = s["downsample"]
    plan = _tile_plan(cfg)
    zs = z_values(plan)
    x0, y0, w, h = _canvas(plan, s["bbox"])
    vz, vy, vx = _voxel_nm(cfg, plan["file"].unique())
    positions = _positions(cfg, zs)
    planes = plane_sources(zs, f, positions, vz * f)
    meta = {
        "volume": s["name"],
        "origin_xy": [x0, y0],
        "canvas_size_xy": [w, h],
        "shape": [len(planes), math.ceil(h / f), math.ceil(w / f)],
        "voxel_nm": [vz * f, vy * f, vx * f],
        "downsample": f,
        "zcorrected": positions is not None,
        "shard": [s["slab"], s["shard_xy"], s["shard_xy"]],
        "settings": s,
        "destreak": _destreak_settings(cfg),
        "planes": planes,
    }
    digest = hashlib.sha256(json.dumps(meta, sort_keys=True).encode())
    digest.update(plan.to_csv(index=False).encode())
    meta["digest"] = digest.hexdigest()

    rdir = step_dir(cfg, "render")
    meta_path, root = rdir / "render.json", rdir / s["name"]
    old = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if old and not overwrite:
        if old.get("digest") == meta["digest"] and (root / "zarr.json").exists():
            log.info("%s matches the current inputs and settings: volume kept", meta_path)
            return 0
        log.error("render inputs or settings changed since %s was written: re-run init with "
                  "--overwrite to re-create %s (deletes everything rendered so far)", meta_path, root)
        return 1
    if old.get("volume", s["name"]) != s["name"] and (rdir / old["volume"]).exists():
        log.warning("%s (previous render.name) is left in place but no longer tracked by render.json "
                    "or done/ markers: delete it if it is not needed", rdir / old["volume"])
    # render.json goes last: it marks a complete init.
    meta_path.unlink(missing_ok=True)
    shutil.rmtree(rdir / "done", ignore_errors=True)
    shutil.rmtree(root, ignore_errors=True)
    omezarr.create(root, meta["shape"], meta["voxel_nm"], num_scales=s["num_scales"], chunk=s["chunk"],
                   shard=meta["shard"], compression=s["compression"], name=cfg.get("name") or "volume")
    atomic_write(rdir / "tiles.csv", lambda p: plan.to_csv(p, index=False))
    atomic_write(meta_path, lambda p: Path(p).write_text(json.dumps(meta, indent=1)))
    n_slabs = math.ceil(meta["shape"][0] / s["slab"])
    log.info("created %s: shape %s (z, y, x), voxel %s nm, origin %s, %d slices -> %d planes in %d slabs "
             "(run array size <= %d)", root, meta["shape"], meta["voxel_nm"], meta["origin_xy"], len(zs),
             len(planes), n_slabs, n_slabs)
    return 0


# ----- run ------------------------------------------------------------------------------

def _ramp(start, n, size, f, blend):
    """Feathering weight of n downsampled pixels whose f-blocks start at full-res index ``start``
    of a tile ``size`` px long: linear from the tile edge over ``blend`` full-res px."""
    centre = start + f * np.arange(n) + (f - 1) / 2
    d = np.minimum(centre + 0.5, size - 0.5 - centre)
    return (np.clip(d / blend, 0, 1) if blend > 0 else np.ones(n)).astype(np.float32)


def _normalise(img, lo, hi):
    """Raw intensities -> float32 [0, 1]; missing lo/hi fall back to the image's p0.5/p99.5."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        lo, hi = np.percentile(img[::4, ::4], (0.5, 99.5))
    v = np.subtract(img, np.float32(lo), dtype=np.float32)
    v *= np.float32(1.0 / max(float(hi) - float(lo), 1e-6))
    return np.clip(v, 0, 1, out=v)


def _clahe(img, valid, clip_limit, tile):
    """CLAHE of a uint8 plane over ``tile``-px tiles; no-data pixels (``~valid``) end up 0.

    No-data is first filled with values drawn from the plane's own histogram: a block of zeros
    would skew the CLAHE tiles at the data's edge, which moves with the drift from plane to plane.
    """
    invalid = ~valid
    n = int(np.count_nonzero(invalid))
    if 0 < n < img.size:
        p = cv2.calcHist([img], [0], valid.view(np.uint8), [256], [0, 256]).ravel().astype(float)
        fill = np.random.default_rng(0).choice(256, 1 << 16, p=p / p.sum()).astype(np.uint8)
        img[invalid] = np.resize(fill, n)
    grid = (math.ceil(img.shape[1] / tile), math.ceil(img.shape[0] / tile))
    out = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=grid).apply(img)
    out[invalid] = 0
    return out


class _Renderer:
    """Renders output planes from the plan frozen by init (render.json + render/tiles.csv)."""

    def __init__(self, cfg, meta, plan, pool):
        s = meta["settings"]
        self.f = int(meta["downsample"])
        self.origin = np.asarray(meta["origin_xy"], float)
        self.shape = tuple(meta["shape"][1:])
        self.blend = float(s["blend_px"])
        self.clahe = s["clahe"]
        self.pool = pool
        self.cache = StackCache(cfg["raw_dir"])
        self.tiles = {z: g.to_dict("records") for z, g in plan.groupby("z")}
        self.kept = {}
        self.destreak = None
        if meta.get("destreak"):
            from . import destreak
            dcfg = {**cfg, "destreak": meta["destreak"]}
            self.destreak = lambda img: destreak.destreak_from_cfg(img, dcfg)

    def plane(self, sources, keep=()):
        """uint8 plane = weighted mean of its source slices (0 = no data), then CLAHE.

        Source slices whose z is in ``keep`` (the next plane's sources) are kept for reuse:
        consecutive z-corrected planes interpolate between mostly the same two slices.
        """
        kept, self.kept = self.kept, {}
        # Submit every tile up front so the source slices render concurrently.
        jobs = {z: [self.pool.submit(self._tile, t) for t in self.tiles.get(z, [])]
                for z, _ in sources if z not in kept}
        slices = []
        for z, w in sources:
            s = kept[z] if z in kept else self._slice(jobs.pop(z))
            if z in keep:
                self.kept[z] = s
            slices.append((w, *s))
        if len(slices) == 1:
            _, val, valid = slices[0]
        else:
            # Normalise by the weights of slices that have data, so drift edges don't darken.
            val, den = np.zeros(self.shape, np.float32), np.zeros(self.shape, np.float32)
            for w, acc, ok in slices:
                val += np.float32(w) * acc
                den += np.float32(w) * ok
            valid = den > 0
            np.divide(val, den, out=val, where=valid)
        out = cv2.convertScaleAbs(val, alpha=255.0)   # rounds to uint8; val (maybe kept) is untouched
        if self.clahe["enabled"]:
            out = _clahe(out, valid, self.clahe["clip_limit"], max(1.0, self.clahe["tile_px"] / self.f))
        return out

    def _slice(self, futures):
        """Blend one slice's placed tiles: (values in [0, 1], valid mask) over the output plane."""
        acc, wsum = np.zeros(self.shape, np.float32), np.zeros(self.shape, np.float32)
        while futures:   # in submission order (reproducible sums); each result freed once added
            placed = futures.pop(0).result()
            if placed is not None:
                win, vw, ww = placed
                acc[win] += vw
                wsum[win] += ww
        valid = wsum > 0
        np.divide(acc, wsum, out=acc, where=valid)
        return acc, valid

    def _tile(self, t):
        """Read, normalise, downsample and place one tile slice.

        Returns (output window, weighted values, weights) for the window of the output plane
        the tile covers, or None if it misses the canvas.
        """
        f, (H, W) = self.f, self.shape
        T = transforms.from_row(t)
        h, w = int(t["height"]), int(t["width"])
        # Only the rows/cols that land on the canvas (all of them unless render.bbox crops).
        canvas = self.origin + f * np.array([[0, 0], [W, 0], [0, H], [W, H]], float)
        box = transforms.apply(transforms.invert(T), canvas)
        c0, r0 = (int(q) for q in np.maximum(np.floor(box.min(axis=0)) - f - 1, 0))
        c1, r1 = (int(q) for q in np.minimum(np.ceil(box.max(axis=0)) + f + 1, (w, h)))
        is_shift = _is_shift(T)
        if f > 1:
            if is_shift and _whole(T[:, 2]):
                # Start on an output block boundary so area-downsampling gives exact block means.
                c0 += int((self.origin[0] - T[0, 2] - c0) % f)
                r0 += int((self.origin[1] - T[1, 2] - r0) % f)
            c1 -= (c1 - c0) % f
            r1 -= (r1 - r0) % f
        if c1 <= c0 or r1 <= r0:
            return None
        img = self.cache.read(t, slice(r0, r1), slice(c0, c1))
        if self.destreak is not None:
            img = self.destreak(img)
        v = _normalise(img, t["lo"], t["hi"])
        if f > 1:
            v = cv2.resize(v, ((c1 - c0) // f, (r1 - r0) // f), interpolation=cv2.INTER_AREA)
        hv, wv = v.shape
        wt = np.outer(_ramp(r0, hv, h, f, self.blend), _ramp(c0, wv, w, f, self.blend))
        v *= wt
        # Downsampled crop pixel (i, j) has its centre at tile (c0 + f*j + c, r0 + f*i + c).
        L, c = T[:, :2], (f - 1) / 2
        M = np.hstack([L, ((L @ (np.array([c0, r0]) + c) + T[:, 2] - self.origin - c) / f)[:, None]])
        if is_shift and _whole(M[:, 2]):
            x, y = (int(round(m)) for m in M[:, 2])
            xa, xb, ya, yb = max(x, 0), min(x + wv, W), max(y, 0), min(y + hv, H)
            if xa >= xb or ya >= yb:
                return None
            src = (slice(ya - y, yb - y), slice(xa - x, xb - x))
            return (slice(ya, yb), slice(xa, xb)), v[src], wt[src]
        # Warp only into the window of the plane the tile covers, never a full-plane temporary.
        p = transforms.apply(M, [[-0.5, -0.5], [wv - 0.5, -0.5], [-0.5, hv - 0.5], [wv - 0.5, hv - 0.5]])
        xa, ya = (int(q) for q in np.maximum(np.floor(p.min(axis=0)), 0))
        xb, yb = (int(q) for q in np.minimum(np.ceil(p.max(axis=0)) + 1, (W, H)))
        if xa >= xb or ya >= yb:
            return None
        Mw = M - [[0, 0, xa], [0, 0, ya]]

        def warp(a):
            return cv2.warpAffine(a, Mw, (xb - xa, yb - ya), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # Warping premultiplied values and weights separately keeps tile edges unbiased.
        return (slice(ya, yb), slice(xa, xb)), warp(v), warp(wt)


def _done(marker, digest):
    """A slab marker counts only if it was written for the current plan (render.json digest)."""
    try:
        return marker.read_text().strip() == digest
    except FileNotFoundError:
        return False


def run(cfg, task_id, num_tasks, overwrite=False):
    """Render this task's slabs (strided) that have no done marker for the current plan yet."""
    rdir = step_dir(cfg, "render")
    meta_path = rdir / "render.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found: run `pipeline.render init` first")
    meta = json.loads(meta_path.read_text())
    arr = omezarr.open_scale(rdir / meta["volume"], 0)
    slab = int(meta["settings"]["slab"])
    if omezarr.shard_shape(arr)[0] != slab:
        raise ValueError(f"render.slab {slab} != shard z {omezarr.shard_shape(arr)[0]} of {meta['volume']}: "
                         "re-run init with --overwrite")
    n, H, W = meta["shape"]
    sy = omezarr.shard_shape(arr)[1]
    mine = my_chunks(list(range(math.ceil(n / slab))), task_id, num_tasks)
    marker = {k: rdir / "done" / f"slab_{k:06d}" for k in mine}
    todo = [k for k in mine if overwrite or not _done(marker[k], meta["digest"])]
    log.info("task %d/%d: %d of my %d slabs to render", task_id, num_tasks, len(todo), len(mine))
    if not todo:
        return 0
    plan = pd.read_csv(rdir / "tiles.csv", dtype={"tile": str, "file": str})
    if plan[["lo", "hi"]].isna().any(axis=None):
        log.warning("some tile slices have no intensity levels: using their own p0.5/p99.5")
    with ThreadPoolExecutor(max(1, int(cfg["render"]["threads"]))) as pool:
        renderer = _Renderer(cfg, meta, plan, pool)
        for k in todo:
            t0 = time.monotonic()
            z0, z1 = k * slab, min((k + 1) * slab, n)
            buf = np.empty((z1 - z0, H, W), np.uint8)
            for i in range(z0, z1):
                nxt = meta["planes"][i + 1] if i + 1 < z1 else []
                buf[i - z0] = renderer.plane(meta["planes"][i], keep={z for z, _ in nxt})
            # Whole shards only, so concurrent tasks never touch the same shard. One row of shards
            # per write: a single write of the whole slab needs ~1.5x the slab again in buffers.
            for y in range(0, H, sy):
                arr[z0:z1, y:min(y + sy, H)].write(buf[:, y:y + sy]).result()
            del buf   # free it before the next slab's buffer fills
            atomic_write(marker[k], lambda p: Path(p).write_text(meta["digest"] + "\n"))
            log.info("slab %d: planes %d-%d written in %.0f s", k, z0, z1 - 1, time.monotonic() - t0)
    return 0


def main(argv=None):
    p = base_parser("Render the corrected volume (scale 0) as OME-Zarr.")
    p.add_argument("command", choices=["init", "run"], help="init (single job), then run (array over z-slabs)")
    args = p.parse_args(argv)
    cfg = setup(args, DEFAULTS)
    if args.command == "init":
        return init(cfg, args.overwrite)
    return run(cfg, *task_info(args), overwrite=args.overwrite)


if __name__ == "__main__":
    sys.exit(main())
