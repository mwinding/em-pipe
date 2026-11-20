"""Reference tissue statistics on the common scale render windows, for intensity.window: reference.

``run`` (array over chunks of sampled slices): every ``refstats.every``-th selected slice of each
segment (and each segment's last) is read at full resolution, and each tile gets exactly what render
applies before the window: flat-field (shading), the patchgrid shape, then the tile's gain and offset
onto the slice's common scale (intensity's tile balance, ``intensity.balanced_scale``). Inside the
reference tissue's cells (work/refmask, eroded by ``erode_cells``, optionally only those whose centre
lies in ``region``) it measures the tissue's median and lc32: the mean SD in 8 px windows at 32 nm
(the 8 nm values area-averaged 4 x 4), as tools/refmask_check.py measures rendered volumes, here per
2 um cell and the median over the slice's cells. Also the tissue's quantiles (all tissue cells,
background left out) for intensity's clipping estimate. work/refstats/chunks/chunk_<z0>-<z1>.csv per
chunk; ``merge``: work/refstats/stats.csv and qc/refstats.png.

``region``: [x0, y0, x1, y1] in voxels of the volume render would make with this config (render.bbox,
rotate_deg and downsample as set; null for an open side), e.g. to leave out tissue another reference
volume does not have.
"""

import hashlib
import json
import logging
import sys

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

from . import patchgrid, refmask, shading
from .cli import atomic_write, base_parser, my_chunks, setup, task_info
from .config import deep_merge, qc_path, step_dir, step_path
from .preview import thumb_to_tile
from .slices import StackCache, load_slices

log = logging.getLogger(__name__)

DEFAULTS = {
    "refstats": {
        "enabled": False,         # run_pipeline.sh: refstats after patchgrid (for intensity.window: reference)
        "every": 32,              # measure every N-th selected slice of each segment (and its last)
        "region": None,           # [x0, y0, x1, y1] volume voxels (null sides open); null: the whole slice
        "erode_cells": 1,         # reference cells eroded by this many cells (away from the tissue's edge)
        "chunk_samples": 4,       # sampled slices per array-task chunk
        "median_stride": 4,       # 8 nm px stride of the values the median is taken from
        "tissue_stride": 8,       # 8 nm px stride of the tissue values the quantiles are taken from
    },
}
FACTOR = 4                        # 8 nm -> 32 nm for lc32
WINDOW = 8                        # lc32 window, px at 32 nm
TISSUE_Q = (0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 99.5, 99.8, 99.9, 99.95, 99.99)
REF_Q = (1, 5, 25, 50, 75, 95, 99)


def settings_of(cfg):
    return {**DEFAULTS["refstats"], **(cfg.get("refstats") or {})}


def sample_z(slices, every):
    """Sampled z: every ``every``-th selected slice of each segment (by position), and its last."""
    per_z = slices.drop_duplicates("z").sort_values("z")
    out = []
    for _, g in per_z.groupby("segment"):
        zs = g["z"].tolist()
        out += zs[::int(every)]
        if zs[-1] not in out:
            out.append(zs[-1])
    return sorted(out)


def lc32_map(c):
    """SD in WINDOW px windows at 32 nm of a tile's values c (8 nm): (sd map, shape)."""
    h, w = (n // FACTOR * FACTOR for n in c.shape)
    y = cv2.resize(np.ascontiguousarray(c[:h, :w]), (w // FACTOR, h // FACTOR), interpolation=cv2.INTER_AREA)
    y = y.astype(np.float64)
    y -= np.median(y)   # small numbers: no cancellation in E[y^2] - E[y]^2
    m = ndimage.uniform_filter(y, WINDOW)
    return np.sqrt(np.maximum(ndimage.uniform_filter(y * y, WINDOW) - m * m, 0))


def in_region(region, M, centres):
    """Which points (tile px, N x 2) map (2x3 M: tile px -> volume voxels) into region."""
    if region is None:
        return np.ones(len(centres), bool)
    x0, y0, x1, y1 = (v if v is not None else s * np.inf for v, s in zip(region, (-1, -1, 1, 1)))
    p = centres @ np.asarray(M)[:, :2].T + np.asarray(M)[:, 2]
    return (p[:, 0] >= x0) & (p[:, 0] < x1) & (p[:, 1] >= y0) & (p[:, 1] < y1)


def cell_selection(code, thumb_shape, tile_shape, cell_px, erode, region=None, M=None):
    """Reference cells to measure in: eroded by ``erode`` cells, centres inside region (bool, cell grid)."""
    ref = code == refmask.CELL_REFERENCE
    if erode:
        ref = ndimage.binary_erosion(ref, np.ones((3, 3), bool), iterations=int(erode), border_value=0)
    if region is not None and ref.any():
        i, j = np.nonzero(ref)
        th, tw = thumb_shape
        cx = np.minimum(j * cell_px + (cell_px - 1) / 2, tw - 1)
        cy = np.minimum(i * cell_px + (cell_px - 1) / 2, th - 1)
        T = thumb_to_tile(thumb_shape, tile_shape)
        pts = np.c_[T[0, 0] * cx + T[0, 2], T[1, 1] * cy + T[1, 2]]
        keep = in_region(region, M, pts)
        ref = ref.copy()
        ref[i[~keep], j[~keep]] = False
    return ref


def _on_grid(cells, cell_px, thumb_shape, shape):
    """A cell-grid array (bool or int) on a pixel grid of ``shape`` covering the tile (nearest)."""
    px = np.repeat(np.repeat(np.asarray(cells), cell_px, 0), cell_px, 1)[:thumb_shape[0], :thumb_shape[1]]
    return cv2.resize(px.astype(np.int32), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


class Measurer:
    """Everything a task needs once: tile balance, corrections, masks, the volume frame."""

    def __init__(self, cfg):
        from .intensity import balanced_scale
        self.cfg, self.s = cfg, settings_of(cfg)
        self.slices = load_slices(cfg)
        bal = balanced_scale(cfg)
        self.go = {(int(z), t): (g, o) for z, t, g, o in zip(bal["z"], bal["tile"], bal["g"], bal["o"])}
        self.fields = shading.load(cfg)
        self.shapes, self.R = patchgrid.load(cfg)
        self.cells = refmask.Cells(cfg)
        self.cell_px = self.cells.cell_px
        idx = pd.read_csv(step_path(cfg, "preview", "thumbs", "index.csv"), dtype={"tile": str})
        self.thumb_npy = {(int(z), t): n for z, t, n in zip(idx["z"], idx["tile"], idx["npy"])}
        self.thumbs_dir = step_path(cfg, "preview", "thumbs")
        self.cache = StackCache(cfg["raw_dir"])
        self.frame = None
        if self.s["region"] is not None:
            from .render import frame
            self.frame = frame(cfg)
            self.T = {(int(r["z"]), r["tile"]): r for r in self.frame["plan"].to_dict("records")}
        if (cfg.get("destreak") or {}).get("enabled"):
            log.warning("destreak is enabled: refstats measures the tiles without it")
        self._tile_fields = {}

    def digest(self, zs):
        """Identifies a chunk's inputs: settings, masks, corrections and the tiles' gain/offset."""
        go = sorted((z, t, round(float(g), 6), round(float(o), 3)) for (z, t), (g, o) in self.go.items()
                    if z in zs and np.isfinite(g))
        h = hashlib.sha256(json.dumps([self.s, refmask.digest(refmask.settings_of(self.cfg)), shading.digest(self.cfg),
                                       patchgrid.digest(self.cfg), go], sort_keys=True, default=str).encode())
        return h.hexdigest()[:16]

    def corrected(self, row):
        """A tile slice's values as render corrects them (before the window), float32."""
        img = self.cache.read(row)
        h, w = img.shape
        key = (int(row["segment"]), row["tile"])
        field = self.fields.get(key)
        rows, cols = slice(0, h), slice(0, w)
        if self.R is not None:
            k = (*key, h, w)
            if k not in self._tile_fields:
                self._tile_fields[k] = patchgrid.tile_fields(self.shapes.get(key) or patchgrid.identity(1), self.R,
                                                             field, (h, w))
            return patchgrid.apply_fields(img, self._tile_fields[k], (h, w), rows, cols)
        if field is not None:
            return np.divide(img, shading.gain_window(field, (h, w), rows, cols), dtype=np.float32)
        return img.astype(np.float32)

    def slice_stats(self, z):
        s = self.s
        rows = self.slices[self.slices["z"] == z]
        ref_vals, tissue_vals, cell_sd = [], [], []
        n_tiles = 0
        for row in rows.to_dict("records"):
            t = row["tile"]
            code = self.cells.get(z, t)
            npy = self.thumb_npy.get((z, t))
            if code is None or npy is None:
                continue
            thumb_shape = np.load(self.thumbs_dir / npy, mmap_mode="r").shape[1:]
            tile_shape = (int(row["height"]), int(row["width"]))
            M = None
            if self.frame is not None:
                from .render import to_volume
                M = to_volume(self.frame, np.array([[self.T[(z, t)][k] for k in ("a", "b", "tx")],
                                                    [self.T[(z, t)][k] for k in ("c", "d", "ty")]]))
            sel = cell_selection(code, thumb_shape, tile_shape, self.cell_px, s["erode_cells"], s["region"], M)
            tissue = code > refmask.CELL_BACKGROUND
            if not tissue.any():
                continue
            g, o = self.go.get((z, t), (np.nan, np.nan))
            g, o = (1.0, 0.0) if not np.isfinite(g) else (float(g), float(o))
            c = self.corrected(row)
            c *= np.float32(g)
            c += np.float32(o)
            n_tiles += 1
            ts = int(s["tissue_stride"])
            sub = c[::ts, ::ts]
            tissue_vals.append(sub[_on_grid(tissue, self.cell_px, thumb_shape, sub.shape) > 0])
            if sel.any():
                ms = int(s["median_stride"])
                sub = c[::ms, ::ms]
                ref_vals.append(sub[_on_grid(sel, self.cell_px, thumb_shape, sub.shape) > 0])
                sd = lc32_map(c)
                ids = np.where(sel, np.arange(sel.size).reshape(sel.shape) + 1, 0)
                idmap = _on_grid(ids, self.cell_px, thumb_shape, sd.shape).ravel()
                inside = idmap > 0
                sums = np.bincount(idmap[inside], sd.ravel()[inside], minlength=sel.size + 1)
                counts = np.bincount(idmap[inside], minlength=sel.size + 1)
                ok = counts > 0
                cell_sd.append(sums[ok] / counts[ok])
            del c
        rec = {"z": int(z), "segment": int(rows["segment"].iloc[0]) if len(rows) else -1, "n_tiles": n_tiles,
               "n_cells": int(sum(len(v) for v in cell_sd))}
        ref = np.concatenate(ref_vals) if ref_vals else np.zeros(0)
        rec["median"] = float(np.median(ref)) if ref.size else np.nan
        rec["lc32"] = float(np.median(np.concatenate(cell_sd))) if cell_sd else np.nan
        for q, v in zip(REF_Q, np.percentile(ref, REF_Q) if ref.size else [np.nan] * len(REF_Q)):
            rec[f"rq_{q:g}"] = float(v)
        tis = np.concatenate(tissue_vals) if tissue_vals else np.zeros(0)
        for q, v in zip(TISSUE_Q, np.percentile(tis, TISSUE_Q) if tis.size else [np.nan] * len(TISSUE_Q)):
            rec[f"tq_{q:g}"] = float(v)
        return rec


def _chunks(cfg):
    s = settings_of(cfg)
    zs = sample_z(load_slices(cfg), s["every"])
    n = max(int(s["chunk_samples"]), 1)
    return [zs[i:i + n] for i in range(0, len(zs), n)]


def _name(zs):
    return f"{zs[0]:06d}-{zs[-1] + 1:06d}"


def run(cfg, args):
    mine = my_chunks(_chunks(cfg), *task_info(args))
    log.info("%d chunks for this task", len(mine))
    if not mine:
        return 0
    m = Measurer(cfg)
    for zs in mine:
        out = step_dir(cfg, "refstats", "chunks", f"chunk_{_name(zs)}.csv")
        dg = m.digest(zs)
        if out.exists() and not args.overwrite:
            done = pd.read_csv(out)
            if set(done["z"]) >= set(zs) and (done["digest"] == dg).all():
                log.info("chunk %s exists, skipping", _name(zs))
                continue
        recs = []
        for z in zs:
            rec = m.slice_stats(z)
            rec["digest"] = dg
            recs.append(rec)
            log.info("z %d: %d tiles, %d cells, median %.1f, lc32 %.2f", z, rec["n_tiles"], rec["n_cells"],
                     rec["median"], rec["lc32"])
        atomic_write(out, lambda f: pd.DataFrame(recs).to_csv(f, index=False))
    return 0


def merge(cfg, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .preview import save
    parts, missing = [], []
    for zs in _chunks(cfg):
        path = step_path(cfg, "refstats", "chunks", f"chunk_{_name(zs)}.csv")
        if not path.exists():
            missing.append(path.name)
            continue
        d = pd.read_csv(path)
        if not set(d["z"]) >= set(zs):
            missing.append(f"{path.name} (other slices)")
            continue
        parts.append(d[d["z"].isin(zs)])
    if missing:
        raise RuntimeError(f"{len(missing)} refstats chunks missing or out of date, e.g. {missing[:3]}: run 'refstats run'")
    st = pd.concat(parts, ignore_index=True).sort_values("z")
    atomic_write(step_dir(cfg, "refstats", "stats.csv"), lambda f: st.to_csv(f, index=False, float_format="%.4f"))
    plot_stats(st, qc_path(cfg, "refstats.png"))
    log.info("refstats: %d sampled slices, %d with reference tissue", len(st), int((st["n_cells"] > 0).sum()))
    return 0


def plot_stats(st, path):
    """Reference tissue along z before any window: median, p5-p95 width, lc32 and cells measured per
    sampled slice, each segment shaded and labelled with its median values and spread within it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .preview import save
    st = st.copy()
    st["width"] = st["rq_95"] - st["rq_5"]
    ok = st["n_cells"] > 0
    panels = (("median", "neuropil median"), ("width", "neuropil p5-p95 width"), ("lc32", "neuropil lc32"),
              ("n_cells", "cells measured"))
    fig, axes = plt.subplots(len(panels), 1, sharex=True, figsize=(13, 10), layout="constrained")
    segs = list(st.groupby("segment"))
    for k, (seg, g) in enumerate(segs):
        z0, z1 = g["z"].min(), g["z"].max()
        for ax in axes:
            if k % 2:
                ax.axvspan(z0, z1, color="0.93", lw=0)
            if k:
                ax.axvline(z0, color="0.6", lw=0.8)
        gg = g[g["n_cells"] > 0]
        for ax, (col, _) in zip(axes[:3], panels[:3]):
            if len(gg):
                v = gg[col]
                ax.hlines(v.median(), z0, z1, color="#eb6834", lw=1.5)
                ax.text((z0 + z1) / 2, 1.0, f"seg {seg}\n{v.median():.0f}  (±{100 * v.std() / v.mean():.1f} %)",
                        transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8, color="#1a1a19")
    for ax, (col, label) in zip(axes, panels):
        ax.plot(st.loc[ok, "z"], st.loc[ok, col], "o", ms=3, color="#2a78d6")
        ax.set_ylabel(label)
    axes[0].set_title("Reference tissue before any contrast window (common scale), sampled slices. "
                      "Orange: each segment's median; ± its variation within the segment", loc="left", fontsize=10)
    axes[-1].set_xlabel("z")
    save(fig, path)


def main(argv=None):
    from . import intensity, preview, render
    p = base_parser("Reference tissue statistics on the common scale, for intensity.window: reference")
    p.add_argument("command", choices=["run", "merge"])
    args = p.parse_args(argv)
    defaults = DEFAULTS
    for d in (intensity.DEFAULTS, shading.DEFAULTS, patchgrid.DEFAULTS, refmask.DEFAULTS, preview.DEFAULTS,
              render.DEFAULTS):
        defaults = deep_merge(defaults, d)
    cfg = setup(args, defaults)
    return run(cfg, args) if args.command == "run" else merge(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
