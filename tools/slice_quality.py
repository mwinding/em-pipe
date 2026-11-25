"""Per-slice quality flags: slices (or single tiles of a slice) with poor focus, more noise, saturation,
bright or dark patches (charging), scan-line faults, curtains, a bad alignment fit or missing tiles.

    python tools/slice_quality.py run   --config C [--out DIR] [--task-id K --num-tasks N] [--overwrite]
    python tools/slice_quality.py merge --config C [--out DIR] [--top N] [--no-sheets]

Flags only: nothing is excluded or changed. A flag says "look at this slice", with its reason and value.

What is measured, per (z, tile):

  from the raw tiles at full resolution, in --bands full-width bands of --window rows (fixed places, at
  1/(bands+1), 2/(bands+1), ... of the tile height), each cut into --window squares; only squares that
  are tissue (>= --min-tissue by the preview thumbnail's background mask) are measured:
    sharp     power of the square's spectrum at 5-16 px periods over the power at 25-100 px, both
              after subtracting the noise floor (the spectrum's level at 2-2.5 px, where pixel noise
              dominates); the two axes of the spectrum are left out so curtains and scan lines don't count.
              Defocus lowers it; pixel noise does not change it.
    noise     pixel-noise sd from that floor (raw values)
    v, h      destreak.stripe_amplitude along and across the curtain direction (destreak.direction)
    sat, zero share of pixels at the data type's maximum / at 0
  from the preview thumbnails (whole tile, no raw read):
    ncc_prev, ncc_next   correlation with the same tile in the previous / next slice
    bright, dark         share of the tile brighter (darker) than BOTH neighbour slices by more than
                         --patch-iqr times the tile's interquartile range, after a 5 px smoothing (each
                         thumbnail first put on its own median and interquartile range): a patch only
                         this slice has
  from existing tables: the tile's tissue median (preview/stats.csv), the alignment residuals
  (align/residuals.csv), the slice list (check/slices.csv: missing tiles, excluded slices).

merge compares every square with its own running median along z (--trend slices, within a segment),
so the tissue's own texture cancels, and takes the median over a tile's squares:
    sharp_rel, noise_rel   this slice / the trend (1 = as its neighbours in z)
    scanline_rel           h / the square's p5-p95 width against the trend (worst band)
    curtain_ratio          v / h per square (texture without direction gives ~1), 90th percentile over
                           the tile's squares; curtain_extent = share of the squares above --curtain-ratio
    level_jump             tissue median against the trend, in units of the tissue's p5-p95 width
    ncc_drop               trend of max(ncc_prev, ncc_next) minus its value: a slice unlike both neighbours
    align_rms              smallest residual of the slice's alignment pairs (a bad slice fits none of them)

Flag rules (LIMITS below; each a limit on the value and, for trend comparisons, also a robust z-score
of at least --z-score against the tile's own scatter):
    focus, noise, scanline, brightness_jump, unlike_neighbours, align   outliers against the trend
    saturated, black, bright_patch, dark_patch, curtain, align_rejected, missing_tile, excluded   absolute
Severity: "major" at twice the limit's distance from normal, else "minor". When every tile of a
multi-tile slice has the same flag, they become one row with tile "all".

Limits of the method: an event lasting longer than about half the trend window (e.g. defocus over
100 slices) becomes the trend and is not flagged (curtains are judged absolutely, so they are); a patch
present in two neighbouring slices at the same place is not a "patch only this slice has"; curtains
are measured on the raw tiles, before any decurtaining, and only where they cross the bands.

Outputs (default <output_dir>/qc/): slice_quality.csv (per z, tile: the metrics), slice_flags.csv
(z, tile, reason, value, limit, severity, zscore, tiles_flagged: how many of the slice's tiles have that flag), slice_quality.png (metrics over z, flags marked),
slice_quality/summary.json (counts per reason, settings), slice_quality/sheet_<reason>.png (the top
--top flagged slices per reason next to the previous and next slice, for checking by eye), and the
chunk files of run in slice_quality/chunks/.

run reads the raw stacks (three bands per tile slice): use slurm/slice_quality.sbatch as an array.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import fft as sfft
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import background, destreak  # noqa: E402
from pipeline.cli import atomic_write, my_chunks, task_info  # noqa: E402
from pipeline.config import load_config, step_defaults, step_path  # noqa: E402
from pipeline.preview import chunk_name, thumb_values, z_chunks  # noqa: E402
from pipeline.slices import StackCache, load_slices  # noqa: E402

VERSION = 1
# spectrum bands in cycles per pixel
LOW, MID, HIGH = (0.01, 0.04), (0.0625, 0.2), (0.4, 0.5)
AXIS_BINS = 2          # bins either side of the spectrum's axes left out of the bands
WINDOW_METRICS = ("sharp", "noise", "width", "v", "h", "tissue")
ROW_METRICS = ("sat", "zero", "ncc_prev", "ncc_next", "bright", "dark")
# reason -> (column, direction, limit): flagged when the value is beyond the limit in that direction
LIMITS = {
    "focus": ("sharp_rel", "below", 0.75),
    "noise": ("noise_rel", "above", 1.25),
    "scanline": ("scanline_rel", "above", 2.0),
    "brightness_jump": ("level_jump", "abs", 0.5),
    "unlike_neighbours": ("ncc_drop", "above", 0.15),
    "align": ("align_rel", "above", 3.0),
    "saturated": ("sat", "above", 1e-3),
    "black": ("zero", "above", 1e-3),
    "bright_patch": ("bright", "above", 0.01),
    "dark_patch": ("dark", "above", 0.01),
    "curtain": ("curtain_extent", "above", 0.15),
}
NORMAL = {"focus": 1.0, "noise": 1.0, "scanline": 1.0, "align": 1.0}   # the value of an unremarkable slice (else 0)
TREND_REASONS = {"focus": "sharp_z", "noise": "noise_z", "scanline": "scanline_z", "brightness_jump": "level_z",
                 "unlike_neighbours": "ncc_z", "align": None}
ALIGN_MIN_PX = 2.0     # an alignment residual below this is never flagged
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


# --- measurements on one window / thumbnail ---------------------------------------------------------

class Spectrum:
    """Hann window and band masks for size x size squares."""

    def __init__(self, size):
        self.size = size
        w = np.hanning(size).astype(np.float32)
        self.hann = np.outer(w, w)
        self.energy = float((self.hann ** 2).sum())
        fy = np.fft.fftfreq(size)[:, None]
        fx = np.fft.rfftfreq(size)[None, :]
        f = np.hypot(fy, fx)
        off_axis = (np.abs(fy) > AXIS_BINS / size) & (np.abs(fx) > AXIS_BINS / size)
        self.masks = [off_axis & (f >= a) & (f < b) for a, b in (LOW, MID, HIGH)]

    def measure(self, win):
        """(sharp, noise sd) of one square."""
        a = np.asarray(win, np.float32)
        p = np.abs(sfft.rfft2((a - a.mean()) * self.hann)) ** 2
        low, mid, high = (p[m] for m in self.masks)
        if not (low.size and mid.size and high.size):   # a window too small for the bands
            return np.nan, np.nan
        floor = float(np.median(high)) / np.log(2)      # the mean of an exponential from its median
        lo, mi = float(low.mean()) - floor, float(mid.mean()) - floor
        sharp = max(mi, 0.0) / lo if lo > 0 else np.nan
        return sharp, float(np.sqrt(floor / self.energy))


def window_grid(h, w, size, n_bands):
    """(size, band top rows, window left columns): bands spread over the tile height, windows side by
    side across the whole width (centred)."""
    size = int(min(size, h, w))
    tops = [int(np.clip(round((k + 1) / (n_bands + 1) * h - size / 2), 0, h - size)) for k in range(n_bands)]
    ncol = w // size
    x0 = (w - ncol * size) // 2
    return size, sorted(set(tops)), [x0 + i * size for i in range(ncol)]


def measure_tile(read_rows, shape, spec, tops, lefts, tissue, direction="vertical", maxval=65535, min_tissue=0.9):
    """Window metrics of one tile slice. ``read_rows(y0, y1)`` returns those rows of the tile;
    ``tissue`` is a bool mask of the tile at any lower resolution (None: all tissue).
    Returns ({metric: array over windows, band-major}, sat, zero)."""
    size = spec.size
    other = "horizontal" if direction == "vertical" else "vertical"
    out = {k: np.full(len(tops) * len(lefts), np.nan, np.float32) for k in WINDOW_METRICS}
    n_sat = n_zero = n_px = 0
    for b, y0 in enumerate(tops):
        fracs = [tissue_fraction(tissue, shape, y0, x0, size) for x0 in lefts]
        if max(fracs, default=0) < min_tissue and tissue is not None:
            for i, fr in enumerate(fracs):
                out["tissue"][b * len(lefts) + i] = fr
            continue
        band = read_rows(y0, y0 + size)
        n_sat += int((band == maxval).sum())
        n_zero += int((band == 0).sum())
        n_px += band.size
        for i, x0 in enumerate(lefts):
            k = b * len(lefts) + i
            out["tissue"][k] = fracs[i]
            if fracs[i] < min_tissue:
                continue
            win = band[:, x0:x0 + size].astype(np.float32)
            out["sharp"][k], out["noise"][k] = spec.measure(win)
            p5, p95 = np.percentile(win[::4, ::4], [5, 95])
            out["width"][k] = p95 - p5
            out["v"][k] = destreak.stripe_amplitude(win, direction)
            out["h"][k] = destreak.stripe_amplitude(win, other)
    return out, (n_sat / n_px if n_px else np.nan), (n_zero / n_px if n_px else np.nan)


def tissue_fraction(tissue, shape, y0, x0, size):
    if tissue is None:
        return 1.0
    sy, sx = tissue.shape[0] / shape[0], tissue.shape[1] / shape[1]
    a = tissue[int(y0 * sy):max(int(np.ceil((y0 + size) * sy)), int(y0 * sy) + 1),
               int(x0 * sx):max(int(np.ceil((x0 + size) * sx)), int(x0 * sx) + 1)]
    return float(a.mean()) if a.size else 0.0


def ncc(a, b):
    a = a.astype(np.float32) - a.mean()
    b = b.astype(np.float32) - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else np.nan


def standardise(t):
    """A thumbnail on a scale of its own: median 0, interquartile range 1 (each thumbnail was scaled to
    its own percentiles, which one saturated patch shifts for the whole tile)."""
    t = np.asarray(t, np.float32)
    q1, med, q3 = np.percentile(t[::2, ::2], [25, 50, 75])
    return (t - med) / max(q3 - q1, 1.0)


def patches(t, prev, nxt, iqr=1.0, smooth=5):
    """(bright, dark): share of the thumbnail brighter (darker) than both neighbours by more than ``iqr``
    times the tile's interquartile range, after a ``smooth`` px box filter."""
    t = standardise(t)
    d1 = ndimage.uniform_filter(t - standardise(prev), smooth)
    d2 = ndimage.uniform_filter(t - standardise(nxt), smooth)
    return float(((d1 > iqr) & (d2 > iqr)).mean()), float(((d1 < -iqr) & (d2 < -iqr)).mean())


class Thumbs:
    """The preview thumbnails by (z, tile); absent when the preview step has not run."""

    def __init__(self, cfg):
        self.dir = step_path(cfg, "preview", "thumbs")
        self.index, self._files = {}, {}
        path = self.dir / "index.csv"
        if path.exists():
            idx = pd.read_csv(path, dtype={"tile": str})
            self.index = {(int(r.z), r.tile): (r.npy, int(r.i)) for r in idx.itertuples()}

    def get(self, z, tile):
        if (z, tile) not in self.index:
            return None
        npy, i = self.index[(z, tile)]
        if npy not in self._files:
            if len(self._files) > 40:
                self._files.clear()
            self._files[npy] = np.load(self.dir / npy, mmap_mode="r")
        return np.asarray(self._files[npy][i])


# --- run ------------------------------------------------------------------------------------------------

def out_dir(cfg, args):
    return Path(args.out) if args.out else Path(cfg["output_dir"]) / "qc"


def settings(cfg, args):
    return {"version": VERSION, "window": args.window, "bands": args.bands, "min_tissue": args.min_tissue,
            "patch_iqr": args.patch_iqr, "direction": cfg["destreak"]["direction"]}


def neighbours(slices):
    """{(z, tile): (previous z, next z)} within the tile's segment (None at the ends)."""
    out = {}
    for (_, tile), g in slices.groupby(["segment", "tile"]):
        zs = g["z"].tolist()
        for i, z in enumerate(zs):
            out[(z, tile)] = (zs[i - 1] if i else None, zs[i + 1] if i + 1 < len(zs) else None)
    return out


def run(cfg, args):
    out = out_dir(cfg, args) / "slice_quality" / "chunks"
    out.mkdir(parents=True, exist_ok=True)
    slices = load_slices(cfg)
    chunks = my_chunks(z_chunks(slices, args.chunk_slices), *task_info(args))
    sett = settings(cfg, args)
    thumbs, cache, nb = Thumbs(cfg), StackCache(cfg["raw_dir"]), neighbours(slices)
    if not thumbs.index:
        print("slice_quality: no preview thumbnails: every window counts as tissue, no thumbnail metrics", flush=True)
    specs = {}
    for zs in chunks:
        path = out / f"chunk_{chunk_name(zs)}.npz"
        want = slices[slices["z"].isin(zs)]
        keys = [f"{r.z}:{r.tile}:{pd.Timestamp(r.timestamp).isoformat()}" for r in want.itertuples()]
        if path.exists() and not args.overwrite:
            with np.load(path, allow_pickle=False) as d:
                if json.loads(str(d["settings"])) == sett and d["keys"].tolist() == keys:
                    continue
        rec = want.iloc[0]
        size, tops, lefts = window_grid(int(rec.height), int(rec.width), args.window, args.bands)
        spec = specs.setdefault(size, Spectrum(size))
        win = {k: [] for k in WINDOW_METRICS}
        row = {k: [] for k in ROW_METRICS}
        for r in want.to_dict("records"):
            z, tile = int(r["z"]), r["tile"]
            t = thumbs.get(z, tile)
            tissue = None if t is None else ~background.mask(t)
            stack = cache.get(r["file"])
            maxval = np.iinfo(stack.dtype).max if np.issubdtype(stack.dtype, np.integer) else np.inf
            m, sat, zero = measure_tile(lambda y0, y1: cache.read(r, slice(y0, y1)), (int(r["height"]), int(r["width"])),
                                        spec, tops, lefts, tissue, sett["direction"], maxval, args.min_tissue)
            for k in WINDOW_METRICS:
                win[k].append(m[k])
            zp, zn = nb[(z, tile)]
            tp = thumbs.get(zp, tile) if zp is not None else None
            tn = thumbs.get(zn, tile) if zn is not None else None
            same = [x is not None and t is not None and x.shape == t.shape for x in (tp, tn)]
            br, dk = patches(t, tp, tn, args.patch_iqr) if all(same) else (np.nan, np.nan)
            vals = {"sat": sat, "zero": zero, "ncc_prev": ncc(t, tp) if same[0] else np.nan,
                    "ncc_next": ncc(t, tn) if same[1] else np.nan, "bright": br, "dark": dk}
            for k in ROW_METRICS:
                row[k].append(vals[k])
        data = {k: np.stack(v) for k, v in win.items()}
        data.update({k: np.asarray(v, np.float32) for k, v in row.items()})
        data.update(z=want["z"].to_numpy(), tile=want["tile"].to_numpy().astype(str), segment=want["segment"].to_numpy(),
                    keys=np.asarray(keys), settings=np.asarray(json.dumps(sett)), size=size,
                    tops=np.asarray(tops), lefts=np.asarray(lefts))
        atomic_write(path, lambda tmp: np.savez_compressed(tmp, **data), suffix=".tmp.npz")
        print(f"slice_quality: chunk {chunk_name(zs)}: {len(want)} tile slices, {len(tops)}x{len(lefts)} windows", flush=True)


# --- merge: trends and flags ------------------------------------------------------------------------------

def running_median(a, window):
    """Centred running median along axis 0, NaNs ignored (at least 5 values, else NaN)."""
    return pd.DataFrame(a).rolling(int(window), center=True, min_periods=min(5, int(window))).median().to_numpy()


def robust_z(resid):
    """Residuals over 1.4826 x their median absolute deviation (NaN if that is 0)."""
    r = np.asarray(resid, float)
    ok = np.isfinite(r)
    if ok.sum() < 5:
        return np.full(r.shape, np.nan)
    sd = 1.4826 * np.median(np.abs(r[ok] - np.median(r[ok])))
    return (r - np.median(r[ok])) / sd if sd > 0 else np.full(r.shape, np.nan)


def nanmedian(a, axis):
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(a, axis=axis)


def tile_series(d, n_bands, args):
    """Per-slice metrics of one (segment, tile) from its window arrays ``d`` (z-sorted, [n, W])."""
    n, W = d["sharp"].shape
    valid = (d["tissue"] >= args.min_tissue) & np.isfinite(d["sharp"])
    out = {"n_windows": valid.sum(1)}
    worst = {}
    with np.errstate(all="ignore"):
        for name, col in (("sharp", "sharp"), ("noise", "noise")):
            v = np.where(valid & (d[col] > 0), np.log(d[col]), np.nan)
            rel = v - running_median(v, args.trend)
            out[name] = np.exp(nanmedian(v, 1))
            out[f"{name}_rel"] = np.exp(nanmedian(rel, 1))
            out[f"{name}_z"] = robust_z(np.log(out[f"{name}_rel"]))
            worst[name] = rel
        hw = np.where(valid & (d["width"] > 0) & (d["h"] > 0), np.log(d["h"] / d["width"]), np.nan)
        rel = hw - running_median(hw, args.trend)
        per_band = np.stack([nanmedian(rel[:, b * (W // n_bands):(b + 1) * (W // n_bands)], 1) for b in range(n_bands)], 1)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out["scanline_rel"] = np.exp(np.nanmax(per_band, 1))
            ratio = np.where(valid & (d["h"] > 0), d["v"] / d["h"], np.nan)
            out["curtain_ratio"] = np.nanpercentile(ratio, 90, axis=1)
        out["scanline_z"] = robust_z(np.log(out["scanline_rel"]))
        out["curtain_extent"] = np.where(out["n_windows"] > 0, (ratio > args.curtain_ratio).sum(1) / np.maximum(out["n_windows"], 1), np.nan)
        worst["scanline"], worst["curtain"] = rel, ratio
    best = np.fmax(d["ncc_prev"], d["ncc_next"])
    out["ncc"] = best
    trend = running_median(best[:, None], args.trend)[:, 0]
    out["ncc_drop"] = trend - best
    out["ncc_z"] = robust_z(out["ncc_drop"])
    for k in ("sat", "zero", "bright", "dark"):
        out[k] = d[k]
    return out, worst


def load_chunks(cdir, slices, sett):
    """{segment: dict of arrays over that segment's rows} from the chunk files; exits on missing or stale chunks."""
    files = sorted(cdir.glob("chunk_*.npz"))
    if not files:
        sys.exit(f"slice_quality merge: no chunk files in {cdir}: run first")
    segs = {}
    for f in files:
        with np.load(f, allow_pickle=False) as d:
            if json.loads(str(d["settings"])) != sett:
                sys.exit(f"slice_quality merge: {f.name} was made with other settings: run again (it redoes such chunks)")
            segs.setdefault(int(d["segment"][0]), []).append({k: d[k] for k in d.files})
    have = set()
    for parts in segs.values():
        for p in parts:
            have.update(zip(p["z"].tolist(), p["tile"].tolist()))
    missing = set(zip(slices["z"].tolist(), slices["tile"].tolist())) - have
    if missing:
        sys.exit(f"slice_quality merge: {len(missing)} selected tile slices not measured yet (e.g. z {min(missing)[0]}): "
                 "run is incomplete")
    return segs


def measure_table(segs, slices, args):
    """(metrics per (z, tile), {(z, tile): {reason: worst window}}, {segment: (size, tops, lefts)})."""
    wanted = set(zip(slices["z"].tolist(), slices["tile"].tolist()))
    rows, worst_win, grids = [], {}, {}
    for seg, parts in sorted(segs.items()):
        W = parts[0]["sharp"].shape[1]
        if any(p["sharp"].shape[1] != W for p in parts):
            sys.exit(f"slice_quality merge: segment {seg} has chunks with different window grids")
        cat = {k: np.concatenate([p[k] for p in parts]) for k in (*WINDOW_METRICS, *ROW_METRICS, "z", "tile")}
        grids[seg] = (int(parts[0]["size"]), parts[0]["tops"].tolist(), parts[0]["lefts"].tolist())
        for tile in sorted(set(cat["tile"].tolist())):
            sel = np.flatnonzero(cat["tile"] == tile)
            sel = sel[np.argsort(cat["z"][sel], kind="stable")]
            sel = sel[[(int(cat["z"][i]), tile) in wanted for i in sel]]
            if not len(sel):
                continue
            d = {k: cat[k][sel] for k in cat}
            series, worst = tile_series(d, len(grids[seg][1]), args)
            df = pd.DataFrame({"z": d["z"], "tile": tile, "segment": seg, **series})
            rows.append(df)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                for i, z in enumerate(d["z"].tolist()):
                    ww = {}
                    for reason, (arr, pick) in {"focus": (worst["sharp"], np.nanargmin), "noise": (worst["noise"], np.nanargmax),
                                                "scanline": (worst["scanline"], np.nanargmax),
                                                "curtain": (worst["curtain"], np.nanargmax)}.items():
                        if np.isfinite(arr[i]).any():
                            ww[reason] = int(pick(arr[i]))
                    worst_win[(z, tile)] = ww
    return pd.concat(rows, ignore_index=True), worst_win, grids


def add_tables(cfg, q, args):
    """Add the tissue level (preview stats) and the alignment residual to the metrics."""
    sp = step_path(cfg, "preview", "stats.csv")
    q["level_jump"], q["level_z"], q["tissue_fraction"] = np.nan, np.nan, np.nan
    if sp.exists():
        st = pd.read_csv(sp, dtype={"tile": str})
        if {"t_p50", "t_p5", "t_p95"} <= set(st.columns):
            q = q.drop(columns=["tissue_fraction"]).merge(st[["z", "tile", "t_p50", "t_p5", "t_p95", "tissue_fraction"]],
                                                          on=["z", "tile"], how="left")
            for _, g in q.groupby(["segment", "tile"]):
                g = g.sort_values("z")
                med = running_median(g[["t_p50"]].to_numpy(float), args.trend)[:, 0]
                width = running_median((g["t_p95"] - g["t_p5"]).to_numpy(float)[:, None], args.trend)[:, 0]
                with np.errstate(all="ignore"):
                    jump = (g["t_p50"].to_numpy(float) - med) / width
                q.loc[g.index, "level_jump"] = jump
                q.loc[g.index, "level_z"] = robust_z(jump)
            q = q.drop(columns=["t_p50", "t_p5", "t_p95"])
    q["align_rms"], q["align_rel"], q["align_rejected"] = np.nan, np.nan, False
    rp = step_path(cfg, "align", "residuals.csv")
    if rp.exists():
        res = pd.read_csv(rp)
        res["rejected"] = res["rejected"].astype(str).str.lower().isin(["true", "1"])
        both = pd.concat([res.rename(columns={"z_a": "z"}), res.rename(columns={"z_b": "z"})])
        per_z = both.groupby("z").agg(align_rms=("rms_px", "min"), align_rejected=("rejected", "all")).sort_index()
        trend = running_median(per_z[["align_rms"]].to_numpy(float), args.trend)[:, 0]
        with np.errstate(all="ignore"):
            per_z["align_rel"] = per_z["align_rms"] / trend
        q = q.drop(columns=["align_rms", "align_rel", "align_rejected"]).merge(per_z, left_on="z", right_index=True, how="left")
        q["align_rejected"] = q["align_rejected"].fillna(False).astype(bool)
    return q


def make_flags(q, all_slices, args):
    """slice_flags rows from the metrics table and the slice list (incl. excluded slices)."""
    rows = []
    for reason, (col, direction, limit) in LIMITS.items():
        limit = getattr(args, f"limit_{reason}", None) or limit
        v = q[col].to_numpy(float)
        normal = NORMAL.get(reason, 0.0)
        if direction == "below":
            hit, major = v < limit, v < normal - 2 * (normal - limit)
        elif direction == "abs":
            hit, major = np.abs(v) > limit, np.abs(v) > 2 * limit
        else:
            hit, major = v > limit, v > normal + 2 * (limit - normal)
        zs = np.full(len(q), np.nan)
        if reason in TREND_REASONS:
            zc = TREND_REASONS[reason]
            if zc:
                zs = q[zc].to_numpy(float)
                hit &= np.abs(zs) >= args.z_score
            else:
                hit &= q["align_rms"].to_numpy(float) > ALIGN_MIN_PX
        for i in np.flatnonzero(hit & np.isfinite(v)):
            r = q.iloc[i]
            rows.append({"z": int(r.z), "tile": r.tile, "reason": reason, "value": float(v[i]), "limit": limit,
                         "severity": "major" if major[i] else "minor", "zscore": float(zs[i])})
    f = pd.DataFrame(rows, columns=["z", "tile", "reason", "value", "limit", "severity", "zscore"])
    # whole-slice rows: every tile of a multi-tile slice has the flag (alignment is per slice anyway)
    n_tiles = q.groupby("z")["tile"].nunique()
    merged = []
    for (z, reason), g in f.groupby(["z", "reason"]):
        g = g.assign(tiles_flagged=f"{len(g)}/{n_tiles[z]}")
        if (len(g) == n_tiles[z] and n_tiles[z] > 1) or reason == "align":
            worst = g.iloc[(g["value"] - NORMAL.get(reason, 0.0)).abs().to_numpy().argmax()]
            merged.append({**worst.to_dict(), "tile": "all", "worst_tile": worst["tile"],
                           "severity": "major" if (g["severity"] == "major").any() else "minor"})
        else:
            merged.extend({**r, "worst_tile": r["tile"]} for r in g.to_dict("records"))
    for z in sorted(q.loc[q["align_rejected"], "z"].unique()):
        merged.append({"z": int(z), "tile": "all", "reason": "align_rejected", "value": 1.0, "limit": np.nan,
                       "severity": "major", "zscore": np.nan, "worst_tile": ""})
    # missing tiles and excluded slices, from the slice list
    used = all_slices[~all_slices["excluded"]]
    expect = used.groupby("segment")["tile"].agg(lambda s: set(s))
    for z, g in used.groupby("z"):
        gone = expect[g["segment"].iloc[0]] - set(g["tile"])
        for t in sorted(gone):
            merged.append({"z": int(z), "tile": t, "reason": "missing_tile", "value": 1.0, "limit": np.nan,
                           "severity": "major", "zscore": np.nan, "worst_tile": t})
    ex = all_slices[all_slices["excluded"]]
    for z, g in ex.groupby("z"):
        whole = z not in set(used["z"])
        for t in (["all"] if whole else sorted(g["tile"])):
            merged.append({"z": int(z), "tile": t, "reason": "excluded", "value": 1.0, "limit": np.nan,
                           "severity": "major", "zscore": np.nan, "worst_tile": "",
                           "note": str(g["exclude_reason"].iloc[0])})
    out = pd.DataFrame(merged, columns=["z", "tile", "reason", "value", "limit", "severity", "zscore", "tiles_flagged",
                                        "worst_tile", "note"])
    return out.sort_values(["z", "reason", "tile"]).reset_index(drop=True)


def plot(path, q, flags, seams):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("sharp_rel", "sharpness / trend", "focus", True), ("noise_rel", "pixel noise / trend", "noise", True),
              ("curtain_extent", "curtains: share of windows", "curtain", False),
              ("bright", "bright patch share", "bright_patch", False), ("dark", "dark patch share", "dark_patch", False),
              ("ncc", "correlation with neighbour slices", "unlike_neighbours", False),
              ("sat", "saturated share", "saturated", False), ("align_rms", "alignment residual (px)", "align", False)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(15, 2.1 * len(panels)), sharex=True)
    tiles = sorted(q["tile"].unique())
    for ax, (col, label, reason, logy) in zip(axes, panels):
        for s in seams:
            ax.axvline(s, color="0.85", lw=0.6, zorder=0)
        fl = flags[flags["reason"] == reason]
        for i, tile in enumerate(tiles):
            g = q[q["tile"] == tile].sort_values("z")
            if col == "align_rms" and i:
                continue
            ax.plot(g["z"], g[col], lw=0.6, color=COLORS[i % 8], ls=["-", "--", ":"][i // 8 % 3],
                    label=tile if col != "align_rms" else None)
        if len(fl):
            key = fl.assign(t=fl["worst_tile"].where(fl["tile"] == "all", fl["tile"]))
            pts = key.merge(q[["z", "tile", col]], left_on=["z", "t"], right_on=["z", "tile"], how="left")
            if col == "align_rms":
                pts = fl.merge(q.drop_duplicates("z")[["z", col]], on="z", how="left")
            ax.plot(pts["z"], pts[col], "o", ms=4, mfc="none", mec="#1a1a19", mew=0.8, label=f"flagged ({len(fl)})")
        if logy:
            ax.set_yscale("log")
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if handles and ax is axes[0]:
            ax.legend(fontsize=6, ncol=14, loc="lower right", frameon=False)
        elif labels and labels[-1].startswith("flagged"):
            ax.legend(handles[-1:], labels[-1:], fontsize=6, loc="lower right", frameon=False)
    axes[-1].set_xlabel("z (slice)")
    fig.suptitle("Slice quality: metrics per tile over z, flagged slices circled (seams: grey lines)", fontsize=10)
    fig.tight_layout()
    atomic_write(path, lambda t: fig.savefig(t, dpi=100))
    plt.close(fig)


# --- contact sheets ---------------------------------------------------------------------------------------

WINDOW_REASONS = ("focus", "noise", "scanline", "curtain")


def to8(imgs, together=False):
    """The images as 8 bit: each over its own 0.5-99.5 percentiles (structure is compared, whatever the
    brightness), or ``together`` over the percentiles of all of them (brightness is compared)."""
    def scale(i, lo, hi):
        return np.clip((np.asarray(i, np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    if together:
        lo, hi = np.percentile(np.concatenate([np.asarray(i, np.float32).ravel()[::7] for i in imgs]), [0.5, 99.5])
        return [scale(i, lo, hi) for i in imgs]
    return [scale(i, *np.percentile(np.asarray(i).ravel()[::7], [0.5, 99.5])) for i in imgs]


def label(img, text):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
    cv2.rectangle(img, (0, 0), (img.shape[1], 18), (255, 255, 255), -1)
    cv2.putText(img, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1, cv2.LINE_AA)
    return img


def sheets(cfg, out, flags, worst_win, grids, slices, top, panel=512):
    """One PNG per reason: the top flagged tile slices (one per z), each next to its previous and next slice."""
    thumbs, cache = Thumbs(cfg), StackCache(cfg["raw_dir"])
    sp = step_path(cfg, "preview", "stats.csv")
    stats = pd.read_csv(sp, dtype={"tile": str}).set_index(["z", "tile"])[["p0_5", "p99_5"]] if sp.exists() else None
    rows = slices.set_index(["z", "tile"])
    nb = neighbours(slices)
    written = []
    for reason, g in flags[~flags["reason"].isin(["excluded", "missing_tile", "align_rejected"])].groupby("reason"):
        g = g.assign(dist=(g["value"] - NORMAL.get(reason, 0.0)).abs()).sort_values("dist", ascending=False)
        g = g.drop_duplicates("z").head(top)
        strips = []
        for f in g.itertuples():
            tile = f.worst_tile if f.tile == "all" else f.tile
            if (f.z, tile) not in nb:
                continue
            zp, zn = nb[(f.z, tile)]
            trio = [z for z in (zp, f.z, zn) if z is not None]
            seg = int(rows.loc[(f.z, tile), "segment"])
            w = worst_win.get((f.z, tile), {}).get(reason)
            if reason in WINDOW_REASONS and w is not None:
                size, tops, lefts = grids[seg]
                y0, x0 = tops[w // len(lefts)], lefts[w % len(lefts)]
                imgs = to8([cache.read(rows.loc[(z, tile)], slice(y0, y0 + size), slice(x0, x0 + size)) for z in trio])
                where = f"window x{x0} y{y0}, each panel on its own grey scale"
            else:
                imgs = [thumbs.get(z, tile) for z in trio]
                where = "whole tile, each panel on its own grey scale"
                if any(i is None for i in imgs):
                    continue
                if reason == "brightness_jump" and stats is not None and all((z, tile) in stats.index for z in trio):
                    # the thumbnails back in raw values (each was scaled to its own percentiles), on one grey scale
                    imgs = to8([thumb_values(i, *stats.loc[(z, tile)]) for z, i in zip(trio, imgs)], together=True)
                    where = "whole tile, one grey scale for the three"
            cells = []
            for z, im in zip(trio, imgs):
                s = panel / max(im.shape)
                im = cv2.resize(im, (max(1, round(im.shape[1] * s)), max(1, round(im.shape[0] * s))), interpolation=cv2.INTER_AREA)
                canvas = np.full((panel, panel, 3), 255, np.uint8)
                canvas[:im.shape[0], :im.shape[1]] = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                text = f"z {z} tile {tile} {reason} {f.value:.3g} ({f.severity}) {where}" if z == f.z else f"z {z}"
                cells.append(label(canvas, text))
                cells.append(np.full((panel, 4, 3), 255, np.uint8))
            strips.append(np.hstack(cells[:-1]))
            strips.append(np.full((6, strips[-1].shape[1], 3), 255, np.uint8))
        if strips:
            width = max(s.shape[1] for s in strips)
            strips = [np.pad(s, ((0, 0), (0, width - s.shape[1]), (0, 0)), constant_values=255) for s in strips]
            path = out / "slice_quality" / f"sheet_{reason}.png"
            atomic_write(path, lambda t: cv2.imwrite(t, np.vstack(strips[:-1])), suffix=".tmp.png")
            written.append(path.name)
    return written


def merge(cfg, args):
    out = out_dir(cfg, args)
    slices = load_slices(cfg)
    all_slices = load_slices(cfg, include_excluded=True)
    sett = settings(cfg, args)
    segs = load_chunks(out / "slice_quality" / "chunks", slices, sett)
    q, worst_win, grids = measure_table(segs, slices, args)
    q = add_tables(cfg, q, args).sort_values(["z", "tile"]).reset_index(drop=True)
    flags = make_flags(q, all_slices, args)
    atomic_write(out / "slice_quality.csv", lambda t: q.round(5).to_csv(t, index=False))
    atomic_write(out / "slice_flags.csv", lambda t: flags.drop(columns=["worst_tile"]).round(5).to_csv(t, index=False))
    seams = slices.loc[slices["seam"], "z"].unique().tolist()
    plot(out / "slice_quality.png", q, flags, seams)
    counts = {r: {"minor": int(((flags["reason"] == r) & (flags["severity"] == "minor")).sum()),
                  "major": int(((flags["reason"] == r) & (flags["severity"] == "major")).sum()),
                  "slices": int(flags.loc[flags["reason"] == r, "z"].nunique())}
              for r in sorted(flags["reason"].unique())}
    summary = {"slices": int(q["z"].nunique()), "tile_slices": int(len(q)), "flagged_slices": int(flags["z"].nunique()),
               "counts": counts, "limits": {r: {"metric": c, "direction": d, "limit": getattr(args, f"limit_{r}", None) or lim}
                                            for r, (c, d, lim) in LIMITS.items()},
               "settings": {**sett, "trend": args.trend, "z_score": args.z_score, "curtain_ratio": args.curtain_ratio},
               "thumbnails": bool(Thumbs(cfg).index)}
    (out / "slice_quality").mkdir(exist_ok=True)
    if not args.no_sheets:
        summary["sheets"] = sheets(cfg, out, flags, worst_win, grids, slices, args.top)
    atomic_write(out / "slice_quality" / "summary.json", lambda t: Path(t).write_text(json.dumps(summary, indent=1)))
    print(f"slice_quality: {summary['slices']} slices, {summary['flagged_slices']} with a flag: "
          + ", ".join(f"{r} {c['slices']}" for r, c in counts.items()))
    print(f"slice_quality: wrote {out}/slice_quality.csv, slice_flags.csv, slice_quality.png, slice_quality/")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("command", choices=["run", "merge"])
    p.add_argument("--config", required=True)
    p.add_argument("--raw-dir")
    p.add_argument("--output-dir")
    p.add_argument("--out", help="folder for the outputs (default <output_dir>/qc)")
    p.add_argument("--task-id", type=int)
    p.add_argument("--num-tasks", type=int)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--window", type=int, default=512, help="window size, full-resolution px")
    p.add_argument("--bands", type=int, default=3, help="full-width bands read per tile slice")
    p.add_argument("--chunk-slices", type=int, default=50)
    p.add_argument("--min-tissue", type=float, default=0.9, help="tissue share a window needs to be measured")
    p.add_argument("--patch-iqr", type=float, default=1.0, help="patch: this many interquartile ranges from both neighbours")
    p.add_argument("--trend", type=int, default=51, help="running-median window along z (slices)")
    p.add_argument("--z-score", type=float, default=5.0, help="robust z-score a trend flag also needs")
    p.add_argument("--curtain-ratio", type=float, default=3.0, help="v/h above which a window counts as curtained")
    for reason, (col, _, limit) in LIMITS.items():
        p.add_argument(f"--limit-{reason.replace('_', '-')}", type=float, help=f"limit on {col} (default {limit})")
    p.add_argument("--top", type=int, default=12, help="flagged slices per reason on the contact sheets")
    p.add_argument("--no-sheets", action="store_true")
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.raw_dir, args.output_dir, step_defaults("destreak", "preview"))
    return run(cfg, args) if args.command == "run" else merge(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
