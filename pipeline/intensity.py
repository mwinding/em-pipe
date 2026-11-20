"""Display levels per (z, tile) without brightness seams between tiles or flicker over z.

Each tile's lo/hi start from its preview statistics, smoothed over z with a running median in
blocks that restart at segment changes (and at seams with ``break_at_seams``). With
``balance_tiles``, neighbouring tiles are compared where they overlap: from the preview thumbnails
every ``balance_every`` slices (``balance_source: thumbnails``), or from the relations the overlaps
step measured on the raw tiles (``balance_source: overlaps``; every slice it measured). A
least-squares solve per slice gives per-tile gain/offset onto a common scale (mean gain 1, mean
offset 0), smoothed over ``balance_smooth_slices``, and one common window per z is mapped back into
each tile's raw values. Writes intensity/levels.csv and intensity.png.

With ``patchgrid.enabled`` (and ``balance_source: overlaps``) the tiles are compared in their
patchgrid-corrected values u (flat-field, then the tile position's within-tile shape, which render
applies too): the solve per slice takes every patch bin of work/overlaps/patches.csv, its relation
carried over to u (patchgrid.shape_corrected) and matched at its own centre, its gain weighted as
in the patchgrid solve, instead of one relation per overlap; levels are then windows on u.

Starting windows (``intensity.window``): median (default) puts the tissue median (preview t_p50,
background left out) at grey ``target_median`` and makes the tissue p5-p95 span ``target_span``
greys, so brightness and contrast stay the same whatever is in the field of view (35h: percentile
windows rendered the tissue median at 145-215 across tile layout changes, as a few % of very dark
pixels moved the black point; median-anchored windows at 134-186, no steps). percentile maps the
preview percentiles lo_percentile / hi_percentile (all pixels) to 0 / 255. reference sets every
slice's window so that the reference tissue (neuropil, pipeline/refmask.py) comes out at the grey
``reference.median`` with the spread ``reference.contrast`` in greys, by ``reference.measure``:
p5_p95, the width between the 5th and 95th percentile of the slice's neuropil (its histogram's width;
the default), or lc32, its local contrast (the mean SD in 8 px windows at 32 nm). The percentiles a
window clips at are not chosen: they follow from matching the reference tissue, and the QC reports
them. From the refstats step's per-slice neuropil median m and spread c on the common
scale the tiles are balanced to (refstats measures it as render applies it), smoothed over z with a
running median of ``reference.smooth_slices`` (restarting at segment changes), interpolated between
the sampled slices (held at a segment's ends) and averaged over ``reference.ramp_slices``, so the
window changes smoothly along z. The window is hi - lo = 255 c / C_ref, lo = m - median_ref (hi - lo)
/ 255; tile balance and per-slice jumps as for the other windows. A segment without reference tissue
keeps the median window. QC: qc/intensity_reference[_<output>].png and .json: predicted neuropil
grey and contrast per sampled slice, clipped tissue fractions, the largest slice-to-slice change of
the neuropil's grey and contrast (flicker), and the step across every segment change next to the
step with the previous levels.

``intensity.output``: a name for a variant's levels (work/intensity/<output>/levels.csv, which
render then reads), so a variant config doesn't replace the base config's levels.
"""

import itertools
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

from . import background, patchgrid, shading, transforms  # noqa: E402
from .cli import atomic_write, base_parser, setup  # noqa: E402
from .config import deep_merge, get, qc_path, step_path  # noqa: E402
from .preview import PERCENTILES, TISSUE, plot_tiles, save, thumb_to_tile, thumb_values  # noqa: E402
from .slices import load_slices  # noqa: E402

log = logging.getLogger(__name__)

DEFAULTS = {
    "intensity": {
        "window": "median",       # median: tissue median -> target_median, tissue p5-p95 over
                                  # target_span greys | percentile: lo/hi_percentile -> 0 / 255
        "target_median": 140,     # median: grey of the tissue median (0-255)
        "target_span": 160,       # median: greys between the tissue p5 and p95
        "lo_percentile": 0.5,     # percentile: preview percentile rendered as 0 (all: 0.5, 1, 50, 99, 99.5;
                                  # tissue: 1, 5, 25, 50, 75, 95, 99)
        "hi_percentile": 99.5,    # percentile: preview percentile rendered as 255
        "percentile_of": "all",   # percentile: of all pixels, or of the tissue only (background left out)
        "smooth_slices": 51,      # running-median window over z, in selected slices
        "break_at_seams": True,   # restart smoothing at seams (it always restarts at segment changes)
        "balance_tiles": True,    # match brightness of overlapping tiles (needs stitch/tiles.csv)
        "balance_source": "thumbnails",   # thumbnails: preview thumbnails every balance_every slices |
                                          # overlaps: the overlaps step's relations from the raw tiles
        "balance_every": 10,      # thumbnails: measure tile overlaps every N selected slices
        "balance_tissue_only": True,   # thumbnails: compare overlaps on tissue pixels only (background left out)
        "balance_smooth_slices": None,   # running median of the tile gains/offsets over this many selected
                                         # slices; null: smooth_slices
        # Follow slice-to-slice brightness jumps (tile median minus its running median), which the
        # smoothing above would otherwise leave as flicker when scrolling through z. true: each tile
        # its own jump; slice: one jump per slice (the median over its tiles, relative to their
        # windows), which keeps tiles matched when the balance follows every slice; false: off.
        "per_slice": True,
        "output": None,           # variant name: levels in work/intensity/<output>/ (render reads them there)
        # window: reference (pipeline/refmask.py, refstats): the reference tissue's grey and contrast
        "reference": {
            "median": None,       # grey of the reference tissue's median (0-255)
            "contrast": None,     # its spread in greys, as measured by ``measure``
            # p5_p95: the width between the 5th and 95th percentile of the slice's reference tissue (its
            # histogram's width) | lc32: its local contrast, the mean SD in 8 px windows at 32 nm
            "measure": "p5_p95",
            # segment: one window per block (a segment, split at ``breaks``) from the median of its sampled
            # slices, so nothing changes from slice to slice within it | slice: follow the samples along z
            # (running median over smooth_slices, interpolated, running mean over ramp_slices)
            "per": "segment",
            "breaks": [],         # z where a new block starts within a segment (e.g. a setting changed mid-run)
            # tissue: the window's position follows every slice's tissue (as the median window), shifted per
            # block so the reference tissue's median is at reference.median | reference: set from the
            # reference tissue's own median (per block, or along z with per: slice)
            "brightness": "tissue",
            "smooth_slices": 601,   # per: slice: running median of the sampled stats over z (slices)
            "ramp_slices": 129,   # running mean over z after interpolation (no kinks at the samples)
            "min_cells": 20,      # samples with fewer reference cells don't count
            "boundary_slices": 300,   # QC: samples within this many slices each side of a segment change
            "band_pct": 5.0,      # QC: target band, +- % of median and contrast
            "clip_warn": 0.01,    # QC: warn when more than this fraction of tissue is at 0 or at 255
            "flicker_warn": 1.0,  # QC: warn when the neuropil grey changes by more than this between slices
        },
    },
}

MIN_OVERLAP_PX = 100   # thumbnail pixels a tile pair must share to be compared


def percentile_column(p, tissue=False):
    """stats.csv column for a percentile, e.g. 0.5 -> 'p0_5' (tissue: 5 -> 't_p5')."""
    known = TISSUE if tissue else PERCENTILES
    col = ("t_p" if tissue else "p") + f"{float(p):g}".replace(".", "_")
    if col not in known:
        raise ValueError(f"intensity percentile {p!r} is not in preview/stats.csv; "
                         f"use one of {list(known.values())}")
    return col


def running_median(s, window):
    return s.rolling(int(window), center=True, min_periods=1).median()


def load_inputs(cfg, need):
    """Selected (z, tile) rows with their preview statistics ``need``, smoothing block and position in z."""
    slices = load_slices(cfg)
    stats = pd.read_csv(step_path(cfg, "preview", "stats.csv"), dtype={"tile": str})
    absent = [c for c in need if c not in stats.columns]
    if absent:
        raise RuntimeError(f"preview/stats.csv has no {', '.join(absent)} (made by an older preview): "
                           "re-run 'preview run' and 'preview merge', or set intensity.window: percentile")
    cols = list(dict.fromkeys(["z", "tile", *need, "p0_5", "p50", "p99_5"]))
    df = slices[["z", "tile", "height", "width", "segment", "seam"]].merge(stats[cols], on=["z", "tile"], how="left")
    missing = df[list(need)].isna().any(axis=1)
    if missing.any():
        raise RuntimeError(f"{int(missing.sum())} selected (z, tile) are missing from preview/stats.csv; "
                           "run 'preview run' and 'preview merge'")
    per_z = df.groupby("z").agg(segment=("segment", "first"), seam=("seam", "any"))
    new = per_z["segment"].ne(per_z["segment"].shift())
    if cfg["intensity"]["break_at_seams"]:
        new |= per_z["seam"]
    per_z["block"] = new.cumsum()
    per_z["pos"] = np.arange(len(per_z))   # neighbours are by position: selected z may have holes
    df = df.merge(per_z[["block", "pos"]], left_on="z", right_index=True)
    return df.sort_values(["z", "tile"]).reset_index(drop=True)


def levels_path(cfg, output=None):
    """work/intensity/levels.csv, or work/intensity/<output>/levels.csv for intensity.output."""
    out = output if output is not None else (cfg.get("intensity") or {}).get("output")
    return step_path(cfg, "intensity", *([str(out)] if out else []), "levels.csv")


def _qc_name(cfg, name):
    out = (cfg.get("intensity") or {}).get("output")
    return f"{name}_{out}" if out else name


WINDOWS = ("median", "percentile", "reference")


def compute_levels(cfg):
    """DataFrame of every selected (z, tile) with lo, hi (and gain g, offset o when balanced)."""
    c = cfg["intensity"]
    if c["window"] not in WINDOWS:
        raise ValueError(f"intensity.window must be one of {WINDOWS}, not {c['window']!r}")
    df = balanced_scale(cfg)
    # One jump per slice, relative to the starting windows: a shift in the tiles' own values, settled before
    # window: reference changes the windows' width (it follows the raw level, not the contrast).
    if c["per_slice"] == "slice":
        rel = df["jump"] / (df["hi"] - df["lo"])
        df["jump"] = rel.groupby(df["z"]).transform("median") * (df["hi"] - df["lo"])
    if c["window"] == "reference":
        df = reference_windows(cfg, df)
    if c["per_slice"]:
        df["lo"] += df["jump"]
        df["hi"] += df["jump"]
    return df


def balanced_scale(cfg):
    """Every selected (z, tile) with its starting window lo, hi (median window for window: reference),
    jump, and tile gain g and offset o onto the slice's common scale (NaN where unbalanced).

    The common scale is g u + o, u a tile's values as render corrects them (flat-field, patchgrid
    shape); intensity's windows are windows on it, and refstats measures there."""
    c = cfg["intensity"]
    if c["window"] not in WINDOWS:
        raise ValueError(f"intensity.window must be one of {WINDOWS}, not {c['window']!r}")
    if int(c["smooth_slices"]) < 1 or int(c["balance_every"]) < 1:
        raise ValueError("intensity.smooth_slices and balance_every must be at least 1")
    if c["window"] in ("median", "reference"):
        tm, ts = float(c["target_median"]), float(c["target_span"])
        if not (0 < tm < 255 and ts > 0):
            raise ValueError("intensity.target_median must be in (0, 255) and target_span positive")
        df = load_inputs(cfg, ["t_p5", "t_p50", "t_p95"])
        spread = np.maximum(df["t_p95"] - df["t_p5"], 1.0)
        df["lo_raw"] = df["t_p50"] - tm / ts * spread
        df["hi_raw"] = df["t_p50"] + (255 - tm) / ts * spread
        mid = "t_p50"
    else:
        if c["percentile_of"] not in ("all", "tissue"):
            raise ValueError(f"intensity.percentile_of must be all or tissue, not {c['percentile_of']!r}")
        tissue = c["percentile_of"] == "tissue"
        lo_col = percentile_column(c["lo_percentile"], tissue)
        hi_col = percentile_column(c["hi_percentile"], tissue)
        if float(c["lo_percentile"]) >= float(c["hi_percentile"]):
            raise ValueError("intensity.lo_percentile must be below hi_percentile")
        mid = "t_p50" if tissue else "p50"
        df = load_inputs(cfg, list(dict.fromkeys([lo_col, hi_col, mid])))
        df["lo_raw"], df["hi_raw"] = df[lo_col], df[hi_col]
    by_run = df.groupby(["tile", "block"])
    for col in ("lo", "hi"):
        df[col] = by_run[f"{col}_raw"].transform(lambda s: running_median(s, c["smooth_slices"]))
    # Raw FIB-SEM tiles (ours) jump by 1-2 % of the display range from slice to slice, independently of each
    # other; tissue changes slowly, so the tile median's deviation from its trend is that jump.
    # (A column, so it stays with its rows through balance's merge.)
    df["jump"] = df[mid] - by_run[mid].transform(lambda s: running_median(s, c["smooth_slices"]))
    if c["per_slice"] not in (True, False, "slice"):
        raise ValueError(f"intensity.per_slice must be true, false or slice, not {c['per_slice']!r}")
    if c["balance_source"] not in ("thumbnails", "overlaps"):
        raise ValueError(f"intensity.balance_source must be thumbnails or overlaps, not {c['balance_source']!r}")
    df["g"], df["o"] = np.nan, np.nan
    if c["balance_tiles"]:
        df = df.drop(columns=["g", "o"])
        tiles_csv = step_path(cfg, "stitch", "tiles.csv")
        if get(cfg, "patchgrid.enabled", False) and c["balance_source"] != "overlaps":
            raise ValueError("patchgrid.enabled needs intensity.balance_source: overlaps (render applies the "
                             "patch shapes, which the thumbnail balance does not know)")
        if c["balance_source"] == "overlaps":
            meas, every = overlap_measurements(cfg)
            df = balance(cfg, df, meas, every)
        elif tiles_csv.exists():
            df = balance(cfg, df, thumbnail_measurements(cfg, df, tiles_csv, shading.load(cfg)),
                         int(c["balance_every"]))
        else:
            log.warning("%s not found: per-tile levels without tile balancing", tiles_csv)
            df["g"], df["o"] = np.nan, np.nan
    return df


# ----- window: reference ----------------------------------------------------------------------

def reference_settings(cfg):
    r = {**DEFAULTS["intensity"]["reference"], **(cfg["intensity"].get("reference") or {})}
    for key in ("median", "contrast"):
        if r[key] is None or not float(r[key]) > 0:
            raise ValueError(f"intensity.window: reference needs intensity.reference.{key} (a positive number)")
    if r["measure"] not in MEASURES:
        raise ValueError(f"intensity.reference.measure must be one of {MEASURES}, not {r['measure']!r}")
    if r["per"] not in ("segment", "slice"):
        raise ValueError(f"intensity.reference.per must be segment or slice, not {r['per']!r}")
    r["breaks"] = sorted(int(b) for b in (r.get("breaks") or []))
    if r["brightness"] not in ("tissue", "reference"):
        raise ValueError(f"intensity.reference.brightness must be tissue or reference, not {r['brightness']!r}")
    if not 0 < float(r["median"]) < 255:
        raise ValueError("intensity.reference.median must be in (0, 255)")
    return r


MEASURES = ("p5_p95", "lc32")


def spread(st, measure):
    """The reference tissue's spread per row of refstats' table: p5_p95 (rq_95 - rq_5, its histogram's
    width) or lc32 (local contrast), on the same scale as its median."""
    if measure == "p5_p95":
        return st["rq_95"] - st["rq_5"]
    if measure == "lc32":
        return st["lc32"]
    raise ValueError(f"unknown measure {measure!r} (one of {MEASURES})")


def block_ids(per_z, breaks=None):
    """Per row of per_z (sorted by z, with segment): the block its window is set over. A new block
    starts at every segment change and at the first slice at or after each z in ``breaks`` (a change
    within a segment, e.g. a microscope setting changed mid-run)."""
    z = per_z["z"].to_numpy()
    seg = per_z["segment"].to_numpy()
    new = np.r_[True, seg[1:] != seg[:-1]]
    for b in breaks or []:
        new |= (z >= b) & np.r_[True, z[:-1] < b]
    return np.cumsum(new) - 1


def window_from_stats(m, c, median_ref, contrast_ref):
    """(lo, hi) on the common scale that put a tissue of median m and contrast c (same scale) at grey
    median_ref with contrast contrast_ref: hi - lo = 255 c / contrast_ref, lo = m - median_ref (hi - lo) / 255."""
    span = 255.0 * np.asarray(c, float) / float(contrast_ref)
    lo = np.asarray(m, float) - float(median_ref) * span / 255.0
    return lo, lo + span


def grey(v, lo, hi):
    """Grey (0-255, unclipped) of value v in the window lo-hi."""
    return 255.0 * (np.asarray(v, float) - lo) / (hi - lo)


def smooth_samples(pos, values, all_pos, smooth, ramp):
    """A smooth curve over the positions all_pos (one segment) from samples (pos, values): running
    median over +-smooth/2 positions, linear interpolation (held at the ends), then a running mean
    over +-ramp/2 positions. NaN samples are left out; None when no sample is valid."""
    pos, values = np.asarray(pos, float), np.asarray(values, float)
    ok = np.isfinite(values)
    pos, values = pos[ok], values[ok]
    if not len(pos):
        return None
    order = np.argsort(pos)
    pos, values = pos[order], values[order]
    half = max(float(smooth), 1.0) / 2
    med = np.array([np.median(values[np.abs(pos - p) <= half]) for p in pos])
    curve = np.interp(all_pos, pos, med)
    k = int(ramp) // 2
    if k > 0 and len(curve) > 1:
        padded = np.pad(curve, k, mode="edge")
        csum = np.cumsum(np.r_[0.0, padded])
        curve = (csum[2 * k + 1:] - csum[:-(2 * k + 1)]) / (2 * k + 1)
    return curve


def load_refstats(cfg):
    from .refstats import stats_path
    path = stats_path(cfg, "stats.csv")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: intensity.window: reference needs the refstats step (run, merge)")
    return pd.read_csv(path)


def reference_windows(cfg, df):
    """Replace the windows by the reference windows (common scale L, H per z, mapped into each tile).

    Adds per-row columns m_s, c_s (smoothed stats), L, H (the slice's common window) and ref (False
    where a segment had no reference tissue and keeps the starting window)."""
    r = reference_settings(cfg)
    st = load_refstats(cfg)
    st = st[st["n_cells"] >= int(r["min_cells"])]
    per_z = df.groupby("z").agg(segment=("segment", "first"), pos=("pos", "first")).reset_index()
    st = st.assign(spread=spread(st, r["measure"]))
    per_z = per_z.merge(st[["z", "median", "spread"]], on="z", how="left")
    per_z["block"] = block_ids(per_z, r["breaks"])
    cols = {k: np.full(len(per_z), np.nan) for k in ("m_s", "c_s")}
    for b, g in per_z.groupby("block"):
        ok = g["median"].notna() & g["spread"].notna()
        span = f"block {b} (segment {g['segment'].iloc[0]}, z {g['z'].min()}-{g['z'].max()})"
        if not ok.any():
            log.warning("%s has no sampled slice with reference tissue: it keeps the median window", span)
            continue
        if ok.sum() < 3:
            log.warning("%s: only %d sampled slices with reference tissue", span, int(ok.sum()))
        if r["per"] == "segment":
            m, cc = float(g.loc[ok, "median"].median()), float(g.loc[ok, "spread"].median())
        else:
            m = smooth_samples(g["pos"], g["median"], g["pos"].to_numpy(float), r["smooth_slices"], r["ramp_slices"])
            cc = smooth_samples(g["pos"], g["spread"], g["pos"].to_numpy(float), r["smooth_slices"], r["ramp_slices"])
        cols["m_s"][g.index], cols["c_s"][g.index] = m, cc
    per_z["m_s"], per_z["c_s"] = cols["m_s"], cols["c_s"]
    if r["brightness"] == "tissue":
        per_z = _tissue_brightness(cfg, df, per_z, r)
    else:
        per_z["L"], per_z["H"] = window_from_stats(per_z["m_s"], per_z["c_s"], r["median"], r["contrast"])
    df = df.drop(columns=[c for c in ("m_s", "c_s", "L", "H") if c in df]).merge(
        per_z[["z", "m_s", "c_s", "L", "H"]], on="z", how="left")
    df["ref"] = df["L"].notna()
    g = df["g"].fillna(1.0)
    o = df["o"].fillna(0.0)
    df.loc[df["ref"], "lo"] = ((df["L"] - o) / g)[df["ref"]]
    df.loc[df["ref"], "hi"] = ((df["H"] - o) / g)[df["ref"]]
    log.info("reference windows for %d of %d slices (median %g, %s %g)", int(per_z["L"].notna().sum()), len(per_z),
             float(r["median"]), r["measure"], float(r["contrast"]))
    return df


def _tissue_brightness(cfg, df, per_z, r):
    """brightness: tissue. The window's width per block from the reference tissue (c_s, as for
    brightness: reference), its position following every slice's starting window (all tissue, as the
    median window places it), shifted once per block so the reference tissue's median comes out at
    reference.median over the block's sampled slices (with the per-slice jumps render will add)."""
    tm = float(cfg["intensity"]["target_median"])
    w0 = common_window(df)                                   # starting windows, before the jumps
    M = (w0["Lc"] + tm / 255.0 * (w0["Hc"] - w0["Lc"])).reindex(per_z["z"]).to_numpy()
    if cfg["intensity"]["per_slice"]:
        J = (df["g"].fillna(1.0) * df["jump"]).groupby(df["z"]).median().reindex(per_z["z"]).to_numpy()
    else:
        J = np.zeros(len(per_z))
    W = 255.0 * per_z["c_s"].to_numpy(float) / float(r["contrast"])
    L = np.full(len(per_z), np.nan)
    for b, g in per_z.groupby("block"):
        i = g.index.to_numpy()
        ok = g["median"].notna().to_numpy() & np.isfinite(W[i])
        if not ok.any():
            continue
        a = np.median(float(r["median"]) * W[i][ok] / 255.0 - (g["median"].to_numpy()[ok] - M[i][ok] - J[i][ok]))
        L[i] = M[i] - a
    per_z["L"], per_z["H"] = L, L + W
    per_z["m_s"] = per_z["L"] + float(r["median"]) * W / 255.0     # the value each window puts at reference.median
    return per_z


def common_window(df):
    """Per z: the common-scale window (median over tiles of g lo + o, g hi + o; g 1, o 0 unbalanced)."""
    g, o = df["g"].fillna(1.0), df["o"].fillna(0.0)
    w = pd.DataFrame({"z": df["z"], "Lc": g * df["lo"] + o, "Hc": g * df["hi"] + o})
    return w.groupby("z")[["Lc", "Hc"]].median()


def _cdf(x, row, qcols):
    """Fraction of tissue below x from a row's tissue quantiles (columns tq_<percent>)."""
    ps = np.array([float(c[3:]) for c in qcols]) / 100
    vs = row[qcols].to_numpy(float)
    if not np.isfinite(vs).all():
        return np.nan
    return float(np.interp(x, vs, ps, left=0.0, right=1.0))


def flicker(per_z, Lc, Hc):
    """Per consecutive selected slices: |change| of the smoothed neuropil's grey and contrast when the
    window changes from one slice to the next (the tissue itself held), and whether they straddle a
    segment change."""
    m, c = per_z["m_s"].to_numpy(float), per_z["c_s"].to_numpy(float)
    L, H = Lc.to_numpy(float), Hc.to_numpy(float)
    d_med = np.abs(grey(m[:-1], L[1:], H[1:]) - grey(m[:-1], L[:-1], H[:-1]))
    d_con = np.abs(255 * c[:-1] / (H[1:] - L[1:]) - 255 * c[:-1] / (H[:-1] - L[:-1]))
    seg = per_z["block" if "block" in per_z else "segment"].to_numpy()
    return pd.DataFrame({"z": per_z["z"].to_numpy()[1:], "d_median": d_med, "d_contrast": d_con,
                         "boundary": seg[1:] != seg[:-1]})


def reference_qc(cfg, df, old=None):
    """QC of reference windows: predicted greys, clipping, flicker and segment-change steps (dict),
    the per-sample table and the flicker table. ``old``: the previous levels (z, tile, lo, hi)."""
    r = reference_settings(cfg)
    st = load_refstats(cfg)
    per_z = df.groupby("z").agg(segment=("segment", "first"), pos=("pos", "first"), m_s=("m_s", "first"),
                                c_s=("c_s", "first"), L=("L", "first"), H=("H", "first")).reset_index()
    new_w = common_window(df)
    windows = {"new": new_w}
    if old is not None:
        o = df[["z", "tile", "g", "o"]].merge(old[["z", "tile", "lo", "hi"]], on=["z", "tile"], how="inner")
        if len(o):
            windows["old"] = common_window(o)
    qcols = [c for c in st.columns if c.startswith("tq_")]
    per_z["block"] = block_ids(per_z, r["breaks"])
    samp = st.merge(per_z[["z", "segment", "block", "pos"]], on="z", how="inner")
    samp = samp[samp["n_cells"] >= int(r["min_cells"])].copy()
    for k, w in windows.items():
        Lw, Hw = w["Lc"].reindex(samp["z"]).to_numpy(), w["Hc"].reindex(samp["z"]).to_numpy()
        samp[f"grey_{k}"] = grey(samp["median"], Lw, Hw)
        samp[f"contrast_{k}"] = 255 * spread(samp, r["measure"]) / (Hw - Lw)
        samp[f"lc32_{k}"] = 255 * samp["lc32"] / (Hw - Lw)        # cross-check: local contrast in greys
        samp[f"p5_{k}"] = grey(samp["rq_5"], Lw, Hw)             # the tissue's 5th / 95th percentile in greys
        samp[f"p95_{k}"] = grey(samp["rq_95"], Lw, Hw)
        if qcols:
            samp[f"clip_lo_{k}"] = [_cdf(lo, row, qcols) for lo, (_, row) in zip(Lw, samp.iterrows())]
            samp[f"clip_hi_{k}"] = [1 - _cdf(hi, row, qcols) for hi, (_, row) in zip(Hw, samp.iterrows())]
    fl = {}
    for k, w in windows.items():
        fl[k] = flicker(per_z, w["Lc"].reindex(per_z["z"]).reset_index(drop=True),
                        w["Hc"].reindex(per_z["z"]).reset_index(drop=True))
    # The reference part alone (before the per-slice jumps, which the other windows have too)
    fl["new_without_jumps"] = flicker(per_z, per_z["L"], per_z["H"])
    out = {"reference": {k: r[k] for k in ("median", "contrast", "measure", "smooth_slices", "ramp_slices")},
           "samples": int(len(samp)), "slices": int(len(per_z)),
           "slices_without_reference": int(per_z["L"].isna().sum())}
    band = float(r["band_pct"]) / 100
    for k in windows:
        g_, c_ = samp[f"grey_{k}"], samp[f"contrast_{k}"]
        out[f"predicted_{k}"] = {
            "median_grey_p5_p50_p95": [round(float(v), 2) for v in np.nanpercentile(g_, [5, 50, 95])] if len(g_) else None,
            "contrast_p5_p50_p95": [round(float(v), 2) for v in np.nanpercentile(c_, [5, 50, 95])] if len(c_) else None,
            "in_band_median": float(np.mean(np.abs(g_ / float(r["median"]) - 1) <= band)) if len(g_) else None,
            "in_band_contrast": float(np.mean(np.abs(c_ / float(r["contrast"]) - 1) <= band)) if len(c_) else None,
        }
        if f"clip_lo_{k}" in samp:
            clip = samp[f"clip_lo_{k}"] + samp[f"clip_hi_{k}"]
            out[f"predicted_{k}"]["clipped_max"] = round(float(np.nanmax(clip)), 5) if len(clip) else None
            out[f"predicted_{k}"]["clipped_median"] = round(float(np.nanmedian(clip)), 5) if len(clip) else None
            out[f"predicted_{k}"]["clip_warn_slices"] = [int(z) for z in samp.loc[clip > float(r["clip_warn"]), "z"]]
    for k, f in fl.items():
        inner = f[~f["boundary"]]
        out[f"flicker_{k}"] = {"max_d_median": round(float(inner["d_median"].max()), 3) if len(inner) else None,
                               "max_d_contrast": round(float(inner["d_contrast"].max()), 3) if len(inner) else None,
                               "slices_over_warn": int((inner["d_median"] > float(r["flicker_warn"])).sum())}
    # Segment changes: each side's neuropil grey from the samples within boundary_slices of the change
    steps = []
    bpos = per_z.loc[per_z["block"].ne(per_z["block"].shift()) & (per_z.index > 0)]
    half = int(r["boundary_slices"])
    for i, row in bpos.iterrows():
        before, after = per_z.loc[i - 1], row
        side = {"before": samp[(samp["block"] == before["block"]) & (samp["pos"] > before["pos"] - half)],
                "after": samp[(samp["block"] == after["block"]) & (samp["pos"] < after["pos"] + half)]}
        rec = {"z": int(after["z"]), "segments": [int(before["segment"]), int(after["segment"])],
               "samples": [int(len(side["before"])), int(len(side["after"]))]}
        for k in windows:
            if all(len(v) for v in side.values()):
                rec[f"step_grey_{k}"] = round(float(side["after"][f"grey_{k}"].median() - side["before"][f"grey_{k}"].median()), 2)
                rec[f"step_contrast_{k}"] = round(float(side["after"][f"contrast_{k}"].median()
                                                        - side["before"][f"contrast_{k}"].median()), 2)
        rec["window_step_grey_new"] = round(float(fl["new"].loc[fl["new"]["z"] == after["z"], "d_median"].iloc[0]), 2)
        if "step_grey_old" in rec and "step_grey_new" in rec:
            rec["bigger_than_old"] = bool(abs(rec["step_grey_new"]) > abs(rec["step_grey_old"]))
        steps.append(rec)
    out["segment_changes"] = steps
    out["flags"] = [f"segment change at z {s['z']}: neuropil step {s['step_grey_new']} greys (old {s['step_grey_old']})"
                    for s in steps if s.get("bigger_than_old")]
    fw = out["flicker_new_without_jumps"]["max_d_median"]
    if fw is not None and fw > float(r["flicker_warn"]):
        out["flags"].append(f"the window moves the neuropil grey by up to {fw} greys between slices away from segment "
                            "changes (not counting the per-slice jumps, which the other windows apply too)")
    return out, samp, fl


def plot_reference(cfg, samp, fl, qc, path):
    r = reference_settings(cfg)
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 12), layout="constrained")
    band = float(r["band_pct"]) / 100
    for ax, col, target, label in ((axes[0], "grey", float(r["median"]), "neuropil median (grey)"),
                                   (axes[1], "contrast", float(r["contrast"]), f"neuropil {r['measure']} (greys)")):
        ax.axhspan(target * (1 - band), target * (1 + band), color="#1baf7a", alpha=0.15, lw=0,
                   label=f"target +-{r['band_pct']:g} %")
        ax.axhline(target, color="#1baf7a", lw=1)
        if f"{col}_old" in samp:
            ax.plot(samp["z"], samp[f"{col}_old"], "o", ms=3, color="0.6", label="previous levels")
        ax.plot(samp["z"], samp[f"{col}_new"], "o", ms=3, color="#2a78d6", label="reference window")
        ax.set_ylabel(label)
        ax.legend(fontsize=8, loc="upper right")
    ax = axes[2]
    if "clip_lo_new" in samp:
        for k, colr in (("old", "0.6"), ("new", "#eb6834")):
            if f"clip_lo_{k}" in samp:
                ax.plot(samp["z"], 100 * (samp[f"clip_lo_{k}"] + samp[f"clip_hi_{k}"]), "o-", ms=3, lw=0.8, color=colr,
                        label=f"{'previous levels' if k == 'old' else 'reference window'}")
        ax.axhline(100 * float(r["clip_warn"]), color="#e34948", lw=1, ls="--", label="warning")
        ax.legend(fontsize=8, loc="upper right")
    ax.set_ylabel("tissue clipped (%)")
    ax = axes[3]
    for k, colr, lab in (("new_without_jumps", "#2a78d6", "reference window (without the per-slice jumps)"),):
        if k in fl:
            ax.plot(fl[k]["z"], fl[k]["d_median"], lw=0.6, color=colr, label=lab)
    ax.axhline(float(r["flicker_warn"]), color="#e34948", lw=1, ls="--", label="1 grey")
    for z in fl["new"].loc[fl["new"]["boundary"], "z"]:
        for a in axes:
            a.axvline(z, color="0.75", lw=0.6)
    ax.set_ylabel("window change between\nslices (greys)")
    ax.set_xlabel("z")
    ax.legend(fontsize=8, loc="upper right")
    axes[0].set_title("Reference window: predicted neuropil at the sampled slices (grey lines: segment changes)")
    save(fig, path)


def _measurements(found):
    # Float columns even when nothing was measured: object columns break the assignments in balance.
    return pd.DataFrame(found, columns=["z", "tile", "g", "o"]).astype({"z": int, "g": float, "o": float})


def thumbnail_measurements(cfg, df, tiles_csv, fields=None):
    """Per-tile gain/offset (z, tile, g, o) from the preview thumbnails' overlaps every balance_every slices.

    fields: shading fields {(segment, tile): gain} divided out of the thumbnails first."""
    c = cfg["intensity"]
    thumbs_dir = step_path(cfg, "preview", "thumbs")
    stitch = transforms.read_csv(tiles_csv, ["z", "tile"])
    df = df.merge(pd.read_csv(thumbs_dir / "index.csv", dtype={"tile": str}), on=["z", "tile"], how="left")
    first = df.groupby("block")["pos"].transform("min")
    sampled = df[(df["pos"] - first) % int(c["balance_every"]) == 0]
    found = []
    for z, rows in sampled.groupby("z"):
        meas_z = measure(rows, stitch, thumbs_dir, fields or {}, c["balance_tissue_only"])
        found += [{"z": z, "tile": t, "g": g, "o": o} for t, (g, o) in meas_z.items()]
    meas = _measurements(found)
    log.info("tile balance measured at %d of %d sampled slices (thumbnails)", meas["z"].nunique(), sampled["z"].nunique())
    return meas


GAIN_CNR = 10.0   # overlaps whose spread is less than this many fit-rms (e.g. only resin) constrain the gain less


def overlap_measurements(cfg):
    """(per-tile gain/offset (z, tile, g, o), every) from the overlaps step's relations (with
    patchgrid: its patch bins, ``patch_measurements``)."""
    if get(cfg, "patchgrid.enabled", False):
        return patch_measurements(cfg)
    path = step_path(cfg, "overlaps", "relations.csv")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: balance_source overlaps needs the overlaps step (run, merge)")
    rel = pd.read_csv(path, dtype={"a": str, "b": str})
    found = []
    for z, g in rel.groupby("z"):
        rels = [(r.a, r.b, r.gain, r.offset, r.n, min(1.0, r.cnr / GAIN_CNR), r.median_b) for r in g.itertuples()]
        found += [{"z": z, "tile": t, "g": gg, "o": oo} for t, (gg, oo) in solve(connected(rels)).items()]
    meas = _measurements(found)
    every = int(get(cfg, "overlaps.every", 1) or 1)
    log.info("tile balance from %d overlap relations at %d slices (every %d)", len(rel), meas["z"].nunique(), every)
    return meas, every


def patch_measurements(cfg):
    """(per-tile gain/offset (z, tile, g, o) on patchgrid-corrected values, every) from the overlaps
    step's patch bins, each carried over to the corrected values by the tile shapes."""
    p = deep_merge(patchgrid.DEFAULTS, {"patchgrid": cfg.get("patchgrid") or {}})["patchgrid"]
    shapes, R = patchgrid.load(cfg)
    patchgrid.check_current(cfg)
    bins = patchgrid.shape_corrected(patchgrid.read_bins(cfg), shapes, R)
    bins["gw"] = patchgrid.bin_weights(bins, p["gain_power"], p["gain_cnr_min"])[1]
    found = []
    for z, g in bins.groupby("z"):
        rels = [(r.a, r.b, r.gain, r.offset, r.n, r.gw, r.centre) for r in g.itertuples()]
        found += [{"z": z, "tile": t, "g": gg, "o": oo} for t, (gg, oo) in solve(connected(rels)).items()]
    meas = _measurements(found)
    every = int(get(cfg, "overlaps.every", 1) or 1)
    log.info("tile balance from %d patch bins (shapes applied, R = %.0f) at %d slices (every %d)", len(bins), R,
             meas["z"].nunique(), every)
    return meas, every


def balance(cfg, df, meas, every):
    """Replace lo/hi by one common window per z, mapped through per-tile gain/offset.

    meas: (z, tile, g, o) measured every ``every`` selected slices; smoothed over
    balance_smooth_slices (default smooth_slices) and interpolated in between."""
    c = cfg["intensity"]
    df = df.merge(meas, on=["z", "tile"], how="left")
    window = max(1, round((c["balance_smooth_slices"] or c["smooth_slices"]) / every))
    # Smoothed (and interpolated) is each tile's map g v + o at two fixed intensities of its own,
    # r1, r2 (its median window ends in the block), not g and o apart: with noisy gains the offset
    # (the map's value at 0, far below the data) swings with the gain, and separate medians would
    # pair one slice's gain with another slice's offset.
    for _, grp in df.groupby(["tile", "block"]):
        ok = grp["g"].notna()
        if ok.any():
            r1, r2 = grp["lo"].median(), grp["hi"].median()
            at = {r: grp.loc[ok, "g"] * r + grp.loc[ok, "o"] for r in (r1, r2)}
            m1, m2 = (np.interp(grp["pos"], grp.loc[ok, "pos"], running_median(at[r], window)) for r in (r1, r2))
            df.loc[grp.index, "g"] = (m2 - m1) / (r2 - r1)
            df.loc[grp.index, "o"] = m1 - df.loc[grp.index, "g"] * r1
    bal = df["g"].notna()
    if not bal.all():
        log.warning("%d (z, tile) have no overlap measurement in their smoothing block: left unbalanced",
                    int((~bal).sum()))
    b = df[bal]
    for col in ("lo", "hi"):
        common = (b["g"] * b[col] + b["o"]).groupby(b["z"]).transform("median")
        df.loc[bal, col] = (common - b["o"]) / b["g"]
    return df


def measure(rows, stitch, thumbs_dir, fields=None, tissue_only=True):
    """{tile: (gain, offset)} at one z from its thumbnails' overlaps; {} if no tile pair overlaps.

    Thumbnails are divided by their shading field (``fields``, by (segment, tile)) when there is
    one; with ``tissue_only`` background pixels (resin, off-sample) are left out of the comparison,
    so an overlap strip that is mostly resin can't set a tile's gain from the resin's brightness.
    """
    z = int(rows["z"].iloc[0])
    imgs, to_montage = {}, {}
    for r in rows.itertuples():
        if pd.isna(r.npy) or (z, r.tile) not in stitch:
            continue
        t8 = np.asarray(np.load(thumbs_dir / r.npy, mmap_mode="r")[int(r.i)])
        v = thumb_values(t8, r.p0_5, r.p99_5)
        field = (fields or {}).get((int(r.segment), r.tile))
        if field is not None:
            v = v / field
        if tissue_only:
            v[background.mask(t8)] = np.nan
        imgs[r.tile] = v
        to_montage[r.tile] = transforms.compose(stitch[(z, r.tile)], thumb_to_tile(t8.shape, (r.height, r.width)))
    rels = []
    for a, b in itertools.combinations(sorted(imgs), 2):
        va, vb = overlap_values(imgs[a], imgs[b], to_montage[a], to_montage[b])
        ok = np.isfinite(va) & np.isfinite(vb)
        va, vb = va[ok], vb[ok]
        if len(va) >= MIN_OVERLAP_PX and (rel := relation(va, vb)):
            rels.append((a, b, *rel, len(va), 1.0, float(np.median(vb))))
    return solve(connected(rels)) if rels else {}


def connected(rels):
    """The relations among the largest set of tiles linked by overlaps.

    Separate sets have no measured relation to each other, so solving them together would
    set their relative brightness arbitrarily; the other tiles count as unmeasured at this z.
    """
    tiles = sorted({t for r in rels for t in r[:2]})
    k = {t: i for i, t in enumerate(tiles)}
    edges = coo_matrix((np.ones(len(rels)), ([k[r[0]] for r in rels], [k[r[1]] for r in rels])),
                       shape=(len(tiles), len(tiles)))
    _, label = connected_components(edges, directed=False)
    big = np.bincount(label).argmax()
    return [r for r in rels if label[k[r[0]]] == big]


def overlap_values(img_a, img_b, A, B):
    """Values of two thumbnails at the montage points both cover, on b's pixel grid.

    A, B: thumbnail pixel -> montage. Nearest-neighbour sampling keeps both value
    distributions unblurred, so their quantiles stay comparable.
    """
    b_to_a = transforms.compose(transforms.invert(A), B)
    (ha, wa), (hb, wb) = img_a.shape, img_b.shape
    x0, y0, x1, y1 = transforms.bbox(transforms.invert(b_to_a), wa, ha)   # a's footprint in b's grid
    xs = np.arange(max(0, int(np.floor(x0)) - 1), min(wb, int(np.ceil(x1)) + 1))
    ys = np.arange(max(0, int(np.floor(y0)) - 1), min(hb, int(np.ceil(y1)) + 1))
    if not len(xs) or not len(ys):
        return np.zeros(0), np.zeros(0)
    xx, yy = (v.ravel() for v in np.meshgrid(xs, ys))
    pa = np.rint(transforms.apply(b_to_a, np.c_[xx, yy])).astype(int)
    ok = (pa[:, 0] >= 0) & (pa[:, 0] < wa) & (pa[:, 1] >= 0) & (pa[:, 1] < ha)
    return img_a[pa[ok, 1], pa[ok, 0]], img_b[yy[ok], xx[ok]]


def relation(va, vb):
    """(gain, offset) with va ~ gain * vb + offset, matching medians and inter-quartile ranges."""
    qa, qb = np.percentile(va, [25, 50, 75]), np.percentile(vb, [25, 50, 75])
    iqr_a, iqr_b = qa[2] - qa[0], qb[2] - qb[0]
    if iqr_a <= 0 or iqr_b <= 0:
        return None
    gain = iqr_a / iqr_b
    return gain, qa[1] - gain * qb[1]


GAIN_PRIOR = 0.05   # weight of each tile's pull towards gain 1, relative to the strongest relation


def solve(rels):
    """{tile: (g, o)} with g_a v_a + o_a = g_b v_b + o_b in every overlap, mean g = 1, mean o = 0.

    rels: (a, b, g_ab, o_ab, n_px[, gain_weight[, centre]]) with v_a ~ g_ab v_b + o_ab. Gains
    first, then offsets given the gains, matching the tiles at v_b = centre (the overlap's typical
    value; default 0): a relation is accurate near its data, and its value at 0 (the offset) is
    off by the gain's error times the data's distance from 0. Equations weighted by sqrt(n_px)
    (the gain equation also by gain_weight, default 1), the mean constraints by a much larger
    weight. Each gain is also pulled weakly towards 1 (GAIN_PRIOR of the strongest relation's
    weight), so gains no relation pins well (low gain_weight) stay near 1 instead of following noise.

    The gains are solved as logs, log g_a + log g_ab - log g_b = 0: in g_a g_ab - g_b = 0 the
    residuals scale with the gains, so where tiles are linked only by relations of ~0 gain weight
    (e.g. two tile columns whose overlaps are all resin) the group with the noisier gains got a
    smaller scale, and the groups' gains drifted apart (by 30 % and more) instead of staying at 1.
    """
    tiles = sorted({t for r in rels for t in r[:2]})
    k = {t: i for i, t in enumerate(tiles)}
    w = np.sqrt([r[4] for r in rels])
    g_ab = np.array([r[2] for r in rels], float)
    usable = np.isfinite(g_ab) & (g_ab > 0)
    wg = np.where(usable, w * np.array([r[5] if len(r) > 5 else 1.0 for r in rels]), 0.0)
    big = 100 * w.max()
    G, D = np.zeros((len(rels) + 1, len(tiles))), np.zeros((len(rels) + 1, len(tiles)))
    prior = GAIN_PRIOR * w.max() * np.eye(len(tiles))   # log g_t = 0, weakly
    for j, (a, b, *_) in enumerate(rels):
        G[j, k[a]], G[j, k[b]] = wg[j], -wg[j]   # log g_a - log g_b = -log g_ab
        D[j, k[a]], D[j, k[b]] = w[j], -w[j]   # o_a - o_b = -g_a o_ab
    G[-1] = D[-1] = big
    rhs_g = -wg * np.log(np.where(usable, g_ab, 1.0))
    log_g = np.linalg.lstsq(np.vstack([G, prior]), np.r_[rhs_g, 0.0, np.zeros(len(tiles))], rcond=None)[0]
    g = np.exp(log_g)
    g /= g.mean()
    def at_centre(j, r):   # o_a - o_b = g_b m - g_a (g_ab m + o_ab), m = the relation's centre
        m = r[6] if len(r) > 6 else 0.0
        return w[j] * (g[k[r[1]]] * m - g[k[r[0]]] * (r[2] * m + r[3]))
    rhs = [at_centre(j, r) for j, r in enumerate(rels)]
    o = np.linalg.lstsq(D, np.r_[rhs, 0.0], rcond=None)[0]
    return {t: (g[i], o[i]) for t, i in k.items()}


def plot(df, path):
    blocks = df.groupby("block")["z"].min().iloc[1:]
    balanced = "g" in df
    fig, axes = plt.subplots(3 if balanced else 2, 1, sharex=True, figsize=(12, 9 if balanced else 6),
                             layout="constrained")
    for ax, col in zip(axes, ["hi", "lo"]):
        plot_tiles(ax, df, "z", f"{col}_raw", blocks, lw=0.6, alpha=0.35)
        plot_tiles(ax, df, "z", col)
        ax.set_ylabel(f"{col} (uint16)")
    handles, labels = axes[0].get_legend_handles_labels()
    n = df["tile"].nunique()
    axes[0].legend(handles[n:], labels[n:], title="tile", fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    axes[0].set_title("Display levels per tile (faint: raw percentiles; grey lines: smoothing restarts)")
    if balanced:
        plot_tiles(axes[2], df, "z", "g", blocks)
        axes[2].set_ylabel("tile gain")
    axes[-1].set_xlabel("z")
    save(fig, path)


def main(argv=None):
    p = base_parser(__doc__.splitlines()[0])
    args = p.parse_args(argv)
    cfg = setup(args, deep_merge(deep_merge(DEFAULTS, shading.DEFAULTS), patchgrid.DEFAULTS))
    out = levels_path(cfg)
    # The previous levels, for the reference window's QC: this output's own, else the base levels
    prev = out if out.exists() else levels_path(cfg, output="")
    old = pd.read_csv(prev, dtype={"tile": str}) if prev.exists() else None
    df = compute_levels(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(out, lambda t: df[["z", "tile", "lo", "hi"]].to_csv(t, index=False, float_format="%.2f"))
    plot(df, qc_path(cfg, f"{_qc_name(cfg, 'intensity')}.png"))
    if cfg["intensity"]["window"] == "reference":
        qc, samp, fl = reference_qc(cfg, df, old)
        qc["previous_levels"] = str(prev) if old is not None else None
        name = _qc_name(cfg, "intensity_reference")
        plot_reference(cfg, samp, fl, qc, qc_path(cfg, f"{name}.png"))
        atomic_write(qc_path(cfg, f"{name}.json"), lambda t: Path(t).write_text(json.dumps(qc, indent=1)))
        atomic_write(qc_path(cfg, f"{name}_samples.csv"), lambda t: samp.to_csv(t, index=False, float_format="%.4f"))
        for flag in qc["flags"]:
            log.warning("reference window: %s", flag)
        log.info("reference window: predicted neuropil grey p5/p50/p95 %s, flicker max %.3f greys",
                 qc["predicted_new"]["median_grey_p5_p50_p95"], qc["flicker_new"]["max_d_median"] or 0)
    log.info("levels for %d (z, tile) over %d slices", len(df), df["z"].nunique())
    return 0


if __name__ == "__main__":
    sys.exit(main())
