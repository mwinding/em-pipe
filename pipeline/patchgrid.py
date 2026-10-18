"""Patch grid: smooth brightness maps within each tile, from the overlaps' patch-pair relations, for
what the flat-field leaves (after saalfeldlab/render's intensity-adjust, Janelia's tile intensity
correction).

The flat-field (``shading``) is one low-order polynomial per (segment, tile) from the thumbnails. It
misses brightness changes near tile edges and corners, which show as seam steps that change along a
seam and as loops of seams that no single level per tile can close (35h, 55 test slices: tissue seam
steps of 6 greys median, 61 seams over 10 greys). The overlaps step measures, with
``overlaps.patch_grid`` G > 0, one brightness relation per (patch of tile a, patch of tile b) of
every overlap, each tile cut into G x G patches (work/overlaps/patches.csv). This step fits maps to
them.

Model: patch p of tile t maps flat-fielded raw values v to g[t,p] (v - R) + R + c[t,p], where c is
the map's change at a fixed reference intensity R inside the data range (``reference``; default
the median bin centre). Maps are smoothed, averaged and interpolated as g and c, never as gain and
offset: a map's offset is its value at v = 0, far below the data (~37000), and swings with every
gain error. Between patch centres the maps are bilinear (clamped at the tile border).

Solve per slice, gains first, then c given the gains, each a weighted linear least squares
(bin j: v_a ~ gain_j v_b + offset_j between patch pa of a and pb of b, measured on n_j pixel pairs):
  gains:  wg_j (gain_j g[a,pa] - g[b,pb]) = 0, wg_j = sqrt(n_j) min(1, cnr_j / 10) ** gain_power;
          no gain equation where cnr_j < gain_cnr_min (resin: its gain is noise, 0.4-1.9 on 35h,
          and bent the maps in the 3x3 bottom corners by up to 15 %). Mostly-resin bins can reach
          cnr 6-8: on 35h one where 2-0 and 2-1 meet at the bottom has gain 1.53 at cnr 6.2 in
          every slice, so the median over slices keeps it, and at gain_cnr_min 5 it still bent
          2-0's shape by 12 % (1.02 at 8, which 35h uses)
  c:      w_j (c[a,pa] - c[b,pb]) = w_j (g[b,pb] (m_j - R) - g[a,pa] (gain_j m_j + offset_j - R)),
          w_j = sqrt(n_j), m_j = the bin's centre (median v_b): each relation matched at its own data
  smoothness: g and c of 4-neighbour patches of a tile equal, weight ``smooth`` x the median w_j
          (not the median wg_j, tiny in slices that are mostly resin, where it let gains vary 1.37x
          within a tile); ``prior``: each tile's mean gain -> 1, weight prior x the median w_j, as
          strong as the same pull on each of its patches; mean g = 1 and mean c = 0 over all patches.

The within-tile part of the maps belongs to the tile position (scan and detector), not to the slice
(solved per slice, the maps fitted noise wherever a seam held only resin). So per (segment, tile)
the shape (sg, sc) = (g / mean g, (c - mean c) / mean g), the map with its tile level taken out,
is the median over the per-slice solves of every ``every``-th measured slice of the segment, then
scaled to mean sg 1 and mean sc 0 (the shape leaves a tile's mean brightness unchanged, so the
preview's windows stay valid). Tile positions without bins keep the identity shape. Applied after
the flat-field: u = sg(x, y) (v - R) + R + sc(x, y).

Each slice's tile levels are then solved by ``intensity`` from the same bins with the shapes
applied (``shape_corrected``), through its usual per-slice balance and windows; render applies the
flat-field and shape as one gain and one offset field (``tile_fields``, ``apply_fields``) before
each tile's window.

A step of its own after overlaps (``python -m pipeline.patchgrid --config C``; run_pipeline.sh runs
it when ``enabled``). Writes work/patchgrid/shapes.npz (segment, tile, sg, sc per (segment, tile);
R; grid), shapes.json (settings, R, grid, the patches.csv digest, per (segment, tile) the slices
used and the gain and c ranges) and qc/patchgrid.png (each segment's shapes, laid out as the tiles).
"""

import hashlib
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import overlaps, shading
from .cli import atomic_write, base_parser, setup
from .config import deep_merge, get, qc_path, step_dir, step_path
from .preview import thumb_to_tile
from .slices import load_slices

log = logging.getLogger(__name__)

DEFAULTS = {
    "patchgrid": {
        "enabled": False,      # run_pipeline.sh: solve tile shapes after overlaps (needs overlaps.patch_grid > 0);
                               # intensity balances and render applies them
        "every": 20,           # shapes from the solves of every N-th measured slice of each segment
        "smooth": 1.0,         # equality of neighbouring patches, relative to the median relation weight sqrt(n)
        "prior": 0.02,         # pull of each tile's mean gain towards 1, relative to the same
        "gain_power": 2.0,     # a bin's gain equation is weighted by min(1, cnr / 10) ** gain_power
        "gain_cnr_min": 5.0,   # bins with less contrast than this (resin) give no gain equation (8 for 2x2 tiles of ~13k px)
        "reference": None,     # R, the raw intensity at which c is defined; null: the median bin centre
    },
}

GAIN_CNR = 10.0   # as intensity.GAIN_CNR: bins whose spread is below this many fit-rms pin the gain less
SAMPLE_PX = 16    # tile px per sample of render's fields when the tile has no flat-field (whose grid they share)


# ----- bilinear maps between patch centres ------------------------------------------------

def interp_weights(x, size, grid):
    """(i0, t): bilinear interpolation between patch centres along one axis of ``size`` px, clamped
    at the border (value of patch i0 times 1 - t plus patch i0 + 1 times t)."""
    u = np.clip((np.asarray(x, float) + 0.5) / size * grid - 0.5, 0, grid - 1)
    if grid == 1:
        return np.zeros(u.shape, int), np.zeros(u.shape)
    i0 = np.minimum(np.floor(u).astype(int), grid - 2)
    return i0, u - i0


def at_points(vals, x, y, h, w):
    """A (grid, grid) patch array, bilinear between patch centres, at tile pixels (x, y) of an h x w tile."""
    G = vals.shape[0]
    if G == 1:
        return np.full(np.shape(x), float(vals[0, 0]))
    cx, tx = interp_weights(x, w, G)
    ry, ty = interp_weights(y, h, G)
    return ((1 - ty) * ((1 - tx) * vals[ry, cx] + tx * vals[ry, cx + 1])
            + ty * ((1 - tx) * vals[ry + 1, cx] + tx * vals[ry + 1, cx + 1]))


def on_grid(vals, ys, xs, h, w):
    """A (grid, grid) patch array, bilinear between patch centres, on tile pixel rows ys x columns xs."""
    G = vals.shape[0]
    if G == 1:
        return np.full((len(ys), len(xs)), float(vals[0, 0]))
    cx, tx = interp_weights(xs, w, G)
    across = (1 - tx)[None, :] * vals[:, cx] + tx[None, :] * vals[:, cx + 1]   # (G, len(xs))
    ry, ty = interp_weights(ys, h, G)
    return (1 - ty)[:, None] * across[ry] + ty[:, None] * across[ry + 1]


def identity(grid):
    return np.ones((grid, grid)), np.zeros((grid, grid))


# ----- solve ------------------------------------------------------------------------------

def bin_weights(bins, gain_power, gain_cnr_min):
    """(sqrt(n), gain weight) per bin: the gain weight is min(1, cnr / 10) ** gain_power, 0 below
    gain_cnr_min; a bin's gain equation is weighted by their product."""
    cnr = bins["cnr"].to_numpy(float)
    gw = np.minimum(1.0, cnr / GAIN_CNR) ** float(gain_power)
    gw[cnr < float(gain_cnr_min)] = 0.0
    return np.sqrt(bins["n"].to_numpy(float)), gw


def _connected_tiles(bins):
    tiles = sorted(set(bins["a"]) | set(bins["b"]))
    k = {t: i for i, t in enumerate(tiles)}
    e = coo_matrix((np.ones(len(bins)), (bins["a"].map(k), bins["b"].map(k))), shape=(len(tiles), len(tiles)))
    _, lab = connected_components(e, directed=False)
    big = np.bincount(lab).argmax()
    return [t for t in tiles if lab[k[t]] == big]


def _neighbours(grid):
    """(p, q) of every pair of 4-neighbour patches on a grid x grid grid."""
    idx = np.arange(grid * grid).reshape(grid, grid)
    return np.r_[np.c_[idx[:, :-1].ravel(), idx[:, 1:].ravel()], np.c_[idx[:-1].ravel(), idx[1:].ravel()]]


def _system(n_tiles, P, ia, ib, coef_a, coef_b, wrel, rhs, w_smooth, w_prior, target, nb):
    """Least squares over n_tiles x P unknowns: relation rows wrel (coef_a x[ia] - coef_b x[ib]) = wrel rhs,
    neighbour rows w_smooth (x[p] - x[q]) = 0, per tile w_prior sqrt(P) (mean x - target) = 0, and
    the mean of all x = target with a large weight."""
    N = n_tiles * P
    keep = wrel > 0
    ia, ib, coef_a, coef_b, wrel, rhs = ia[keep], ib[keep], coef_a[keep], coef_b[keep], wrel[keep], rhs[keep]
    nr = len(wrel)
    pairs = (np.arange(n_tiles)[:, None, None] * P + nb[None]).reshape(-1, 2)
    ns = len(pairs)
    A = np.zeros((nr + ns + n_tiles + 1, N))
    b = np.zeros(len(A))
    j = np.arange(nr)
    np.add.at(A, (j, ia), wrel * coef_a)
    np.add.at(A, (j, ib), -wrel * coef_b)
    b[:nr] = wrel * rhs
    A[nr + np.arange(ns), pairs[:, 0]] = w_smooth
    A[nr + np.arange(ns), pairs[:, 1]] = -w_smooth
    r0 = nr + ns
    for i in range(n_tiles):
        A[r0 + i, i * P:(i + 1) * P] = w_prior * math.sqrt(P) / P
        b[r0 + i] = w_prior * math.sqrt(P) * target
    big = 100.0 * max(float(wrel.max()) if nr else 0.0, w_smooth, 1.0)
    A[-1] = big / N
    b[-1] = big * target
    # rank-revealing QR: as exact as numpy's SVD solve here, ~4x faster on a 3 x 3 slice's 576 unknowns
    return scipy.linalg.lstsq(A, b, lapack_driver="gelsy", check_finite=False)[0]


def solve_slice(bins, grid, R, smooth=1.0, prior=0.02, gain_power=2.0, gain_cnr_min=5.0):
    """{tile: (g, c)} ((grid, grid) arrays) from one slice's bins (columns a, b, pa, pb, gain,
    offset, n, cnr, centre), for the tiles of the largest set linked by bins; see the module docstring."""
    if bins.empty:
        return {}
    tiles = _connected_tiles(bins)
    bins = bins[bins["a"].isin(tiles) & bins["b"].isin(tiles)]
    P = grid * grid
    k = {t: i for i, t in enumerate(tiles)}
    ia = bins["a"].map(k).to_numpy() * P + bins["pa"].to_numpy(int)
    ib = bins["b"].map(k).to_numpy() * P + bins["pb"].to_numpy(int)
    gain, off, m = (bins[c].to_numpy(float) for c in ("gain", "offset", "centre"))
    w, gw = bin_weights(bins, gain_power, gain_cnr_min)
    ref = float(np.median(w))
    nb = _neighbours(grid)
    ones = np.ones_like(gain)
    g = _system(len(tiles), P, ia, ib, gain, ones, w * gw, np.zeros_like(gain), smooth * ref, prior * ref, 1.0, nb)
    rhs = g[ib] * (m - R) - g[ia] * (gain * m + off - R)
    c = _system(len(tiles), P, ia, ib, ones, ones, w, rhs, smooth * ref, 0.0, 0.0, nb)
    return {t: (g[i * P:(i + 1) * P].reshape(grid, grid), c[i * P:(i + 1) * P].reshape(grid, grid))
            for t, i in k.items()}


def shape_of(g, c):
    """A map's within-tile shape: (g / mean g, (c - mean c) / mean g), so that the map is the shape
    followed by the tile level (mean g, mean c)."""
    mg = g.mean()
    return g / mg, (c - c.mean()) / mg


def shape_corrected(bins, shapes, R):
    """The bins as relations between shape-corrected values u = sg (v - R) + R + sc.

    bins: columns segment, a, b, gain, offset, centre, xa, ya, xb, yb (mean positions in the two
    tiles) and ha, wa, hb, wb (tile sizes); shapes {(segment, tile): (sg, sc)}, identity for
    (segment, tile) not in it. Each side's shape is taken where its pixel pairs are (bilinear, as
    render applies it). Returns a copy with gain, offset and centre (now in u_b) replaced.
    """
    out = bins.copy()
    sga, sca = _shape_values(bins, shapes, "a")
    sgb, scb = _shape_values(bins, shapes, "b")
    gain, off, m = (bins[c].to_numpy(float) for c in ("gain", "offset", "centre"))
    # u_a = R + sca + sga (gain v_b + off - R), with v_b = R + (u_b - R - scb) / sgb
    g2 = sga * gain / sgb
    out["gain"] = g2
    out["offset"] = R + sca + sga * (gain * R + off - R) - g2 * (R + scb)
    out["centre"] = sgb * (m - R) + R + scb
    return out


def _shape_values(bins, shapes, side):
    """(sg, sc) of side a or b's shape at each bin's mean position on that side."""
    sg, sc = np.ones(len(bins)), np.zeros(len(bins))
    for (seg, tile), idx in (bins.groupby(["segment", side]).indices.items() if len(bins) else ()):
        shape = shapes.get((int(seg), tile))
        if shape is None:
            continue
        sub = bins.iloc[idx]
        x, y = sub[f"x{side}"].to_numpy(float), sub[f"y{side}"].to_numpy(float)
        h, w = sub[f"h{side}"].to_numpy(float), sub[f"w{side}"].to_numpy(float)
        sg[idx], sc[idx] = at_points(shape[0], x, y, h, w), at_points(shape[1], x, y, h, w)
    return sg, sc


# ----- inputs and outputs ------------------------------------------------------------------

def read_bins(cfg, slices=None):
    """work/overlaps/patches.csv with each bin's segment and tile sizes (ha, wa, hb, wb) from the
    slice list; bins of (z, tile) not in it (not selected) are left out."""
    path = step_path(cfg, "overlaps", "patches.csv")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: patchgrid needs the overlaps step with overlaps.patch_grid > 0 "
                                "(run, merge)")
    grid = int(get(cfg, "overlaps.patch_grid", 0) or 0)
    info = path.with_suffix(".json")
    measured = json.loads(info.read_text()).get("patch_grid") if info.exists() else None
    if grid <= 0 or measured != grid:
        raise ValueError(f"{path} holds bins of patch grid {measured}, overlaps.patch_grid is {grid}: "
                         "set overlaps.patch_grid > 0 and re-run overlaps (run, merge)")
    bins = pd.read_csv(path, dtype={"a": str, "b": str})
    sl = load_slices(cfg) if slices is None else slices
    sizes = sl[["z", "tile", "segment", "height", "width"]]
    for side in ("a", "b"):
        s = sizes.rename(columns={"tile": side, "height": f"h{side}", "width": f"w{side}", "segment": f"seg_{side}"})
        bins = bins.merge(s, on=["z", side], how="inner")
    bins = bins.rename(columns={"seg_a": "segment"}).drop(columns="seg_b")
    return bins.sort_values(["z", "a", "b", "pa", "pb"]).reset_index(drop=True)


def _file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shape_slices(zs_by_segment, every):
    """Per segment, every ``every``-th of its measured slices (the first one included)."""
    return {seg: sorted(zs)[::max(1, int(every))] for seg, zs in zs_by_segment.items()}


def solve(cfg):
    """Solve and write the shapes; returns ({(segment, tile): (sg, sc)}, R)."""
    p = cfg["patchgrid"]
    grid = int(cfg["overlaps"]["patch_grid"])
    slices = load_slices(cfg)
    bins = read_bins(cfg, slices)
    if bins.empty:
        raise ValueError(f"{step_path(cfg, 'overlaps', 'patches.csv')} has no bins of selected slices")
    R = float(p["reference"]) if p["reference"] is not None else float(np.median(bins["centre"]))
    per_seg = bins.groupby("segment")["z"].unique()
    chosen = shape_slices({int(s): list(zs) for s, zs in per_seg.items()}, p["every"])
    kw = dict(smooth=float(p["smooth"]), prior=float(p["prior"]), gain_power=float(p["gain_power"]),
              gain_cnr_min=float(p["gain_cnr_min"]))
    by_z = dict(tuple(bins.groupby("z")))
    acc = {}
    for seg, zs in chosen.items():
        for z in zs:
            for t, (g, c) in solve_slice(by_z[z], grid, R, **kw).items():
                acc.setdefault((seg, t), []).append(shape_of(g, c))
    shapes, info = {}, []
    for key in sorted({(int(s), t) for s, t in zip(slices["segment"], slices["tile"])} | set(acc)):
        found = acc.get(key, [])
        if found:
            sg = np.median([s[0] for s in found], axis=0)
            sc = np.median([s[1] for s in found], axis=0)
            sg, sc = sg / sg.mean(), sc - sc.mean()   # the median of shapes need not have mean 1 and 0
        else:
            sg, sc = identity(grid)
        shapes[key] = (sg, sc)
        info.append({"segment": key[0], "tile": key[1], "slices": len(found), "gain_min": float(sg.min()),
                     "gain_max": float(sg.max()), "c_min": float(sc.min()), "c_max": float(sc.max())})
        if found:
            log.info("segment %d tile %s: shape from %d slices, gain %.3f-%.3f, c %+.0f to %+.0f", key[0], key[1],
                     len(found), sg.min(), sg.max(), sc.min(), sc.max())
        else:
            log.warning("segment %d tile %s: no bins in its shape slices, identity shape", *key)
    out = step_dir(cfg, "patchgrid")
    write_shapes(out / "shapes.npz", shapes, R, grid)
    meta ={"settings": p, "R": R, "R_from": "config" if p["reference"] is not None else "median bin centre",
            "grid": grid, "patches": _file_digest(step_path(cfg, "overlaps", "patches.csv")),
            "shape_slices": {str(s): len(zs) for s, zs in chosen.items()}, "shapes": info}
    atomic_write(out / "shapes.json", lambda f: Path(f).write_text(json.dumps(meta, indent=1)))
    plot(shapes, slices, R, qc_path(cfg, "patchgrid.png"))
    return shapes, R


def write_shapes(path, shapes, R, grid):
    """shapes.npz: segment, tile, sg, sc per (segment, tile) of ``shapes``; R; grid."""
    keys = sorted(shapes)
    atomic_write(path, lambda f: np.savez(
        f, segment=np.array([k[0] for k in keys], int), tile=np.array([k[1] for k in keys], str),
        sg=np.array([shapes[k][0] for k in keys], float), sc=np.array([shapes[k][1] for k in keys], float),
        R=np.float64(R), grid=np.int64(grid)))


def load(cfg):
    """({(segment, tile): (sg, sc)}, R) written by ``solve``, or ({}, None) when patchgrid is off."""
    if not (cfg.get("patchgrid") or {}).get("enabled"):
        return {}, None
    path = step_path(cfg, "patchgrid", "shapes.npz")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: patchgrid.enabled needs the patchgrid step run first")
    with np.load(path) as z:
        shapes = {(int(s), str(t)): (g, c) for s, t, g, c in zip(z["segment"], z["tile"], z["sg"], z["sc"])}
        return shapes, float(z["R"])


def digest(cfg):
    """Identifies the shapes in use (for render's plan digest); None without patchgrid."""
    if not (cfg.get("patchgrid") or {}).get("enabled"):
        return None
    path = step_path(cfg, "patchgrid", "shapes.npz")
    return _file_digest(path) if path.exists() else "missing"


def check_current(cfg):
    """Warn when the shapes were solved from another work/overlaps/patches.csv than the one there now."""
    meta = step_path(cfg, "patchgrid", "shapes.json")
    patches = step_path(cfg, "overlaps", "patches.csv")
    if meta.exists() and patches.exists() and json.loads(meta.read_text()).get("patches") != _file_digest(patches):
        log.warning("%s was solved from an older %s: re-run patchgrid for shapes from the current bins",
                    meta.with_suffix(".npz"), patches)


# ----- render -----------------------------------------------------------------------------

def tile_fields(shape, R, field, tile_shape):
    """(gain, offset) float32 fields with flat-field and shape in one: u = gain * raw + offset, i.e.
    sg (raw / field - R) + R + sc. Sampled on the flat-field's thumbnail grid (every SAMPLE_PX tile px
    without one); ``apply_fields`` resamples them to tile pixels as shading.gain_window does."""
    h, w = tile_shape
    size = field.shape if field is not None else (math.ceil(h / SAMPLE_PX), math.ceil(w / SAMPLE_PX))
    T = thumb_to_tile(size, tile_shape)
    xs, ys = T[0, 0] * np.arange(size[1]) + T[0, 2], T[1, 1] * np.arange(size[0]) + T[1, 2]
    sg, sc = on_grid(shape[0], ys, xs, h, w), on_grid(shape[1], ys, xs, h, w)
    gain = sg / field if field is not None else sg
    return gain.astype(np.float32), (R + sc - sg * R).astype(np.float32)


def apply_fields(img, fields, tile_shape, rows, cols, window=None, block=256):
    """gain * img + offset (float32) on tile pixels rows x cols (slices), in row blocks so the
    resampled fields stay small.

    window (lo, hi): the tile's display window too, (u - lo) / (hi - lo) clipped to [0, 1] as
    render's _normalise does it, folded into the fields at their own (thumbnail) resolution: one
    tile-sized float32 array instead of two and one pass fewer over it (a 35h 2 x 2 tile: 5-12 %
    faster, peak memory 0.69 GB instead of 1.35 GB, within 0.001 grey).
    """
    gain, offset = fields
    if window is not None:
        lo, hi = (float(v) for v in window)
        s = 1.0 / max(hi - lo, 1e-6)
        gain, offset = (gain * s).astype(np.float32), ((offset - lo) * s).astype(np.float32)
    out = np.empty(img.shape, np.float32)
    r0 = rows.start or 0
    for b0 in range(0, img.shape[0], block):
        b1 = min(img.shape[0], b0 + block)
        rs, o = slice(r0 + b0, r0 + b1), out[b0:b1]
        np.multiply(img[b0:b1], shading.gain_window(gain, tile_shape, rs, cols), out=o)
        o += shading.gain_window(offset, tile_shape, rs, cols)
        if window is not None:
            np.clip(o, 0, 1, out=o)
    return out


# ----- QC ---------------------------------------------------------------------------------

def plot(shapes, slices, R, path, samples=48):
    """Per segment: the shapes' gain sg and change sc at R (raw), tiles laid out as in the slice (row-col)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    segs = sorted({s for s, _ in shapes})
    if not segs:
        return
    fig, axes = plt.subplots(len(segs), 2, figsize=(8, 3.2 * len(segs) + 0.6), squeeze=False, layout="constrained")
    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_bad("0.8")   # the gaps between tiles
    size = slices.groupby(["segment", "tile"])[["height", "width"]].first()
    ims = [None, None]
    for ax_row, seg in zip(axes, segs):
        tiles = sorted(t for s, t in shapes if s == seg)
        rc = {t: tuple(int(v) for v in t.split("-")) for t in tiles}
        nr, nc = max(r for r, _ in rc.values()) + 1, max(c for _, c in rc.values()) + 1
        gap = 2
        mos = [np.full((nr * (samples + gap), nc * (samples + gap)), np.nan) for _ in range(2)]
        for t in tiles:
            sg, sc = shapes[(seg, t)]
            h, w = size.loc[(seg, t)] if (seg, t) in size.index else (1000, 1000)
            ys, xs = (np.linspace(0, n - 1, samples) for n in (h, w))
            r, c = rc[t]
            win = (slice(r * (samples + gap), r * (samples + gap) + samples),
                   slice(c * (samples + gap), c * (samples + gap) + samples))
            mos[0][win] = on_grid(sg, ys, xs, h, w)
            mos[1][win] = on_grid(sc, ys, xs, h, w)
        ims[0] = ax_row[0].imshow(mos[0], cmap=cmap, vmin=0.95, vmax=1.05)
        ims[1] = ax_row[1].imshow(mos[1], cmap=cmap, vmin=-300, vmax=300)
        ax_row[0].set_title(f"segment {seg}: shape gain sg", fontsize=9)
        ax_row[1].set_title(f"segment {seg}: shape change sc at R (raw)", fontsize=9)
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(ims[0], ax=axes[:, 0].tolist(), shrink=min(0.6, 2.0 / len(segs)), label="gain")
    fig.colorbar(ims[1], ax=axes[:, 1].tolist(), shrink=min(0.6, 2.0 / len(segs)), label="raw")
    fig.suptitle(f"Patch grid: within-tile shapes per segment (R = {R:.0f}), tiles as in the slice")
    atomic_write(path, lambda f: fig.savefig(f, dpi=80))
    plt.close(fig)


def main(argv=None):
    p = base_parser(__doc__.splitlines()[0])
    args = p.parse_args(argv)
    cfg = setup(args, deep_merge(DEFAULTS, overlaps.DEFAULTS))
    if not cfg["patchgrid"]["enabled"]:
        log.error("patchgrid.enabled is false: nothing to do")
        return 1
    shapes, R = solve(cfg)
    log.info("%d tile shapes (R = %.0f) written to %s", len(shapes), R, step_path(cfg, "patchgrid", "shapes.npz"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
