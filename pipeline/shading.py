"""Flat-field (shading) correction of the tiles: a smooth multiplicative gain field per tile position
and acquisition segment, estimated from the data and divided out before display levels are set.

An SEM tile is not evenly bright: the detector response falls off towards one side of the scan
(35h: ~8 % across the right-hand tiles, the same in every slice of a tile layout), which no
per-tile gain/offset can flatten and which shows as dark tile edges and steps at the seams. The
field is estimated retrospectively, as in BaSiC/CIDRE and the flat-field stages of TEM and
multi-SEM montage pipelines: every ``every``-th slice's preview thumbnail is divided by its own
tissue median, the ratio is averaged per pixel over the segment's slices (tissue at full weight,
so the anatomy averages out; plain bright resin, scaled to its own median, at ``resin_weight``,
so the empty areas between tiles get flattened too), and a low-order 2D polynomial is fitted to
the log of that map so only the slow fall-off is kept. Where a tile position has little or no
tissue in any slice (a corner that always holds the lumen), the fit is pulled towards no
correction (``prior_weight``) rather than left to extrapolate. Each field is scaled to a
tissue-weighted mean of 1, so a tile's median is unchanged and the preview percentiles stay
valid as windows.

A step of its own after preview (``python -m pipeline.shading --config C``; run_pipeline.sh runs it
when ``shading.enabled``). Writes work/shading/fields.npz (one field per segment and tile, at
thumbnail resolution, with fields.json) and qc/shading.png (measured map and fitted field per
tile). overlaps and intensity divide the tile data by the fields before comparing tiles; render
divides each tile slice by its field (resampled to tile pixels) before applying the display window.
"""

import hashlib
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import background
from .cli import atomic_write, base_parser, setup
from .config import qc_path, step_dir, step_path
from .preview import thumb_to_tile, thumb_values
from .slices import load_slices

log = logging.getLogger(__name__)

DEFAULTS = {
    "shading": {
        "enabled": False,      # flat-field each tile by a smooth gain field per (segment, tile)
        "every": 10,           # measure every N-th selected slice
        "poly_degree": 2,      # degree of the 2D polynomial fitted to log(brightness / tissue median)
        "min_slices": 10,      # (segment, tile) with fewer measured slices keep gain 1
        "min_count": 5,        # pixels measured in fewer slices than this don't enter the fit
        "resin_weight": 0.3,   # weight of plain bright resin pixels (scaled to the resin's own median)
                               # relative to tissue pixels; 0: tissue only
        "prior_weight": 0.2,   # every pixel also pulls the fit towards gain 1 with this weight (in units of
                               # one measured slice): the field decays to 1 where few slices have tissue
        "max_gain": 1.5,       # fields are clipped to [1 / max_gain, max_gain]
    },
}

MIN_TISSUE_PX = 2000   # thumbnail tissue pixels a slice needs to count (resin: the same, to be used)
RESIN_MIN_GREY = 128   # background darker than this (of 255, on the autoscaled thumbnail) is not resin


def _terms(degree):
    return [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]


def _design(shape, degree):
    """Polynomial design matrix over a (h, w) grid in coordinates [-1, 1]; (h*w, n_terms)."""
    h, w = shape
    y = np.linspace(-1, 1, h)[:, None] * np.ones((1, w))
    x = np.linspace(-1, 1, w)[None, :] * np.ones((h, 1))
    return np.stack([x.ravel() ** i * y.ravel() ** j for i, j in _terms(degree)], axis=1)


def fit_field(mean_map, count, degree=2, min_count=5, max_gain=1.5, prior_weight=0.2):
    """Smooth gain field (float32, mean 1 over the counted pixels) from the per-pixel mean relative
    brightness and the number of slices behind each pixel.

    Weighted least squares on log(brightness): a measured pixel weighs its slice count; every
    pixel of the grid also contributes a 'gain 1' observation of weight ``prior_weight``, so the
    polynomial follows the data where there is plenty and relaxes to no correction where there
    is none (instead of extrapolating into an unmeasured corner).
    """
    ok = count >= min_count
    if ok.sum() < 10 * len(_terms(degree)):
        return None
    A = _design(mean_map.shape, degree)
    w = np.sqrt(np.concatenate([count[ok].astype(float), np.full(A.shape[0], prior_weight)]))
    y = np.concatenate([np.log(np.clip(mean_map[ok], 1 / max_gain, max_gain)), np.zeros(A.shape[0])])
    rows = np.vstack([A[ok.ravel()], A])
    coef = np.linalg.lstsq(rows * w[:, None], y * w, rcond=None)[0]
    field = np.exp(A @ coef).reshape(mean_map.shape)
    field /= np.average(field[ok], weights=count[ok])
    return np.clip(field, 1 / max_gain, max_gain).astype(np.float32)


def measure(cfg):
    """Per (segment, tile): (mean relative brightness map, count map, n slices) from the thumbnails."""
    c = cfg["shading"]
    sl = load_slices(cfg)
    thumbs_dir = step_path(cfg, "preview", "thumbs")
    idx = pd.read_csv(thumbs_dir / "index.csv", dtype={"tile": str})
    stats = pd.read_csv(step_path(cfg, "preview", "stats.csv"), dtype={"tile": str})[["z", "tile", "p0_5", "p99_5"]]
    df = sl[["z", "tile", "segment"]].merge(idx, on=["z", "tile"]).merge(stats, on=["z", "tile"])
    zs = sorted(df["z"].unique())[::max(1, int(c["every"]))]
    df = df[df["z"].isin(zs)]
    acc = {}
    rw = float(c["resin_weight"])
    for r in df.itertuples():
        t8 = np.asarray(np.load(thumbs_dir / r.npy, mmap_mode="r")[int(r.i)])
        bg = background.mask(t8)
        tissue = ~bg
        if tissue.sum() < MIN_TISSUE_PX:
            continue
        v = thumb_values(t8, r.p0_5, r.p99_5)
        s, n, k = acc.setdefault((int(r.segment), r.tile), [np.zeros(t8.shape), np.zeros(t8.shape), 0])
        s[tissue] += v[tissue] / np.median(v[tissue])
        n[tissue] += 1
        resin = bg & (t8 > RESIN_MIN_GREY)
        if rw > 0 and resin.sum() >= MIN_TISSUE_PX:
            s[resin] += rw * v[resin] / np.median(v[resin])
            n[resin] += rw
        acc[(int(r.segment), r.tile)][2] = k + 1
    return {key: (np.divide(s, n, out=np.ones_like(s), where=n > 0), n, k) for key, (s, n, k) in acc.items()}


def solve(cfg):
    """Fit and write the fields; returns {(segment, tile): field}."""
    c = cfg["shading"]
    measured = measure(cfg)
    fields, info = {}, []
    for (seg, tile), (mean_map, count, k) in sorted(measured.items()):
        field = fit_field(mean_map, count, int(c["poly_degree"]), int(c["min_count"]), float(c["max_gain"]),
                          float(c["prior_weight"])) if k >= int(c["min_slices"]) else None
        if field is None:
            log.warning("shading: segment %d tile %s measured in %d slices (< %d): no field", seg, tile, k, c["min_slices"])
            continue
        fields[(seg, tile)] = field
        info.append({"segment": seg, "tile": tile, "slices": k, "shape": list(field.shape),
                     "min": float(field.min()), "max": float(field.max())})
        log.info("shading: segment %d tile %s: gain %.3f-%.3f from %d slices", seg, tile, field.min(), field.max(), k)
    out = step_dir(cfg, "shading")
    atomic_write(out / "fields.npz", lambda p: np.savez(p, **{f"s{s}_t{t}": f for (s, t), f in fields.items()}))
    atomic_write(out / "fields.json", lambda p: Path(p).write_text(json.dumps({"settings": c, "fields": info}, indent=1)))
    plot(measured, fields, qc_path(cfg, "shading.png"))
    return fields


def load(cfg):
    """{(segment, tile): field} written by ``solve``, or {} when shading is off or not yet solved."""
    if not (cfg.get("shading") or {}).get("enabled"):
        return {}
    path = step_path(cfg, "shading", "fields.npz")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: shading.enabled needs the shading step run first")
    with np.load(path) as z:
        out = {}
        for key in z.files:
            s, t = key[1:].split("_t")
            out[(int(s), t)] = z[key]
    return out


def digest(cfg):
    """Identifies the fields in use (for render's plan digest); None without shading."""
    if not (cfg.get("shading") or {}).get("enabled"):
        return None
    path = step_path(cfg, "shading", "fields.npz")
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"


def gain_window(field, tile_shape, rows, cols):
    """The field resampled to the tile's pixels ``rows`` x ``cols`` (slices), as float32."""
    T = thumb_to_tile(field.shape, tile_shape)         # thumbnail px -> tile px
    sx, sy = T[0, 0], T[1, 1]
    r0, c0 = rows.start or 0, cols.start or 0
    # Destination px (j, i) -> thumbnail px: (c0 + j - T[0, 2]) / sx, (r0 + i - T[1, 2]) / sy
    M = np.array([[1 / sx, 0, (c0 - T[0, 2]) / sx], [0, 1 / sy, (r0 - T[1, 2]) / sy]], np.float64)
    h, w = (rows.stop or tile_shape[0]) - r0, (cols.stop or tile_shape[1]) - c0
    return cv2.warpAffine(field, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def plot(measured, fields, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    segs = sorted({s for s, _ in measured})
    rows = []
    for seg in segs:
        tiles = sorted(t for s, t in measured if s == seg)
        rows += [(seg, t) for t in tiles]
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 2, figsize=(7, 2.6 * len(rows) + 0.5), squeeze=False)
    im = None
    for ax_row, (seg, tile) in zip(axes, rows):
        mean_map, count, k = measured[(seg, tile)]
        shown = np.where(count >= 1, mean_map, np.nan)
        im = ax_row[0].imshow(shown, cmap="RdBu_r", vmin=0.9, vmax=1.1)
        ax_row[0].set_title(f"segment {seg} tile {tile}: measured ({k} slices, tissue + weighted resin)", fontsize=8)
        f = fields.get((seg, tile))
        if f is not None:
            ax_row[1].imshow(f, cmap="RdBu_r", vmin=0.9, vmax=1.1)
            ax_row[1].set_title(f"fitted gain {f.min():.3f}-{f.max():.3f}", fontsize=8)
        else:
            ax_row[1].set_title("no field", fontsize=8)
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.4, label="brightness / tile tissue median")
    fig.suptitle("Tile shading: mean relative brightness by position (tissue), and the fitted flat-field")
    atomic_write(path, lambda p: fig.savefig(p, dpi=80))
    plt.close(fig)


def main(argv=None):
    p = base_parser(__doc__.splitlines()[0])
    args = p.parse_args(argv)
    cfg = setup(args, DEFAULTS)
    if not cfg["shading"]["enabled"]:
        log.error("shading.enabled is false: nothing to do")
        return 1
    fields = solve(cfg)
    log.info("%d fields written to %s", len(fields), step_path(cfg, "shading", "fields.npz"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
