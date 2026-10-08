"""Per-slice statistics and thumbnails for QC and for intensity normalisation.

``run`` (array over z-chunks): every (z, tile) slice is area-averaged down by ``preview.factor``;
stats are computed on the downsampled uint16 values and a uint8 thumbnail is autoscaled to its
own p0.5-p99.5. ``merge``: preview/stats.csv, thumbs/index.csv, one contact sheet per calendar
day (sheets/YYYY-MM-DD.png) and stats.png.
"""

import logging
import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import features  # noqa: E402
from .cli import atomic_write, base_parser, my_chunks, setup, task_info  # noqa: E402
from .config import step_dir  # noqa: E402
from .slices import StackCache, load_slices  # noqa: E402

log = logging.getLogger(__name__)

DEFAULTS = {
    "preview": {
        "factor": 16,         # downsampling factor (area average) for stats and thumbnails
        "chunk_slices": 50,   # z per array-task chunk (fixed blocks of global z, split at segment changes)
        "sheet_slices": 6,    # evenly spaced slices shown on each daily contact sheet
    },
}

# stats.csv percentile columns -> percentile
PERCENTILES = {"p0_5": 0.5, "p1": 1.0, "p50": 50.0, "p99": 99.0, "p99_5": 99.5}
STATS = ["mean", "std", *PERCENTILES, "min", "max", "frac_zero", "frac_saturated"]
SATURATED = np.iinfo(np.uint16).max
MAX_POINTS = 5000   # per plotted series
# Categorical tile colours in fixed order; past 8 tiles the line style changes instead.
TILE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def downsample(img, factor):
    """Area-average downsample, rounded back to uint16."""
    return np.clip(np.rint(features.downsample(img, factor)), 0, SATURATED).astype(np.uint16)


def slice_stats(small):
    """Stats of one downsampled uint16 slice, keyed like STATS."""
    v = small.ravel()
    pct = np.percentile(v, list(PERCENTILES.values()))
    return {"mean": float(v.mean()), "std": float(v.std()), **dict(zip(PERCENTILES, pct.tolist())),
            "min": int(v.min()), "max": int(v.max()),
            "frac_zero": float(np.mean(v == 0)), "frac_saturated": float(np.mean(v == SATURATED))}


def thumb_values(thumb, lo, hi):
    """Approximate uint16 values of a thumbnail autoscaled to [lo, hi] (bin centres)."""
    return lo + (np.asarray(thumb, np.float64) + 0.5) * ((hi - lo) / 255.0)


def z_chunks(slices, size):
    """Lists of selected z, one per (segment, z // size).

    Chunks never span a segment, so a tile's thumbnails in one chunk share a shape. Their bounds
    are fixed in global z, so a new selection start or a newly excluded slice changes only the
    chunks holding those z instead of shifting (and recomputing) every later chunk.
    """
    per_z = slices.drop_duplicates("z")
    return [g["z"].tolist() for _, g in per_z.groupby([per_z["segment"], per_z["z"] // int(size)])]


def chunk_name(zs):
    return f"{zs[0]:06d}-{zs[-1] + 1:06d}"


def read_chunk(path):
    return pd.read_csv(path, dtype={"tile": str})


def has_rows(df, want):
    """True if ``df`` has a row for every (z, tile) of ``want``."""
    have = pd.MultiIndex.from_frame(df[["z", "tile"]])
    return bool(pd.MultiIndex.from_frame(want[["z", "tile"]]).isin(have).all())


def thumb_name(name, tile):
    return f"z{name}_tile{tile}.npy"


def run(cfg, args):
    p = cfg["preview"]
    slices = load_slices(cfg)
    cache = StackCache(cfg["raw_dir"])
    mine = my_chunks(z_chunks(slices, p["chunk_slices"]), *task_info(args))
    log.info("%d chunks for this task", len(mine))
    for zs in mine:
        name = chunk_name(zs)
        want = slices[slices["z"].isin(zs)]
        out = step_dir(cfg, "preview", "stats", f"chunk_{name}.csv")
        if out.exists() and not args.overwrite:
            if has_rows(read_chunk(out), want):
                log.info("chunk %s exists, skipping", name)
                continue
            log.info("chunk %s lacks some selected tile slices (e.g. a late file): recomputing", name)
        # The stats file marks a complete chunk, so it must not outlive the thumbnails it indexes.
        out.unlink(missing_ok=True)
        rows, thumbs = [], {}
        for rec in want.to_dict("records"):   # sorted by (z, tile)
            small = downsample(cache.read(rec), p["factor"])
            st = slice_stats(small)
            rows.append({"z": rec["z"], "timestamp": rec["timestamp"].isoformat(), "tile": rec["tile"], **st})
            thumbs.setdefault(rec["tile"], []).append(features.to_uint8(small, st["p0_5"], st["p99_5"]))
        for tile, stack in thumbs.items():
            data = np.stack(stack)
            atomic_write(step_dir(cfg, "preview", "thumbs", thumb_name(name, tile)), lambda t: np.save(t, data))
        # Stats last: their presence marks the chunk complete.
        atomic_write(out, lambda t: pd.DataFrame(rows).to_csv(t, index=False))
        log.info("chunk %s: %d tile slices", name, len(rows))


def merge(cfg, args):
    p = cfg["preview"]
    slices = load_slices(cfg)
    stats_dir = step_dir(cfg, "preview", "stats")
    groups = z_chunks(slices, p["chunk_slices"])
    if not groups:
        raise RuntimeError("no selected slices in check/slices.csv")
    parts, bad = [], []
    for zs in groups:
        n = chunk_name(zs)
        path = stats_dir / f"chunk_{n}.csv"
        df = read_chunk(path) if path.exists() else None
        if df is None or not has_rows(df, slices[slices["z"].isin(zs)]):
            bad.append(n)
            continue
        df["npy"] = [thumb_name(n, t) for t in df["tile"]]
        df["i"] = df.groupby("tile").cumcount()   # rows were written in z order per tile
        parts.append(df)
    if bad:
        raise RuntimeError(f"{len(bad)} of {len(groups)} preview chunks missing or out of date (first: {bad[0]}); "
                           "run 'preview run' first")
    stats = pd.concat(parts, ignore_index=True)
    # Rows excluded since their chunk was computed: drop them (thumbnail positions 'i' stay valid).
    keep = pd.MultiIndex.from_frame(stats[["z", "tile"]]).isin(pd.MultiIndex.from_frame(slices[["z", "tile"]]))
    if not keep.all():
        log.info("ignoring %d tile slices in preview chunks that are no longer selected", int((~keep).sum()))
    stats = stats[keep].sort_values(["z", "tile"]).reset_index(drop=True)
    atomic_write(step_dir(cfg, "preview", "stats.csv"),
                 lambda t: stats[["z", "timestamp", "tile", *STATS]].to_csv(t, index=False))
    atomic_write(step_dir(cfg, "preview", "thumbs", "index.csv"),
                 lambda t: stats[["z", "tile", "npy", "i"]].to_csv(t, index=False))
    log.info("stats for %d slices x tiles", len(stats))

    stats["timestamp"] = pd.to_datetime(stats["timestamp"])
    contact_sheets(cfg, stats.merge(slices[["z", "tile", "tile_row", "tile_col"]], on=["z", "tile"]),
                   p["sheet_slices"])
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 8), layout="constrained")
    seams = slices.loc[slices["seam"], "timestamp"].unique()
    for ax, col in zip(axes, ["p99", "p50", "p1"]):
        plot_tiles(ax, stats, "timestamp", col, seams)
        ax.set_ylabel(f"{col} (uint16)")
    axes[0].legend(title="tile", fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    axes[0].set_title("Preview intensity per tile (grey lines: seams)")
    save(fig, step_dir(cfg, "preview", "stats.png"))


def contact_sheets(cfg, df, n):
    """One PNG per calendar day of ``n`` evenly spaced slices, tiles placed by (tile_row, tile_col)."""
    thumbs_dir = step_dir(cfg, "preview", "thumbs")
    for day, g in df.groupby(df["timestamp"].dt.date):
        zs = sorted(g["z"].unique())
        pick = [zs[i] for i in np.unique(np.linspace(0, len(zs) - 1, n).round().astype(int))]
        panels = [mosaic(g[g["z"] == z], thumbs_dir) for z in pick]
        ncols = min(3, len(pick))
        nrows = math.ceil(len(pick) / ncols)
        h, w = panels[0][0].shape
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, nrows * (5 * h / w + 0.4)), squeeze=False,
                                 layout="constrained")
        fig.get_layout_engine().set(wspace=0.06, hspace=0.04)   # slices further apart than tiles
        for ax in axes.flat:
            ax.axis("off")
        for ax, z, (img, labels) in zip(axes.flat, pick, panels):
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            for (x, y), tile in labels:
                ax.text(x, y, tile, color="white", fontsize=7, va="top",
                        bbox={"facecolor": "black", "alpha": 0.5, "pad": 1, "lw": 0})
            ts = g.loc[g["z"] == z, "timestamp"].iloc[0]
            ax.set_title(f"z {z}   {ts:%Y-%m-%d %H:%M:%S}", fontsize=10)
        fig.suptitle(f"{cfg.get('name') or ''} {day}".strip())
        save(fig, step_dir(cfg, "preview", "sheets", f"{day}.png"))
    log.info("contact sheets for %d days", df["timestamp"].dt.date.nunique())


def mosaic(rows, thumbs_dir):
    """Thumbnails of one z on a white canvas by (tile_row, tile_col) with small gaps (no stitching).

    Returns (image, [((x, y), tile), ...]) with each tile's top-left corner.
    """
    imgs = [np.load(thumbs_dir / r.npy, mmap_mode="r")[r.i] for r in rows.itertuples()]
    h, w = max(i.shape[0] for i in imgs), max(i.shape[1] for i in imgs)
    gap = max(2, max(h, w) // 100)
    r0, c0 = rows["tile_row"].min(), rows["tile_col"].min()
    nr, nc = rows["tile_row"].max() - r0 + 1, rows["tile_col"].max() - c0 + 1
    canvas = np.full((nr * (h + gap) - gap, nc * (w + gap) - gap), 255, np.uint8)
    labels = []
    for r, img in zip(rows.itertuples(), imgs):
        y, x = (r.tile_row - r0) * (h + gap), (r.tile_col - c0) * (w + gap)
        canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
        labels.append(((x, y), r.tile))
    return canvas, labels


def plot_tiles(ax, df, x, y, seams=(), **kw):
    """One line per tile (subsampled to MAX_POINTS), seams as grey vertical lines."""
    for s in seams:
        ax.axvline(s, color="0.8", lw=0.6, zorder=0)
    for i, (tile, g) in enumerate(df.groupby("tile", sort=True)):
        g = g.iloc[::max(1, math.ceil(len(g) / MAX_POINTS))]
        style = {"lw": 1, "color": TILE_COLORS[i % 8], "ls": ["-", "--", ":"][i // 8 % 3], **kw}
        ax.plot(g[x], g[y], label=tile, **style)
    ax.grid(alpha=0.3)


def save(fig, path):
    atomic_write(path, lambda t: fig.savefig(t, dpi=100))
    plt.close(fig)


def main(argv=None):
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("command", choices=["run", "merge"])
    args = p.parse_args(argv)
    cfg = setup(args, DEFAULTS)
    {"run": run, "merge": merge}[args.command](cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
