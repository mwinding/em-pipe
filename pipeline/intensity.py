"""Display levels per (z, tile) without brightness seams between tiles or flicker over z.

Each tile's lo/hi start from its preview percentiles, smoothed over z with a running median in
blocks that restart at segment changes (and at seams with ``break_at_seams``). With
``balance_tiles``, every ``balance_every`` slices the preview thumbnails of neighbouring tiles
are compared where they overlap (placed by the stitch transforms), a least-squares solve gives
per-tile gain/offset onto a common scale (mean gain 1, mean offset 0), and one common window per
z is mapped back into each tile's raw values. Writes intensity/levels.csv and intensity.png.
"""

import itertools
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

from . import transforms  # noqa: E402
from .cli import atomic_write, base_parser, setup  # noqa: E402
from .config import step_dir  # noqa: E402
from .preview import PERCENTILES, plot_tiles, save, thumb_values  # noqa: E402
from .slices import load_slices  # noqa: E402

log = logging.getLogger(__name__)

DEFAULTS = {
    "intensity": {
        "lo_percentile": 0.5,     # preview percentile rendered as 0 (one of 0.5, 1, 50, 99, 99.5)
        "hi_percentile": 99.5,    # preview percentile rendered as 255
        "smooth_slices": 51,      # running-median window over z, in selected slices
        "break_at_seams": True,   # restart smoothing at seams (it always restarts at segment changes)
        "balance_tiles": True,    # match brightness of overlapping tiles (needs stitch/tiles.csv)
        "balance_every": 10,      # measure tile overlaps every N selected slices
    },
}

MIN_OVERLAP_PX = 100   # thumbnail pixels a tile pair must share to be compared


def percentile_column(p):
    """stats.csv column for a percentile, e.g. 0.5 -> 'p0_5'."""
    col = "p" + f"{float(p):g}".replace(".", "_")
    if col not in PERCENTILES:
        raise ValueError(f"intensity percentile {p!r} is not in preview/stats.csv; "
                         f"use one of {list(PERCENTILES.values())}")
    return col


def running_median(s, window):
    return s.rolling(int(window), center=True, min_periods=1).median()


def load_inputs(cfg, lo_col, hi_col):
    """Selected (z, tile) rows with their preview percentiles, smoothing block and position in z."""
    slices = load_slices(cfg)
    stats = pd.read_csv(Path(cfg["output_dir"]) / "preview" / "stats.csv", dtype={"tile": str})
    cols = list(dict.fromkeys(["z", "tile", lo_col, hi_col, "p0_5", "p99_5"]))
    df = slices[["z", "tile", "height", "width", "segment", "seam"]].merge(stats[cols], on=["z", "tile"], how="left")
    missing = df[lo_col].isna() | df[hi_col].isna()
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
    df["lo_raw"], df["hi_raw"] = df[lo_col], df[hi_col]
    return df.sort_values(["z", "tile"]).reset_index(drop=True)


def compute_levels(cfg):
    """DataFrame of every selected (z, tile) with lo, hi (and gain g, offset o when balanced)."""
    c = cfg["intensity"]
    lo_col, hi_col = percentile_column(c["lo_percentile"]), percentile_column(c["hi_percentile"])
    if PERCENTILES[lo_col] >= PERCENTILES[hi_col]:
        raise ValueError("intensity.lo_percentile must be below hi_percentile")
    if int(c["smooth_slices"]) < 1 or int(c["balance_every"]) < 1:
        raise ValueError("intensity.smooth_slices and balance_every must be at least 1")
    df = load_inputs(cfg, lo_col, hi_col)
    by_run = df.groupby(["tile", "block"])
    for col in ("lo", "hi"):
        df[col] = by_run[f"{col}_raw"].transform(lambda s: running_median(s, c["smooth_slices"]))
    if c["balance_tiles"]:
        tiles_csv = Path(cfg["output_dir"]) / "stitch" / "tiles.csv"
        if tiles_csv.exists():
            df = balance(cfg, df, tiles_csv)
        else:
            log.warning("%s not found: per-tile levels without tile balancing", tiles_csv)
    return df


def balance(cfg, df, tiles_csv):
    """Replace lo/hi by one common window per z, mapped through per-tile gain/offset (columns g, o)."""
    c = cfg["intensity"]
    thumbs_dir = Path(cfg["output_dir"]) / "preview" / "thumbs"
    stitch = transforms.read_csv(tiles_csv, ["z", "tile"])
    df = df.merge(pd.read_csv(thumbs_dir / "index.csv", dtype={"tile": str}), on=["z", "tile"], how="left")
    first = df.groupby("block")["pos"].transform("min")
    sampled = df[(df["pos"] - first) % int(c["balance_every"]) == 0]
    found = []
    for z, rows in sampled.groupby("z"):
        found += [{"z": z, "tile": t, "g": g, "o": o} for t, (g, o) in measure(rows, stitch, thumbs_dir).items()]
    meas = pd.DataFrame(found, columns=["z", "tile", "g", "o"])
    log.info("tile balance measured at %d of %d sampled slices", meas["z"].nunique(), sampled["z"].nunique())
    df = df.merge(meas, on=["z", "tile"], how="left")

    window = max(1, round(c["smooth_slices"] / c["balance_every"]))
    for _, grp in df.groupby(["tile", "block"]):
        ok = grp["g"].notna()
        if ok.any():
            for col in ("g", "o"):
                smooth = running_median(grp.loc[ok, col], window)
                df.loc[grp.index, col] = np.interp(grp["pos"], grp.loc[ok, "pos"], smooth)
    bal = df["g"].notna()
    if not bal.all():
        log.warning("%d (z, tile) have no overlap measurement in their smoothing block: left unbalanced",
                    int((~bal).sum()))
    b = df[bal]
    for col in ("lo", "hi"):
        common = (b["g"] * b[col] + b["o"]).groupby(b["z"]).transform("median")
        df.loc[bal, col] = (common - b["o"]) / b["g"]
    return df


def measure(rows, stitch, thumbs_dir):
    """{tile: (gain, offset)} at one z from its thumbnails' overlaps; {} if no tile pair overlaps."""
    z = int(rows["z"].iloc[0])
    imgs, to_montage = {}, {}
    for r in rows.itertuples():
        if pd.isna(r.npy) or (z, r.tile) not in stitch:
            continue
        t8 = np.load(thumbs_dir / r.npy, mmap_mode="r")[int(r.i)]
        imgs[r.tile] = thumb_values(t8, r.p0_5, r.p99_5)
        to_montage[r.tile] = transforms.compose(stitch[(z, r.tile)], thumb_to_tile(t8.shape, (r.height, r.width)))
    rels = []
    for a, b in itertools.combinations(sorted(imgs), 2):
        va, vb = overlap_values(imgs[a], imgs[b], to_montage[a], to_montage[b])
        if len(va) >= MIN_OVERLAP_PX and (rel := relation(va, vb)):
            rels.append((a, b, *rel, len(va)))
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


def thumb_to_tile(thumb_shape, tile_shape):
    """Thumbnail pixel index -> tile pixel index (pixel centres)."""
    sy, sx = tile_shape[0] / thumb_shape[0], tile_shape[1] / thumb_shape[1]
    return np.array([[sx, 0.0, 0.5 * sx - 0.5], [0.0, sy, 0.5 * sy - 0.5]])


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


def solve(rels):
    """{tile: (g, o)} with g_a v_a + o_a = g_b v_b + o_b in every overlap, mean g = 1, mean o = 0.

    rels: (a, b, g_ab, o_ab, n_px) with v_a ~ g_ab v_b + o_ab. Gains first, then offsets given
    the gains; equations weighted by sqrt(n_px), the mean constraints by a much larger weight.
    """
    tiles = sorted({t for r in rels for t in r[:2]})
    k = {t: i for i, t in enumerate(tiles)}
    w = np.sqrt([r[4] for r in rels])
    big = 100 * w.max()
    G, D = np.zeros((len(rels) + 1, len(tiles))), np.zeros((len(rels) + 1, len(tiles)))
    for j, (a, b, g_ab, _, _) in enumerate(rels):
        G[j, k[a]] += w[j] * g_ab   # g_a g_ab - g_b = 0
        G[j, k[b]] -= w[j]
        D[j, k[a]], D[j, k[b]] = w[j], -w[j]   # o_a - o_b = -g_a o_ab
    G[-1] = D[-1] = big
    g = np.linalg.lstsq(G, np.r_[np.zeros(len(rels)), big * len(tiles)], rcond=None)[0]
    rhs = [-w[j] * g[k[a]] * o_ab for j, (a, _, _, o_ab, _) in enumerate(rels)]
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
    cfg = setup(args, DEFAULTS)
    df = compute_levels(cfg)
    atomic_write(step_dir(cfg, "intensity", "levels.csv"),
                 lambda t: df[["z", "tile", "lo", "hi"]].to_csv(t, index=False, float_format="%.2f"))
    plot(df, step_dir(cfg, "intensity", "intensity.png"))
    log.info("levels for %d (z, tile) over %d slices", len(df), df["z"].nunique())
    return 0


if __name__ == "__main__":
    sys.exit(main())
