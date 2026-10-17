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
preview percentiles lo_percentile / hi_percentile (all pixels) to 0 / 255.
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

from . import background, patchgrid, shading, transforms  # noqa: E402
from .cli import atomic_write, base_parser, setup  # noqa: E402
from .config import deep_merge, get, qc_path, step_dir, step_path  # noqa: E402
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


def compute_levels(cfg):
    """DataFrame of every selected (z, tile) with lo, hi (and gain g, offset o when balanced)."""
    c = cfg["intensity"]
    if c["window"] not in ("median", "percentile"):
        raise ValueError(f"intensity.window must be median or percentile, not {c['window']!r}")
    if int(c["smooth_slices"]) < 1 or int(c["balance_every"]) < 1:
        raise ValueError("intensity.smooth_slices and balance_every must be at least 1")
    if c["window"] == "median":
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
    # Raw P667 tiles jump by 1-2 % of the display range from slice to slice, independently of each
    # other; tissue changes slowly, so the tile median's deviation from its trend is that jump.
    # (A column, so it stays with its rows through balance's merge.)
    df["jump"] = df[mid] - by_run[mid].transform(lambda s: running_median(s, c["smooth_slices"]))
    if c["per_slice"] not in (True, False, "slice"):
        raise ValueError(f"intensity.per_slice must be true, false or slice, not {c['per_slice']!r}")
    if c["balance_source"] not in ("thumbnails", "overlaps"):
        raise ValueError(f"intensity.balance_source must be thumbnails or overlaps, not {c['balance_source']!r}")
    if c["balance_tiles"]:
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
    if c["per_slice"] == "slice":
        rel = df["jump"] / (df["hi"] - df["lo"])
        df["jump"] = rel.groupby(df["z"]).transform("median") * (df["hi"] - df["lo"])
    if c["per_slice"]:
        df["lo"] += df["jump"]
        df["hi"] += df["jump"]
    return df


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
    """
    tiles = sorted({t for r in rels for t in r[:2]})
    k = {t: i for i, t in enumerate(tiles)}
    w = np.sqrt([r[4] for r in rels])
    wg = w * np.array([r[5] if len(r) > 5 else 1.0 for r in rels])
    big = 100 * w.max()
    G, D = np.zeros((len(rels) + 1, len(tiles))), np.zeros((len(rels) + 1, len(tiles)))
    prior = GAIN_PRIOR * w.max() * np.eye(len(tiles))   # g_t - 1 = 0, weakly
    for j, (a, b, g_ab, *_) in enumerate(rels):
        G[j, k[a]] += wg[j] * g_ab   # g_a g_ab - g_b = 0
        G[j, k[b]] -= wg[j]
        D[j, k[a]], D[j, k[b]] = w[j], -w[j]   # o_a - o_b = -g_a o_ab
    G[-1] = D[-1] = big
    g = np.linalg.lstsq(np.vstack([G, prior]), np.r_[np.zeros(len(rels)), big * len(tiles), prior.diagonal()],
                        rcond=None)[0]
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
    df = compute_levels(cfg)
    atomic_write(step_dir(cfg, "intensity", "levels.csv"),
                 lambda t: df[["z", "tile", "lo", "hi"]].to_csv(t, index=False, float_format="%.2f"))
    plot(df, qc_path(cfg, "intensity.png"))
    log.info("levels for %d (z, tile) over %d slices", len(df), df["z"].nunique())
    return 0


if __name__ == "__main__":
    sys.exit(main())
