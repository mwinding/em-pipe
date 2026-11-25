"""Slice-to-slice brightness flicker in rendered volumes (what x-z and y-z views show as stripes), and in
the raw tiles and display windows behind it.

    python tools/zflicker.py --out DIR [--config C] [--scale 3] [--grid 6] [--box-px N] [--window 9] [NAME=VOLUME ...]

Rendered volumes: per plane, the mean grey of each box of a grid x grid split of the plane at the given
scale (pixels > 0 only; boxes less than half covered are left out). Tissue changes slowly over z and
brightness jumps don't, so a box's flicker is its mean minus the running mean of its means over
--window planes. That is split into the part shared by the whole plane (the median over boxes, "common")
and the rest ("own": box minus common), each summarised by a robust standard deviation (1.4826 x MAD)
in greys, and the common part's lag-1 autocorrelation (near -1: planes alternating bright and dark; near 0:
independent jumps; positive: slower waves).

The pyramid scales above 0 are downsampled in z as well (a plane at scale k averages 2^k slices), which
hides most single-slice flicker. For slice-level flicker use --scale 0 with --box-px (e.g. 512): a square
of that size at the centre of each grid cell, read through every plane.

With --config (the run's config, for its preview statistics and display windows): the same for
  raw        each tile's tissue median (preview t_p50) in greys of its display window (255 / (hi - lo)),
             i.e. the flicker a fixed window would show;
  windows    each tile's display-window shift (lo) in the same greys, i.e. the correction render applies;
  raw - windows   what is left after the correction, which the rendered volume should show.
Raw "common" is the median over tiles per slice, "own" each tile minus that.

Writes DIR/zflicker.csv (series, z, box, value, resid), DIR/zflicker.json (summary) and DIR/zflicker.png.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def robust_sd(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(1.4826 * np.median(np.abs(a - np.median(a)))) if a.size else float("nan")


def lag1(a):
    a = np.asarray(a, float)
    ok = np.isfinite(a[:-1]) & np.isfinite(a[1:])
    if ok.sum() < 3 or np.std(a[:-1][ok]) == 0 or np.std(a[1:][ok]) == 0:
        return float("nan")
    x, y = a[:-1][ok], a[1:][ok]
    return float(np.corrcoef(x, y)[0, 1])


def box_means(path, scale, grid, box_px=None):
    """DataFrame (z, box, value): mean grey of pixels > 0 per grid box and plane (with box_px, of a box_px
    square at the centre of each grid cell instead of the whole cell)."""
    from pipeline import omezarr
    arr = omezarr.open_scale(path, scale)
    nz, ny, nx = (int(v) for v in arr.shape)
    by, bx = ny // grid, nx // grid
    rows = []
    if box_px:
        b = min(int(box_px), by, bx)
        for gy in range(grid):
            for gx in range(grid):
                y0, x0 = gy * by + (by - b) // 2, gx * bx + (bx - b) // 2
                col = arr[:, y0:y0 + b, x0:x0 + b].read().result()
                n = (col > 0).sum(axis=(1, 2))
                s = col.sum(axis=(1, 2), dtype=np.float64)
                rows += [(z, f"{gy}-{gx}", s[z] / n[z]) for z in range(nz) if n[z] >= 0.5 * b * b]
                print(f"  {Path(path).name}: box {gy}-{gx} of {grid}x{grid}", flush=True)
        return pd.DataFrame(rows, columns=["z", "box", "value"])
    depth = int(arr.chunk_layout.read_chunk.shape[0] or 64)
    for z0 in range(0, nz, depth):
        slab = arr[z0:min(z0 + depth, nz), :by * grid, :bx * grid].read().result()
        for i, plane in enumerate(slab):
            a = plane.reshape(grid, by, grid, bx)
            n = (a > 0).sum(axis=(1, 3))
            s = a.sum(axis=(1, 3), dtype=np.float64)
            for (gy, gx), cnt in np.ndenumerate(n):
                if cnt >= 0.5 * by * bx:
                    rows.append((z0 + i, f"{gy}-{gx}", s[gy, gx] / cnt))
        print(f"  {Path(path).name}: planes {z0}-{min(z0 + depth, nz) - 1} of {nz}", flush=True)
    return pd.DataFrame(rows, columns=["z", "box", "value"])


def residuals(df, window):
    """Add resid (value minus its running mean over z, per box) and split into common and own."""
    # A running mean, not a median: a median over an odd window absorbs planes that alternate in brightness.
    df = df.sort_values(["box", "z"]).copy()
    df["resid"] = df["value"] - df.groupby("box")["value"].transform(
        lambda s: s.rolling(int(window), center=True, min_periods=1).mean())
    common = df.groupby("z")["resid"].median()
    df["own"] = df["resid"] - df["z"].map(common)
    return df, common


def summarise(df, common):
    return {"flicker_common_sd": round(robust_sd(common.to_numpy()), 3),
            "flicker_own_sd": round(robust_sd(df["own"].to_numpy()), 3),
            "flicker_total_sd": round(robust_sd(df["resid"].to_numpy()), 3),
            "common_lag1": round(lag1(common.sort_index().to_numpy()), 3),
            "largest_common": [[int(z), round(float(v), 2)]
                               for z, v in common.abs().sort_values(ascending=False).head(8).items()],
            "planes": int(common.size), "boxes": int(df["box"].nunique())}


def raw_series(cfg_path, window):
    """(name, DataFrame z, box=tile, value) for the raw tile medians, the window shifts and their difference,
    in greys of each tile's display window."""
    from pipeline.config import load_config, step_defaults, step_path
    from pipeline import intensity
    cfg = load_config(cfg_path, step_defaults=step_defaults("intensity"))
    stats = pd.read_csv(step_path(cfg, "preview", "stats.csv"), dtype={"tile": str})[["z", "tile", "t_p50"]]
    lv = pd.read_csv(intensity.levels_path(cfg), dtype={"tile": str})
    df = lv.merge(stats, on=["z", "tile"], how="inner").rename(columns={"tile": "box"})
    k = 255.0 / (df["hi"] - df["lo"])
    return [("raw (fixed window)", df.assign(value=df["t_p50"] * k)[["z", "box", "value"]]),
            ("windows (correction)", df.assign(value=df["lo"] * k)[["z", "box", "value"]]),
            ("raw - windows", df.assign(value=(df["t_p50"] - df["lo"]) * k)[["z", "box", "value"]])]


def draw(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.2 * n + 0.8), sharex=True, layout="constrained", squeeze=False)
    lim = max(np.nanpercentile(np.abs(df["resid"]), 99.5) for _, df, _, _ in results)
    for ax, (name, df, common, s) in zip(axes[:, 0], results):
        for _, g in df.groupby("box"):
            ax.plot(g["z"], g["resid"], color="0.75", lw=0.5)
        ax.plot(common.index, common.to_numpy(), color="black", lw=1.0)
        ax.axhline(0, color="0.4", lw=0.5)
        ax.set_ylim(-lim, lim)
        ax.set_ylabel("greys")
        ax.set_title(f"{name}: common {s['flicker_common_sd']:.2f}, own {s['flicker_own_sd']:.2f} greys "
                     f"(robust SD); common lag-1 {s['common_lag1']:.2f}", fontsize=10, loc="left")
    axes[-1, 0].set_xlabel("z (plane)")
    fig.suptitle("Brightness minus its running mean over z: each box or tile (grey), median over them (black)",
                 fontsize=11)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("volumes", nargs="*", help="NAME=VOLUME (an OME-Zarr folder); none with --config: raw tiles and "
                   "windows only (the flicker the levels predict, without reading a volume)")
    p.add_argument("--out", required=True)
    p.add_argument("--config", help="the run's config: also measure raw tile medians and display windows")
    p.add_argument("--scale", type=int, default=3)
    p.add_argument("--grid", type=int, default=6)
    p.add_argument("--box-px", type=int, help="measure a square of this size (px at --scale) at the centre of each grid "
                   "cell instead of the whole cell: full z resolution at scale 0 without reading the whole volume")
    p.add_argument("--window", type=int, default=9, help="running-mean window over z (planes)")
    args = p.parse_args(argv)
    if not args.volumes and not args.config:
        p.error("give at least one NAME=VOLUME, or --config")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    series = []
    if args.config:
        series += raw_series(args.config, args.window)
    for spec in args.volumes:
        name, _, path = spec.partition("=")
        if not path:
            sys.exit(f"expected NAME=VOLUME, got {spec!r}")
        series.append((name, box_means(path, args.scale, args.grid, args.box_px)))
    results, frames, summary = [], [], {"scale": args.scale, "grid": args.grid, "window": args.window, "series": {}}
    for name, df in series:
        df, common = residuals(df, args.window)
        s = summarise(df, common)
        summary["series"][name] = s
        results.append((name, df, common, s))
        frames.append(df.assign(series=name)[["series", "z", "box", "value", "resid"]])
        print(f"{name}: {json.dumps(s)}", flush=True)
    pd.concat(frames).to_csv(out / "zflicker.csv", index=False, float_format="%.4f")
    (out / "zflicker.json").write_text(json.dumps(summary, indent=1))
    draw(results, out / "zflicker.png")
    print(f"wrote {out}/zflicker.csv, .json, .png")


if __name__ == "__main__":
    main()
