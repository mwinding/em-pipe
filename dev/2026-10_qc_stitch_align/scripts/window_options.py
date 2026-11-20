"""Predicted rendered brightness through z for several per-tile window choices (35h).

Tissue-only percentiles per tile slice from the preview thumbnails (large uniform bright / dark
areas masked); windows smoothed over 51 slices per tile (restarting at layout changes) as the
intensity step does; output grey of the tissue p5 / p50 / p95 = 255 (p - lo) / (hi - lo), clipped.
Writes a plot and prints flatness scores. Percentiles are cached in <out>.parquet-like CSV.
"""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.expandvars("${EM_PIPE_ROOT}"))
from pipeline.preview import thumb_values  # noqa: E402

MAIN = Path(os.path.expandvars("${DEV_35H}"))
OUT = Path(sys.argv[1])
PCTS = [1, 5, 25, 50, 75, 95, 99]
stats = pd.read_csv(MAIN / "work/preview/stats.csv", dtype={"tile": str}, parse_dates=["timestamp"])
index = pd.read_csv(MAIN / "work/preview/thumbs/index.csv", dtype={"tile": str})
seg = pd.read_csv(MAIN / "work/check/slices.csv", usecols=["z", "segment"]).drop_duplicates("z")
df = stats.merge(index[["z", "tile", "npy", "i"]], on=["z", "tile"]).merge(seg, on="z")


def background(t8):
    t = t8.astype(np.float32)
    sm = ndimage.gaussian_filter(t, 1.5)
    sd = np.sqrt(np.maximum(ndimage.gaussian_filter(t * t, 3) - ndimage.gaussian_filter(t, 3) ** 2, 0))
    bg = ndimage.binary_opening(((sm > 235) | (sm < 20)) & (sd < 10), iterations=2)
    lab, n = ndimage.label(bg)
    if n:
        sizes = ndimage.sum(bg, lab, range(1, n + 1))
        bg = np.isin(lab, 1 + np.flatnonzero(sizes >= 400))
    return bg


def one_file(args):
    npy, rows = args
    arr = np.load(MAIN / "work/preview/thumbs" / npy, mmap_mode="r")
    out = []
    for r in rows:
        t8 = np.asarray(arr[int(r["i"])])
        bg = background(t8)
        keep = ~bg if (~bg).mean() >= 0.05 else np.ones_like(bg)
        v = thumb_values(t8[keep], r["p0_5"], r["p99_5"])
        out.append((r["z"], r["tile"], *np.percentile(v, PCTS)))
    return out


if __name__ == "__main__":
    cache = OUT.with_suffix(".csv")
    if cache.exists():
        tp = pd.read_csv(cache, dtype={"tile": str})
    else:
        jobs = [(npy, g.to_dict("records")) for npy, g in df.groupby("npy")]
        with Pool(16) as pool:
            res = [x for part in pool.map(one_file, jobs, chunksize=4) for x in part]
        tp = pd.DataFrame(res, columns=["z", "tile"] + [f"t{p}" for p in PCTS])
        tp.to_csv(cache, index=False)
    d = df[["z", "tile", "timestamp", "segment", "p0_5", "p99_5"]].merge(tp, on=["z", "tile"]).sort_values(["tile", "z"])

    def smooth(col):   # 51-slice running median per tile, restarting at layout changes
        return d.groupby(["segment", "tile"])[col].transform(lambda s: s.rolling(51, center=True, min_periods=1).median())

    S = {c: smooth(c) for c in ["p0_5", "p99_5", "t1", "t5", "t50", "t95", "t99"]}
    spread = S["t95"] - S["t5"]
    options = {
        "1. current: all-pixel p0.5 -> 0, p99.5 -> 255": (S["p0_5"], S["p99_5"], 0, 255),
        "2. tissue p1 -> 10, p99 -> 245": (S["t1"], S["t99"], 10, 245),
        os.path.expandvars("3. tissue p5 -> 10, p95 -> 245 (the segmentation collaborator-like)"): (S["t5"], S["t95"], 10, 245),
        "4. median-anchored: tissue p50 -> 150, p5-p95 spans 180 greys": (S["t50"] - spread * 140 / 180, S["t50"] + spread * 105 / 180, 10, 245),
    }
    # option 4 as (lo, hi) at grey 10 / 245: p50 -> 10 + 235 * 140/245 ~ 144 ... set so p50 -> ~150, p5..p95 ~ 180 greys
    fig, axes = plt.subplots(len(options), 1, figsize=(15, 3.2 * len(options)), sharex=True, sharey=True)
    t_of_z = d.groupby("z")["timestamp"].first()
    changes = d.groupby("segment")["z"].min().sort_values().to_numpy()[1:]
    print(f"{d['z'].nunique()} slices, {len(d)} tile slices")
    for ax, (name, (lo, hi, g0, g1)) in zip(axes, options.items()):
        def grey(p):
            return np.clip(g0 + (g1 - g0) * (p - lo) / (hi - lo), 0, 255)
        g = pd.DataFrame({"z": d["z"], "p5": grey(d["t5"]), "p50": grey(d["t50"]), "p95": grey(d["t95"])})
        per_z = g.groupby("z").median().rolling(25, center=True, min_periods=1).median()
        tz = t_of_z.loc[per_z.index]
        ax.fill_between(tz, per_z["p5"], per_z["p95"], color="#2a78d6", alpha=0.15, label="tissue p5-p95")
        ax.plot(tz, per_z["p50"], color="#2a78d6", lw=1.5, label="tissue median")
        for z in changes:
            ax.axvline(t_of_z.loc[z], color="#b5b4ae", lw=0.8, ls="--")
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_ylabel("rendered grey")
        ax.grid(color="#e6e5e0", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        steps = []
        for z in changes:
            before = per_z.loc[(per_z.index < z) & (per_z.index >= z - 500), "p50"].median()
            after = per_z.loc[(per_z.index >= z) & (per_z.index < z + 500), "p50"].median()
            steps.append(after - before)
        day = g.merge(d[["z", "tile", "timestamp"]], on=["z", "tile"] if "tile" in g else "z") if False else None
        print(f"{name}\n   median grey: overall {per_z['p50'].median():.0f}, range {per_z['p50'].min():.0f}-{per_z['p50'].max():.0f}, "
              f"spread of medians (sd) {per_z['p50'].std():.1f}; steps at layout changes {', '.join(f'{s:+.0f}' for s in steps)}; "
              f"tissue p5-p95 span {np.median(per_z['p95'] - per_z['p5']):.0f} greys (range {np.min(per_z['p95'] - per_z['p5']):.0f}-{np.max(per_z['p95'] - per_z['p5']):.0f})")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("35h: predicted rendered brightness for four window choices (dashed: tile layout changes)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT, dpi=95)
