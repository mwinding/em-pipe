"""Tissue-only percentiles per tile slice from the preview thumbnails (test of masked normalisation).

Reads <main>/work/preview/{stats.csv, thumbs/}; masks large uniform bright / dark areas in each
thumbnail; writes <test>/work/preview/stats.csv = the main stats with the tissue-only p0.5 / p50 /
p99.5 in the p1 / p50 / p99 columns (p0_5 / p99_5 stay: they decode the thumbnails) and a
tissue_fraction column.
"""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, os.path.expandvars("${EM_PIPE_ROOT}"))
from pipeline.preview import thumb_values  # noqa: E402

MAIN, TEST = Path(sys.argv[1]), Path(sys.argv[2])
if TEST.resolve() == MAIN.resolve():
    sys.exit("masked_stats: <test> must be a separate scratch copy, not <main> (it rewrites <test>/work/preview/stats.csv)")
stats = pd.read_csv(MAIN / "work/preview/stats.csv", dtype={"tile": str})
index = pd.read_csv(MAIN / "work/preview/thumbs/index.csv", dtype={"tile": str})
df = stats.merge(index[["z", "tile", "npy", "i"]], on=["z", "tile"], how="left")


def background(t8):
    t = t8.astype(np.float32)
    sm = ndimage.gaussian_filter(t, 1.5)
    sd = np.sqrt(np.maximum(ndimage.gaussian_filter(t * t, 3) - ndimage.gaussian_filter(t, 3) ** 2, 0))
    bg = ((sm > 235) | (sm < 20)) & (sd < 10)
    bg = ndimage.binary_opening(bg, iterations=2)
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
        frac = 1 - bg.mean()
        if frac < 0.05:
            out.append((r["z"], r["tile"], r["p0_5"], r["p50"], r["p99_5"], frac))
            continue
        v = thumb_values(t8[~bg], r["p0_5"], r["p99_5"])
        p = np.percentile(v, [0.5, 50, 99.5])
        out.append((r["z"], r["tile"], p[0], p[1], p[2], frac))
    return out


if __name__ == "__main__":
    jobs = [(npy, g.to_dict("records")) for npy, g in df.dropna(subset=["npy"]).groupby("npy")]
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 16) as pool:
        res = [x for part in pool.map(one_file, jobs, chunksize=4) for x in part]
    m = pd.DataFrame(res, columns=["z", "tile", "m_p0_5", "m_p50", "m_p99_5", "tissue_fraction"])
    out = stats.merge(m, on=["z", "tile"], how="left")
    miss = out["m_p0_5"].isna()
    out.loc[miss, "m_p0_5"], out.loc[miss, "m_p50"], out.loc[miss, "m_p99_5"] = out["p0_5"], out["p50"], out["p99_5"]
    out["tissue_fraction"] = out["tissue_fraction"].fillna(1.0)
    report = out.assign(d_lo=out["m_p0_5"] - out["p0_5"], d_hi=out["m_p99_5"] - out["p99_5"])
    out["p1"], out["p50"], out["p99"] = out["m_p0_5"], out["m_p50"], out["m_p99_5"]
    (TEST / "work/preview").mkdir(parents=True, exist_ok=True)
    out.drop(columns=["m_p0_5", "m_p50", "m_p99_5"]).to_csv(TEST / "work/preview/stats.csv", index=False)
    bgd = report[report["tissue_fraction"] < 0.95]
    print(f"{len(out)} tile slices; {len(bgd)} ({len(bgd) / len(out):.0%}) have > 5% background "
          f"(median {1 - bgd['tissue_fraction'].median():.0%} of the tile)")
    print(f"  in those: black point moved by median {bgd['d_lo'].median():+.0f}, white point by median "
          f"{bgd['d_hi'].median():+.0f} raw grey levels (window median {(out['p99_5'] - out['p0_5']).median():.0f} wide)")
    for z in (3270, 3290, 8000, 15000):
        r = report[report["z"] == z]
        if len(r):
            print(f"  z {z}: " + "  ".join(f"{t}: tissue {f:.0%} lo {lo:+.0f} hi {hi:+.0f}"
                                         for t, f, lo, hi in zip(r["tile"], r["tissue_fraction"], r["d_lo"], r["d_hi"])))
