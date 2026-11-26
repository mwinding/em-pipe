"""Compare two refstats tables of the same volume, e.g. with and without decurtaining.

The refstats step measures the reference tissue (neuropil) per sampled slice on the common scale,
before any window. Two runs of it on the same volume and the same sampled slices (for example the
plain tiles and the decurtained ones, refstats.output) differ only by what was changed between them.
This tool lines the two up by z and reports, per segment and overall:

  the neuropil median, the p5-p95 width and lc32 of each run (medians over the segment's samples),
  the ratio B / A of each, and how much the width varies within the segment in each run.

The width ratio is what intensity.window: reference would change: the window width is proportional
to the neuropil width, so B / A = 0.97 means B's window is 3 % narrower (3 % more contrast).

  python dev/2026-11_reference_standard/tools/refstats_compare.py STATS_A.csv STATS_B.csv [--labels plain decurtained]
         [--out compare.png] [--json compare.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # the repository root

COLS = ("median", "width", "lc32")


def load(path):
    st = pd.read_csv(path)
    st = st[st["n_cells"] > 0].copy()
    st["width"] = st["rq_95"] - st["rq_5"]
    return st[["z", "segment", "n_cells", *COLS]]


def pair(a, b):
    """The samples measured in both runs, side by side (suffixes _a, _b) with the ratios B / A."""
    m = a.merge(b, on=["z", "segment"], suffixes=("_a", "_b"))
    for c in COLS:
        m[f"{c}_ratio"] = m[f"{c}_b"] / m[f"{c}_a"]
    return m.sort_values("z").reset_index(drop=True)


def summary(m):
    """Per segment and overall: medians of each run, ratio of medians, within-segment width CV (%)."""
    rows = []
    for seg, g in [*m.groupby("segment"), ("all", m)]:
        r = {"segment": seg, "z0": int(g["z"].min()), "z1": int(g["z"].max()), "n": len(g)}
        for c in COLS:
            r[f"{c}_a"], r[f"{c}_b"] = float(g[f"{c}_a"].median()), float(g[f"{c}_b"].median())
            r[f"{c}_ratio"] = float(g[f"{c}_ratio"].median())
        for s in ("a", "b"):
            r[f"width_cv_{s}"] = float(100 * g[f"width_{s}"].std() / g[f"width_{s}"].mean()) if len(g) > 1 else 0.0
        rows.append(r)
    return pd.DataFrame(rows)


def plot(m, summ, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pipeline.preview import save
    ca, cb = "0.55", "#2a78d6"
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(13, 9), layout="constrained")
    segs = summ[summ["segment"] != "all"]
    for k, s in enumerate(segs.itertuples()):
        for ax in axes:
            if k % 2:
                ax.axvspan(s.z0, s.z1, color="0.93", lw=0)
            if k:
                ax.axvline(s.z0, color="0.6", lw=0.8)
        axes[1].text((s.z0 + s.z1) / 2, 1.0, f"seg {s.segment}\n×{s.width_ratio:.3f}",
                     transform=axes[1].get_xaxis_transform(), ha="center", va="top", fontsize=8, color="#1a1a19")
        axes[2].text((s.z0 + s.z1) / 2, 1.0, f"seg {s.segment}\n×{s.median_ratio:.4f}",
                     transform=axes[2].get_xaxis_transform(), ha="center", va="top", fontsize=8, color="#1a1a19")
    axes[0].plot(m["z"], m["width_a"], "o", ms=3, color=ca, label=labels[0])
    axes[0].plot(m["z"], m["width_b"], "o", ms=3, color=cb, label=labels[1])
    axes[0].set_ylabel("neuropil p5-p95 width\n(raw, common scale)")
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    axes[1].plot(m["z"], m["width_ratio"], "o", ms=3, color=cb)
    axes[1].axhline(1.0, color="0.4", lw=0.8)
    axes[1].set_ylabel(f"width ratio\n{labels[1]} / {labels[0]}")
    axes[2].plot(m["z"], m["median_ratio"], "o", ms=3, color=cb)
    axes[2].axhline(1.0, color="0.4", lw=0.8)
    axes[2].set_ylabel(f"median ratio\n{labels[1]} / {labels[0]}")
    axes[2].set_xlabel("z")
    axes[0].set_title(f"Neuropil before any window, {labels[1]} vs {labels[0]}, same sampled slices. "
                      "Labels: each segment's median ratio", loc="left", fontsize=10)
    save(fig, path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("stats_a", type=Path)
    p.add_argument("stats_b", type=Path)
    p.add_argument("--labels", nargs=2, default=["A", "B"])
    p.add_argument("--out", type=Path, help="plot (PNG)")
    p.add_argument("--json", type=Path, help="per-segment summary (JSON)")
    args = p.parse_args(argv)
    m = pair(load(args.stats_a), load(args.stats_b))
    if m.empty:
        sys.exit("no sampled slice is measured in both tables")
    summ = summary(m)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summ.round({**{f"{c}_{s}": 1 for c in COLS for s in "ab"}, **{f"{c}_ratio": 4 for c in COLS},
                          "width_cv_a": 2, "width_cv_b": 2}).to_string(index=False))
    if args.out:
        plot(m, summ, args.labels, args.out)
    if args.json:
        args.json.write_text(json.dumps({"labels": args.labels, "a": str(args.stats_a), "b": str(args.stats_b),
                                         "segments": summ.to_dict("records")}, indent=1))


if __name__ == "__main__":
    main()
