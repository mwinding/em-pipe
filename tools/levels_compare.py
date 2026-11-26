"""Compare two sets of display levels of the same dataset (work/intensity[/<output>]/levels.csv), e.g. the
levels of an automatic run with those of a hand-made config.

    python tools/levels_compare.py A/levels.csv B/levels.csv [--labels a b] [--out DIR]

Levels are the raw values render maps to grey 0 (lo) and 255 (hi), per (z, tile). For every (z, tile) both
files have, in greys of A's window:

    d_lo, d_hi    how far B's lo and hi are from A's: 255 (lo_B - lo_A) / (hi_A - lo_A), likewise hi
    d_grey        the grey B gives the raw value that A puts at mid-grey (127.5), minus 127.5: the brightness
                  difference a viewer would see
    width_ratio   (hi_B - lo_B) / (hi_A - lo_A): above 1, B shows less contrast

d_grey is split into the part a slice's tiles share (the median over its tiles, per z: a slice-wide brightness
difference) and each tile's own part (what is left: a difference between tiles of one slice). Printed: for
each quantity the median, the 5th and 95th percentile and the largest absolute value; --out also writes
levels_compare.csv (per z, tile) and levels_compare.json there.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def compare(a, b):
    """Per-(z, tile) table of the differences between level tables a and b (columns z, tile, lo, hi)."""
    m = a.merge(b, on=["z", "tile"], suffixes=("_a", "_b"))
    wa, wb = m["hi_a"] - m["lo_a"], m["hi_b"] - m["lo_b"]
    m["d_lo"] = 255.0 * (m["lo_b"] - m["lo_a"]) / wa
    m["d_hi"] = 255.0 * (m["hi_b"] - m["hi_a"]) / wa
    mid = m["lo_a"] + 0.5 * wa
    m["d_grey"] = 255.0 * (mid - m["lo_b"]) / wb - 127.5
    m["width_ratio"] = wb / wa
    m["d_grey_slice"] = m.groupby("z")["d_grey"].transform("median")
    m["d_grey_tile"] = m["d_grey"] - m["d_grey_slice"]
    return m


def summarise(m, n_a, n_b):
    out = {"compared": int(len(m)), "only_in_a": int(n_a - len(m)), "only_in_b": int(n_b - len(m)),
           "slices": int(m["z"].nunique())}
    for col in ("d_lo", "d_hi", "d_grey", "d_grey_slice", "d_grey_tile", "width_ratio"):
        v = m[col].to_numpy(float)
        centre = 1.0 if col == "width_ratio" else 0.0
        out[col] = {"median": round(float(np.median(v)), 4), "p5": round(float(np.percentile(v, 5)), 4),
                    "p95": round(float(np.percentile(v, 95)), 4),
                    "largest": round(float(v[np.argmax(np.abs(v - centre))]), 4)}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--labels", nargs=2, default=["A", "B"])
    p.add_argument("--out", help="folder for levels_compare.csv and .json")
    args = p.parse_args(argv)
    a, b = (pd.read_csv(f, dtype={"tile": str})[["z", "tile", "lo", "hi"]] for f in (args.a, args.b))
    m = compare(a, b)
    if not len(m):
        sys.exit("levels_compare: the two files share no (z, tile)")
    s = {"a": args.labels[0], "b": args.labels[1], **summarise(m, len(a), len(b))}
    print(f"{s['compared']} (z, tile) in both over {s['slices']} slices ({s['only_in_a']} only in {s['a']}, "
          f"{s['only_in_b']} only in {s['b']}); differences {s['b']} - {s['a']} in greys of {s['a']}'s window:")
    names = {"d_lo": "lo", "d_hi": "hi", "d_grey": "brightness (grey of A's mid-grey value)",
             "d_grey_slice": "  shared by a slice's tiles", "d_grey_tile": "  each tile's own", "width_ratio": "window width ratio"}
    for col, name in names.items():
        v = s[col]
        print(f"  {name:42s} median {v['median']:8.3f}   p5 {v['p5']:8.3f}   p95 {v['p95']:8.3f}   largest {v['largest']:8.3f}")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        m.to_csv(out / "levels_compare.csv", index=False, float_format="%.4f")
        (out / "levels_compare.json").write_text(json.dumps(s, indent=1))
        print(f"wrote {out}/levels_compare.csv, .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
