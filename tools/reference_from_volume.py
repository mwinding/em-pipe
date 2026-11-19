"""Reference tissue numbers of a volume as it is rendered now: for intensity.window: reference.

The refstats step measures the reference tissue (neuropil) on the common scale before any window
(median, 5th and 95th percentile per sampled slice). This tool maps those through the window the
volume was rendered with (its intensity levels, work/intensity/levels.csv, or --levels) and reports,
in greys:

  median, p5, p95 and the p5-p95 width of the reference tissue, per sampled slice and as the median
  over slices (the numbers to use as intensity.reference.median / contrast when this volume is the
  reference), plus lc32 (local contrast) as a cross-check.

For the reference volume itself this is also its consistency check: the numbers describe exactly the
grey values render wrote, because they use the same corrections and the same window.

  python tools/reference_from_volume.py --config CONFIG [--levels LEVELS.csv] [--out OUT.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import intensity, patchgrid, refstats, shading  # noqa: E402
from pipeline.config import deep_merge, load_config  # noqa: E402


def reference_numbers(cfg, levels):
    """Per sampled slice (DataFrame) and summary (dict) of the reference tissue in greys under levels."""
    st = intensity.load_refstats(cfg)
    bal = intensity.balanced_scale(cfg)[["z", "tile", "g", "o"]]
    lv = levels[["z", "tile", "lo", "hi"]].merge(bal, on=["z", "tile"], how="inner")
    w = intensity.common_window(lv)
    st = st[st["n_cells"] > 0].merge(w, left_on="z", right_index=True, how="inner")
    out = pd.DataFrame({"z": st["z"], "n_cells": st["n_cells"]})
    for name, col in (("median", "median"), ("p5", "rq_5"), ("p95", "rq_95")):
        out[name] = intensity.grey(st[col], st["Lc"], st["Hc"])
    out["width"] = out["p95"] - out["p5"]
    out["lc32"] = 255 * st["lc32"] / (st["Hc"] - st["Lc"])
    summary = {"sampled_slices": int(len(out)),
               **{k: round(float(out[k].median()), 2) for k in ("median", "p5", "p95", "width", "lc32")},
               "width_p10_p90_over_slices": [round(float(v), 2) for v in np.percentile(out["width"], [10, 90])]
               if len(out) else None}
    return out, summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--levels", help="levels CSV (z, tile, lo, hi); default: the config's intensity levels")
    p.add_argument("--out", help="write the summary and per-slice table here (JSON)")
    a = p.parse_args(argv)
    defaults = deep_merge(deep_merge(deep_merge(intensity.DEFAULTS, shading.DEFAULTS), patchgrid.DEFAULTS),
                          refstats.DEFAULTS)
    cfg = load_config(a.config, step_defaults=defaults)
    path = a.levels or intensity.levels_path(cfg)
    levels = pd.read_csv(path, dtype={"tile": str})
    per, summary = reference_numbers(cfg, levels)
    summary["levels"] = str(path)
    print(json.dumps(summary, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"summary": summary, "slices": per.round(3).to_dict("records")}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
