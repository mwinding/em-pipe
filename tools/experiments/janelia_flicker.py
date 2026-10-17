"""Slice-to-slice brightness change of tile interiors in before/after stacks of consecutive slices.

    python tools/experiments/janelia_flicker.py OUT_CSV CONFIG OLD_VOLUME Z[,Z...]|@ZFILE NAME=DIR [NAME=DIR ...]

Per slice and tile: the mean grey of the tile's interior (the middle half of the tile in x and y,
mapped into the 32 nm plane with the render plan's transforms; no-data pixels left out) in frame 1
of each DIR (and frame 0 of the first DIR as 'before'). Prints, per column, the median and largest
|change| between consecutive slices, of each tile's mean ('abs') and of each tile's mean minus the
mean over the slice's tiles ('rel': tile-against-tile flicker, which a seam would show). Writes the
per-(z, tile) means to OUT_CSV.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline import render, shading, transforms  # noqa: E402
from pipeline.config import deep_merge, load_config  # noqa: E402


def main():
    out_csv = Path(sys.argv[1])
    cfg = load_config(sys.argv[2], step_defaults=deep_merge(render.DEFAULTS, shading.DEFAULTS))
    old = sys.argv[3]
    text = Path(sys.argv[4][1:]).read_text() if sys.argv[4].startswith("@") else sys.argv[4]
    zs = sorted({int(v) for v in text.replace("\n", ",").split(",") if v.strip()})
    dirs = dict(a.split("=", 1) for a in sys.argv[5:])
    meta = json.loads((Path(cfg["output_dir"]) / "work" / "render" / old.removesuffix(".zarr").removesuffix(".ome")
                       / "render.json").read_text())
    ox, oy = meta["origin_xy"]
    f = 4
    plan = render._tile_plan(cfg)
    plan = plan[plan["z"].isin(zs)]
    rows = []
    for z in zs:
        stacks = {name: tifffile.imread(Path(d) / f"z{z}.tif") for name, d in dirs.items()}
        images = {"before": stacks[next(iter(dirs))][0], **{name: s[1] for name, s in stacks.items()}}
        for t in plan[plan["z"] == z].to_dict("records"):
            T = transforms.from_row(t)
            w, h = t["width"], t["height"]
            pts = transforms.apply(T, [[w / 4, h / 4], [3 * w / 4, h / 4], [w / 4, 3 * h / 4], [3 * w / 4, 3 * h / 4]])
            x0, y0 = ((pts.min(axis=0) - [ox, oy] - 1.5) / f).astype(int)
            x1, y1 = ((pts.max(axis=0) - [ox, oy] - 1.5) / f).astype(int)
            row = {"z": z, "tile": t["tile"]}
            for name, img in images.items():
                crop = img[max(y0, 0):y1, max(x0, 0):x1]
                row[name] = float(crop[crop > 0].mean()) if (crop > 0).any() else np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, float_format="%.3f")
    cols = list(images)
    wide = df.pivot(index="z", columns="tile", values=cols)
    summary = []
    for name in cols:
        m = wide[name]
        d = m.diff().iloc[1:]
        rel = (m.sub(m.mean(axis=1), axis=0)).diff().iloc[1:]
        summary.append({"column": name, "abs_median": float(d.abs().median().median()), "abs_max": float(d.abs().max().max()),
                        "rel_median": float(rel.abs().median().median()), "rel_max": float(rel.abs().max().max())})
    print(df.to_string(index=False, float_format="%.1f"))
    print(pd.DataFrame(summary).to_string(index=False, float_format="%.2f"))


if __name__ == "__main__":
    main()
