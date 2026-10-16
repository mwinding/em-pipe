"""Before/after of the current intensity levels (and shading fields) on chosen slices, at 32 nm.

    python tools/shading_before_after.py CONFIG OLD_VOLUME Z[,Z...] [NAME]

old = OLD_VOLUME's 32 nm plane (an earlier render in the same output_dir, its render.json under
work/render/); new = the slice rendered now from the raw tiles with the current levels and fields.
Writes <output_dir>/qc/<NAME, default shading_test>/z<z>.tif (ImageJ stack: old, new) and z<z>.png side by side.
Runs on NEMO (reads the raw tiles): sbatch --cpus-per-task=8 --mem=64G --wrap "source slurm/env.sh; python ..."
"""
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import omezarr, render, shading  # noqa: E402
from pipeline.config import deep_merge, load_config  # noqa: E402

CFG, OLD, ZS = sys.argv[1], sys.argv[2], [int(v) for v in sys.argv[3].split(",")]
NAME = sys.argv[4] if len(sys.argv) > 4 else "shading_test"
cfg = load_config(CFG, step_defaults=deep_merge(render.DEFAULTS, shading.DEFAULTS))
out = Path(cfg["output_dir"])
qc = out / "qc" / NAME
qc.mkdir(parents=True, exist_ok=True)
old_stem = OLD.removesuffix(".zarr").removesuffix(".ome")
old_meta = json.loads((out / "work" / "render" / old_stem / "render.json").read_text())
f = 4
w, h = old_meta["canvas_size_xy"]
meta = {**old_meta, "downsample": f, "shape": [1, math.ceil(h / f), math.ceil(w / f)],
        "settings": {**old_meta["settings"], "downsample": f}, "shading": shading.digest(cfg)}
plan = render._tile_plan(cfg)
plan = plan[plan["z"].isin(ZS)]
old = omezarr.open_scale(out / OLD, 2)
with ThreadPoolExecutor(8) as pool:
    r = render._Renderer(cfg, meta, plan, pool)
    for z in ZS:
        new = r.plane([(z, 1.0)])
        o = old[z // 4].read().result()
        hh, ww = min(o.shape[0], new.shape[0]), min(o.shape[1], new.shape[1])
        o, new = o[:hh, :ww], new[:hh, :ww]
        tifffile.imwrite(qc / f"z{z}.tif", np.stack([o, new]), imagej=True, resolution=(1 / 0.032, 1 / 0.032),
                         metadata={"spacing": 0.032, "unit": "micron", "axes": "ZYX",
                                   "Labels": [f"z {z}: old levels, no flat-field (32 nm plane {z // 4})",
                                              f"z {z}: new levels + flat-field"]})
        side = np.concatenate([o, np.full((hh, 20), 255, np.uint8), new], axis=1)
        side = cv2.resize(side, (side.shape[1] // 3, side.shape[0] // 3), interpolation=cv2.INTER_AREA)
        cv2.putText(side, f"z {z}   left: before   right: after (flat-field + tissue-only balance)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)
        cv2.imwrite(str(qc / f"z{z}.png"), side)
        print("done", z, flush=True)
