"""Before/after of the shading correction on chosen slices, at 32 nm: old = the finished unrotated
volume's 32 nm plane (old levels, no flat-field); new = the slice rendered now with the new levels
and the flat-field. Writes qc/shading_test/z<z>.tif (ImageJ stack: old, new) and z<z>.png side by side.
"""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import tifffile

sys.path.insert(0, os.path.expandvars("${EM_PIPE_ROOT}"))
from pipeline import omezarr, render, shading  # noqa: E402
from pipeline.config import deep_merge, load_config  # noqa: E402

CFG = os.path.expandvars("${EM_PIPE_ROOT}/configs/${DEV_PREFIX}35h_overview.yaml")
ZS = [int(v) for v in sys.argv[1].split(",")]
cfg = load_config(CFG, step_defaults=deep_merge(render.DEFAULTS, shading.DEFAULTS))
out = Path(cfg["output_dir"])
qc = out / "qc" / "shading_test"
qc.mkdir(parents=True, exist_ok=True)
old_meta = json.loads((out / "work" / "render" / os.path.expandvars("${DEV_PREFIX}35h") / "render.json").read_text())
f = 4
w, h = old_meta["canvas_size_xy"]
meta = {**old_meta, "downsample": f, "shape": [1, math.ceil(h / f), math.ceil(w / f)],
        "settings": {**old_meta["settings"], "downsample": f}, "shading": shading.digest(cfg)}
plan = render._tile_plan(cfg)
plan = plan[plan["z"].isin(ZS)]
old = omezarr.open_scale(out / os.path.expandvars("${DEV_PREFIX}35h.ome.zarr"), 2)
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
