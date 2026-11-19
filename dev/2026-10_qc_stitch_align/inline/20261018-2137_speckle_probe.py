# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-18T21:37:21.905Z
# Ran: local
# Heredoc line: pkill -f "grep -o .em-clipping" 2>/dev/null; mkdir -p ${DEV_WORK}/cluster && cat > ${DEV_WORK}
# Assistant note before it: I'm checking whether the grey speckle in the "unclipped" panels is real detail or a bug in my check, since it decides the recommendation.

"""Where does the widened render stop being a monotone map of the volume's render? (z 4000 neuropil crop)"""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "tools")
import json
import numpy as np
import pandas as pd
import cv2
from pipeline import render, shading, patchgrid, omezarr
from pipeline.config import deep_merge, load_config
import clipping_check as cc

cfg = load_config(os.path.expandvars("configs/${DEV_PREFIX}35h.yaml"),
                  step_defaults=deep_merge(deep_merge(render.DEFAULTS, shading.DEFAULTS), patchgrid.DEFAULTS))
root, wdir = render.volume_paths(cfg)
meta0 = json.loads((wdir / "render.json").read_text())
tiles = pd.read_csv(wdir / "tiles.csv", dtype={"tile": str, "file": str})
z, R, C, c = 4000, 15244, 21403, 1024
rows, cols = slice(R - c // 2, R + c // 2), slice(C - c // 2, C + c // 2)

stages = {}
orig_tile, orig_slice, orig_warp, orig_csa = (render._Renderer._tile, render._Renderer._slice,
                                              render._Renderer._warp, cv2.convertScaleAbs)
cur = [None]

def tile(self, t):
    r = orig_tile(self, t)
    if r is not None:
        v = r[1]
        stages.setdefault(cur[0], {}).setdefault("tile", []).append(
            (t["tile"], float(np.nanmin(v)), float(np.nanmax(v)), int(np.isnan(v).sum()), str(v.dtype)))
    return r

def slc(self, futures):
    acc, valid = orig_slice(self, futures)
    stages.setdefault(cur[0], {})["slice"] = acc.copy()
    return acc, valid

def warp(self, z, val, valid, **kw):
    out, ok = orig_warp(self, z, val, valid, **kw)
    stages.setdefault(cur[0], {})["warp"] = out.copy()
    return out, ok

render._Renderer._tile, render._Renderer._slice, render._Renderer._warp = tile, slc, warp
for name, widen in (("same", None), ("wide", (15.0, 240.0))):
    cur[0] = name
    stages.setdefault(name, {})["out"] = cc.render_crop(cfg, meta0, tiles, z, rows, cols, widen)

plan = tiles[tiles["z"] == z]
print(plan[["tile", "lo", "hi"]].to_string())
for name in ("same", "wide"):
    s = stages[name]
    print(name, "tiles (tile, min, max, nans, dtype):", s["tile"])
    for k in ("slice", "warp"):
        a = s[k]
        print(f"  {k}: dtype {a.dtype} min {np.nanmin(a):.4f} max {np.nanmax(a):.4f} nan {int(np.isnan(a).sum())}")
# stage-by-stage relation: where same == 0 in the output, what is wide at each stage?
rot = meta0["settings"].get("rotate_deg", 0) == 180
m = stages["same"]["out"] == 0
m_unrot = m[::-1, ::-1] if rot else m
for k in ("slice", "warp"):
    a, b = stages["same"][k], stages["wide"][k]
    print(f"{k}: same at out==0 px: pct {np.percentile(a[m_unrot], [1, 50, 99]).round(4)}; "
          f"wide there: pct {np.percentile(b[m_unrot], [1, 50, 99]).round(4)}")
    pred = 15 / 255 + a * 225 / 255
    d = b - pred
    print(f"  {k}: wide - predicted: pct {np.percentile(d, [0.1, 50, 99.9]).round(4)}, |d|>0.02: {(np.abs(d) > 0.02).mean():.4f}")
w = stages["wide"]["out"]
print("out: wide at same==0: pct", np.percentile(w[m], [5, 50, 95]))
