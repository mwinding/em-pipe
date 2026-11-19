# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-18T21:27:32.394Z
# Ran: remote, via ssh
# Heredoc line: cat > ${DEV_WORK}/clip_debug.py <<'EOF'
# Assistant note before it: Yes, for setting the black and white points it is. You've put your finger on the cause. **Why it's like that:** the preview step reads every raw tile once and keeps 1/16-scale versions for statistics, thumbnails and QC. The window percentiles were taken from those because it was cheap and the window

import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expandvars("${EM_PIPE_ROOT}"))
sys.path.insert(0, os.path.expandvars("${EM_PIPE_ROOT}/tools"))
from pipeline import omezarr, patchgrid, render, shading
from pipeline.config import deep_merge, load_config
from pipeline.slices import StackCache
import clipping_check as cc
cfg = load_config(os.path.expandvars("${EM_PIPE_ROOT}/configs/${DEV_PREFIX}35h.yaml"),
                  step_defaults=deep_merge(deep_merge(render.DEFAULTS, shading.DEFAULTS), patchgrid.DEFAULTS))
root, wdir = render.volume_paths(cfg)
meta0 = json.loads((wdir / "render.json").read_text())
tiles = pd.read_csv(wdir / "tiles.csv", dtype={"tile": str, "file": str})
z, r0, c0, c = 400, 2048, 5120, 512
vol = omezarr.open_scale(root, 0)
cur = vol[z, r0:r0 + c, c0:c0 + c].read().result()
wide = cc.render_crop(cfg, meta0, tiles, z, slice(r0, r0 + c), slice(c0, c0 + c), (15.0, 240.0))
same = cc.render_crop(cfg, meta0, tiles, z, slice(r0, r0 + c), slice(c0, c0 + c), None)
blk = cur == 0
print("pixels black in volume:", int(blk.sum()), "of", blk.size)
print("widened values where the volume is 0: percentiles 1/10/50/90/99:", np.percentile(wide[blk], [1, 10, 50, 90, 99]).round(1))
print("widened values where the volume is >= 20:", np.percentile(wide[cur >= 20], [1, 50, 99]).round(1))
print("same-window re-render where volume is 0:", np.percentile(same[blk], [50, 90, 99]).round(1), "| equal overall:", float((same == cur).mean()))
pz = tiles[tiles.z == z]
print("windows (lo, hi) of the tiles at z:", pz[["tile", "lo", "hi"]].round(0).to_dict("records"))
# raw values of the tile that covers the crop centre (rough: the tile whose aligned box contains it)
from pipeline import transforms
H, W = meta0["shape"][1:]
ox, oy = meta0["origin_xy"]
yc, xc = H - 1 - (r0 + c // 2), W - 1 - (c0 + c // 2)       # unrotated canvas
pt = np.array([[ox + xc, oy + yc]], float)
cache = StackCache(cfg["raw_dir"])
for t in pz.to_dict("records"):
    T = transforms.from_row(t)
    q = transforms.apply(transforms.invert(T), pt)[0]
    if 0 <= q[0] < t["width"] and 0 <= q[1] < t["height"]:
        x0, y0 = int(q[0]) - c // 2, int(q[1]) - c // 2
        raw = cache.read(t, slice(max(y0, 0), y0 + c), slice(max(x0, 0), x0 + c))
        print("tile", t["tile"], "raw values in the crop: min", int(raw.min()), "p1/p10/p50/p99", np.percentile(raw, [1, 10, 50, 99]).round(0), "| window", round(t["lo"]), round(t["hi"]))
        print("   raw values at exactly 0:", int((raw == 0).sum()), "| >= 65000:", int((raw >= 65000).sum()))
