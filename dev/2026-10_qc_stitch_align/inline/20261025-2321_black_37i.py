# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-25T23:21:02.008Z
# Ran: remote, via ssh
# Heredoc line: cat > ${DEV_WORK}/cluster/black_37i.py <<'EOF'
# Assistant note before it: That's worth checking: the first 1,914 planes are the single-tile approach, which is roughly the first 60% of the stack. There are two possibilities: - **Expected:** the single tile images a smaller field than the 2×2 grid, so in parts of the 2×2 area the early planes have no data. - **A real proble

"""Why are 37i's single-tile planes black? One tile slice through render's own correction code."""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import sys
sys.path.insert(0, os.path.expandvars("${EM_PIPE_ROOT}"))
import json
import numpy as np
import pandas as pd
from pipeline import render, shading, patchgrid
from pipeline.config import load_config, deep_merge
from pipeline.slices import StackCache
cfg = load_config(os.path.expandvars("${EM_PIPE_ROOT}/configs/lab/${DEV_PREFIX}37i.yaml"),
                  step_defaults=deep_merge(deep_merge(render.DEFAULTS, shading.DEFAULTS), patchgrid.DEFAULTS))
root, wdir = render.volume_paths(cfg)
meta = json.loads((wdir / "render.json").read_text())
plan = pd.read_csv(wdir / "tiles.csv", dtype={"tile": str, "file": str})
r = render._Renderer(cfg, meta, plan, None)
for z in (1000, 2500):
    t = plan[plan.z == z].iloc[0].to_dict()
    h, w = int(t["height"]), int(t["width"])
    rows, cols = slice(h // 2 - 256, h // 2 + 256), slice(w // 2 - 256, w // 2 + 256)
    img = r.cache.read(t, rows, cols)
    key = (r.segment.get(t["z"]), t["tile"])
    fields = r._fields(key, r.shading.get(key), (h, w))
    u = patchgrid.apply_fields(img, fields, (h, w), rows, cols)
    g, o = fields
    print(f"z {z} tile {t['tile']} segment {key[0]}: raw p1/p50/p99 {np.percentile(img, [1, 50, 99]).round().tolist()}; "
          f"corrected {np.percentile(u, [1, 50, 99]).round().tolist()}; window {t['lo']:.0f}-{t['hi']:.0f}; "
          f"gain field {np.min(g):.4f}-{np.max(g):.4f}, offset field {np.min(o):.0f} to {np.max(o):.0f}; R {r.R:.0f}; "
          f"shape {'yes' if key in r.shapes else 'identity'}", flush=True)
