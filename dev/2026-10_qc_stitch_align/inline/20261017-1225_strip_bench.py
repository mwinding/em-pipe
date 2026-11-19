# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-17T12:25:38.290Z
# Ran: remote, via ssh
# Heredoc line: cat > ${DEV_WORK}/strip_bench.py <<'EOF'
# Assistant note before it: Narrow column strips are read one row at a time, which may be slow on the cluster filesystem. I'll time it on the cluster before designing around it.

"""Time reading overlap strips (column and row strips) vs whole slices from raw 35h tiles."""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import sys, time
sys.path.insert(0, os.path.expandvars("${EM_PIPE_ROOT}"))
import pandas as pd
from pipeline.slices import StackCache
from pipeline.config import load_config
from pipeline.slices import load_slices
cfg = load_config(os.path.expandvars("${EM_PIPE_ROOT}/configs/${DEV_PREFIX}35h.yaml"))
sl = load_slices(cfg)
cache = StackCache(cfg["raw_dir"])
for z in (6000, 6001, 9000, 16000):
    rows = sl[sl.z == z].to_dict("records")
    r = rows[0]
    h, w = int(r["height"]), int(r["width"])
    t = time.time(); a = cache.read(r, slice(0, h), slice(w - 200, w)); tc = time.time() - t
    t = time.time(); b = cache.read(r, slice(h - 200, h), slice(0, w)); tr = time.time() - t
    t = time.time(); c = cache.read(rows[1]); tf = time.time() - t
    print(f"z {z} tile {r['tile']} {w}x{h}: column strip 200 px {tc:.2f} s, row strip 200 px {tr:.2f} s, whole slice {tf:.2f} s", flush=True)
