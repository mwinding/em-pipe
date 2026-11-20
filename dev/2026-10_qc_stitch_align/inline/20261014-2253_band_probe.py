# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-14T22:53:54.300Z
# Ran: remote, via ssh
# Heredoc line: ssh -o BatchMode=yes -o ConnectTimeout=20 -o LogLevel=QUIET ${CLUSTER_LOGIN} 'cat > ${DEV_WORK}/band_probe.py <<EOF
# Assistant note before it: I've fixed it, and the 3×3 test is re-running with the fix. - **The fix:** with refinement on, each measured slice keeps its own tile positions instead of being smoothed towards its neighbours. All 211 tests pass, and it's pushed to main and pulled on the cluster. - **3×3 test:** stitching, alignment and i

"""Shift of each tile vs the previous slice, measured separately in horizontal bands (top -> bottom):
a jump during the scan shows as different shifts in the upper and lower bands."""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import sys
import cv2, numpy as np
sys.path.insert(0, os.path.expandvars("${EM_PIPE_ROOT}"))
from pipeline.config import load_config
from pipeline.slices import StackCache, load_slices
cfg = load_config(sys.argv[1]); z0, z1 = int(sys.argv[2]), int(sys.argv[3])
df = load_slices(cfg); df = df[(df.z >= z0) & (df.z <= z1)]
cache = StackCache(cfg["raw_dir"]); B, W = 1024, 2048
win = cv2.createHanningWindow((W, B), cv2.CV_32F)
prev = {}
for z, g in df.groupby("z"):
    out = [str(z)]
    for _, r in g.sort_values("tile").iterrows():
        h, w = int(r.height), int(r.width)
        bands = []
        for y0 in np.linspace(200, h - B - 200, 6).astype(int):
            img = cache.read(r, slice(y0, y0 + B), slice(w // 2 - W // 2, w // 2 + W // 2)).astype(np.float32)
            lo, hi = np.percentile(img, [1, 99]); bands.append(((img - lo) / (hi - lo)).astype(np.float32))
        if r.tile in prev:
            sh = [cv2.phaseCorrelate(a, b, win)[0] for a, b in zip(prev[r.tile], bands)]
            out.append(r.tile + " " + " ".join(f"({x:4.0f},{y:4.0f})" for x, y in sh))
        prev[r.tile] = bands
    print("  ".join(out), flush=True)
