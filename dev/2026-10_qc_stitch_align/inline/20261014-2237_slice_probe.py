# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-14T22:37:51.868Z
# Ran: remote, via ssh
# Heredoc line: ssh -o BatchMode=yes -o ConnectTimeout=20 -o LogLevel=QUIET ${CLUSTER_LOGIN} 'cat > ${DEV_WORK}/slice_probe.py <<EOF
# Assistant note before it: The reader can read just a window of a slice, so this stays light. Writing the probe and running it as a short batch job:

"""Per tile, for a z range: sharpness (band-pass contrast) and shift vs the previous slice, from raw
tile centres; plus a PNG of the centres of a few slices. Writes nothing in the pipeline folders."""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import sys
import cv2, numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.expandvars("${EM_PIPE_ROOT}"))
from pipeline.config import load_config
from pipeline.slices import StackCache, load_slices
cfg = load_config(sys.argv[1]); z0, z1 = int(sys.argv[2]), int(sys.argv[3]); show = [int(v) for v in sys.argv[4].split(",")]
df = load_slices(cfg); df = df[(df.z >= z0) & (df.z <= z1)]
cache = StackCache(cfg["raw_dir"]); C = 2048
prev, rows, crops = {}, [], {}
win = cv2.createHanningWindow((C, C), cv2.CV_32F)
for z, g in df.groupby("z"):
    line = [f"{z}"]
    for _, r in g.sort_values("tile").iterrows():
        h, w = int(r.height), int(r.width)
        img = cache.read(r, slice(h // 2 - C // 2, h // 2 + C // 2), slice(w // 2 - C // 2, w // 2 + C // 2)).astype(np.float32)
        lo, hi = np.percentile(img, [1, 99]); n = (img - lo) / (hi - lo)
        band = ndimage.gaussian_filter(n, 1.5) - ndimage.gaussian_filter(n, 4)
        sharp = band.std() / ndimage.gaussian_filter(n, 4).std()
        sh, resp = ((0.0, 0.0), 0.0) if r.tile not in prev else cv2.phaseCorrelate(prev[r.tile], n, win)
        prev[r.tile] = n
        line.append(f"{r.tile}: sharp {sharp:.3f} shift ({sh[0]:6.1f},{sh[1]:6.1f}) peak {resp:.2f}")
        if z in show:
            crops[(z, r.tile)] = (np.clip(n[C//2-512:C//2+512, C//2-512:C//2+512], 0, 1) * 255).astype(np.uint8)
    print("  ".join(line), flush=True)
tiles = sorted({t for _, t in crops})
grid = np.vstack([np.hstack([crops.get((z, t), np.zeros((1024, 1024), np.uint8)) for t in tiles]) for z in show])
cv2.imwrite(sys.argv[5], cv2.resize(grid, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
print("png:", sys.argv[5], "rows = z", show, "cols = tiles", tiles)
