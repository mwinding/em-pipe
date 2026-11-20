# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-20T09:28:28.635Z
# Ran: remote, via ssh
# Heredoc line: cat > ${DEV_WORK}/cluster/raw_top.py <<'EOF'
# Assistant note before it: Both volumes go into one new folder on the lab share, next to 35h's: `${DEV_37I}/` (on the Mac: `${DEV_37I}/`) When finished, it will contain: - `EM05

"""Top end of the raw histograms: is 37i's bright tail clipped? (37i tiles vs one 35h tile)"""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from pipeline.imagej_tiff import ImageJStack

R37 = os.path.expandvars("${DEV_RAW_37I}/")
R35 = os.path.expandvars("${DEV_RAW_35H}/")
sl = pd.read_csv(os.path.expandvars("${DEV_37I}/work/check/slices.csv"),
                 dtype={"tile": str, "file": str})
picks = [(R37, r.file, int(r["index"]), f"37i z {r.z} tile {r.tile}")
         for _, r in pd.concat([sl[sl.z == z].head(1) for z in (100, 1000, 1900)] + [sl[(sl.z == 2050)]]).iterrows()]
s35 = pd.read_csv(os.path.expandvars("${DEV_35H}/work/check/slices.csv"),
                  dtype={"tile": str, "file": str})
r = s35[s35.z == 8936].iloc[0]
picks.append((R35, r.file, int(r["index"]), f"35h z 8936 tile {r.tile}"))
for root, f, i, name in picks:
    a = ImageJStack(root + f).read(i).ravel()
    mx = int(a.max())
    vals, counts = np.unique(a[a >= mx - 40], return_counts=True)
    top = sorted(zip(counts, vals), reverse=True)[:3]
    p = np.percentile(a, [0.5, 50, 99, 99.9])
    print(f"{name}: dtype {a.dtype}, min {a.min()}, p0.5 {p[0]:.0f}, p50 {p[1]:.0f}, p99 {p[2]:.0f}, p99.9 {p[3]:.0f}, "
          f"max {mx}; at max: {100 * (a == mx).mean():.3f} %, within 10 of max: {100 * (a >= mx - 10).mean():.3f} %; "
          f"commonest values near max: {[(int(v), int(c)) for c, v in top]}", flush=True)
