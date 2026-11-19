# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-09T15:29:09.463Z
# Ran: local
# Heredoc line: T=${DEV_USER_AREA}/em-pipe_test-M10-D01-D02 && ${EM_PIPE_ENV}/bin/python - <<EOF
# Assistant note before it: That depends on what kind of flicker it is. If whole tiles get brighter or darker from one slice to the next, it's a real (fixable) shortcoming. If tiles shift position relative to each other, that would be a problem for tracing. Let me measure both rather than guess. First brightness: I'll take a p

import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import numpy as np, pandas as pd
from scipy.ndimage import median_filter
from pipeline import omezarr
v = omezarr.open_scale(os.path.expandvars("$T/${DEV_PROJECT}_M10-D01-D02.ome.zarr"), 0)
centres = {"0-0": (1594, 1734), "0-1": (1594, 5051), "1-0": (4650, 1957), "1-1": (4638, 5260)}   # (y, x) at 32 nm
m = {}
for t, (y, x) in centres.items():
    m[t] = v[:, y - 256:y + 256, x - 256:x + 256].read().result().reshape(v.shape[0], -1).mean(1)
M = pd.DataFrame(m)
hp = M - M.apply(lambda c: median_filter(c.values, 9, mode="nearest"))      # plane-to-plane part
common = hp.mean(axis=1)
own = hp.sub(common, axis=0)                                                 # tile-specific part
print("RENDERED (grey levels of 255), per tile mean over planes:", M.mean().round(1).to_dict())
print("  plane-to-plane fluctuation, all tiles together: std %.2f" % common.std())
print("  plane-to-plane fluctuation, tile-specific:      std", own.std().round(2).to_dict(), " max |.|", own.abs().max().round(1).to_dict())
s = pd.read_csv("$T/work/preview/stats.csv", dtype={"tile": str})
p = s.pivot(index="z", columns="tile", values="p50")
rng = (s.p99_5 - s.p0_5).median()
hp = p - p.apply(lambda c: median_filter(c.values, 9, mode="nearest"))
own = hp.sub(hp.mean(axis=1), axis=0)
print("RAW p50 (as % of the p0.5-p99.5 range %.0f):" % rng)
print("  slice-to-slice, tile-specific std %:", (100 * own.std() / rng).round(2).to_dict(), " max %:", (100 * own.abs().max() / rng).round(1).to_dict())
lv = pd.read_csv("$T/work/intensity/levels.csv", dtype={"tile": str})
print("levels: lo/hi change between consecutive z (median abs):", lv.sort_values("z").groupby("tile")[["lo", "hi"]].diff().abs().median().round(1).to_dict())
