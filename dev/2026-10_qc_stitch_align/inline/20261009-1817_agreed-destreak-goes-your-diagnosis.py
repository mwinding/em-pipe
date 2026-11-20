# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-09T18:17:14.960Z
# Ran: remote, via ssh
# Heredoc line: ssh -o BatchMode=yes ${CLUSTER_LOGIN} 'cat ${DEV_USER_AREA}/em-pipe_test-M10-D01-D02/work/preview/stats.csv' 2>/dev/null > ${DEV_WORK}
# Assistant note before it: Agreed, destreak goes. And your diagnosis matches what I measured earlier. The tile-specific fluctuation was largest on two tiles, 0-1 (1.9%) and 1-0 (2.3%), against about 1% on the other two. So it's in those tiles' raw data, not caused by stitching. The pipeline already normalises each tile stack 

import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import numpy as np, pandas as pd
from scipy.ndimage import median_filter
s = pd.read_csv(os.path.expandvars("${DEV_WORK}/stats.csv"), dtype={"tile": str})
s["spread"] = s.p99_5 - s.p0_5
rng = s.spread.median()
out = {}
for col in ("p50", "spread"):
    p = s.pivot(index="z", columns="tile", values=col)
    hp = p - p.apply(lambda c: median_filter(c.values, 51, mode="nearest"))   # slice-to-slice part vs 51-slice trend
    out[col] = (100 * hp.std() / rng).round(2)
print("slice-to-slice fluctuation per tile, % of the display range (p0.5-p99.5):")
print(pd.DataFrame(out).rename(columns={"p50": "brightness (median)", "spread": "contrast (p99.5-p0.5)"}).to_string())
p = s.pivot(index="z", columns="tile", values="spread")
print("\nlag-1 autocorrelation of the contrast fluctuation (near 0 = independent per slice):")
hp = p - p.apply(lambda c: median_filter(c.values, 51, mode="nearest"))
print({t: round(float(hp[t].autocorr(1)), 2) for t in hp.columns})
