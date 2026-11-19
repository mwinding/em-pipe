# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-13T20:49:17.812Z
# Ran: local
# Heredoc line: ${EM_PIPE_ENV}/bin/python - <<'EOF'
# Assistant note before it: Yes, fully raw would flicker more than the current render, and it would also show brightness steps at the tile seams. The raw tiles themselves change brightness from slice to slice, especially 0-1 and 1-0. I'll put numbers on it from the per-tile raw statistics I already have, simulating "fully raw"

import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import numpy as np, pandas as pd
from scipy.ndimage import median_filter
s = pd.read_csv(os.path.expandvars("${DEV_WORK}/stats.csv"), dtype={"tile": str})
lo, hi = np.percentile(s.p0_5, 1), np.percentile(s.p99_5, 99)        # one fixed window for everything
print(f"fixed raw window {lo:.0f}-{hi:.0f} (uint16) -> 0-255")
v = s.assign(v=(s.p50 - lo) / (hi - lo) * 255).pivot(index="z", columns="tile", values="v")
v4 = v.groupby(np.arange(len(v)) // 4).mean()                         # 32 nm planes = 4 slices
hp = v4 - v4.apply(lambda c: median_filter(c.values, 9, mode="nearest"))
own = hp.sub(hp.mean(axis=1), axis=0)
slow = v4.sub(v4.mean(axis=1), axis=0)
print("plane-to-plane tile flicker (grey levels): std", own.std().round(1).to_dict(), " max", own.abs().max().round(0).to_dict())
print("slow drift of each tile vs the others over the 2 days (p5-p95 range):", (slow.quantile(.95) - slow.quantile(.05)).round(0).to_dict())
print("tile median brightness (mean over z):", v4.mean().round(0).to_dict())
sp = s.assign(sp=(s.p99_5 - s.p0_5) / (hi - lo) * 255).groupby("tile").sp.median().round(0)
print("grey-level range each tile's tissue uses (p0.5-p99.5):", sp.to_dict())
