# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-16T08:22:44.020Z
# Ran: local
# Heredoc line: cd ${DEV_WORK} && ${EM_PIPE_ENV}/bin/python - <<'EOF'
# Assistant note before it: B was built: tissue statistics at 23:20, intensity at 23:25, then the render. Checking from its logs that the render and pyramid completed:

import pandas as pd, numpy as np
A = pd.read_csv("levels_A.csv", dtype={"tile": str}); B = pd.read_csv("levels_B.csv", dtype={"tile": str})
S = pd.read_csv("stats_B.csv", dtype={"tile": str})[["z", "tile", "tissue_fraction", "timestamp"]]
d = A.merge(B, on=["z", "tile"], suffixes=("_A", "_B")).merge(S, on=["z", "tile"])
d["wA"], d["wB"] = d.hi_A - d.lo_A, d.hi_B - d.lo_B
# how a tissue grey value maps: difference in output (0-255) at the tile's mid-grey, and contrast ratio
d["contrast"] = d.wA / d.wB
mid = (d.lo_B + d.hi_B) / 2
d["shift"] = 255 * ((mid - d.lo_A) / d.wA - (mid - d.lo_B) / d.wB)
bg = d.tissue_fraction < 0.95
print(f"{len(d)} tile slices; {bg.mean():.0%} have >5% background (masked)")
print(f"contrast B/A: all median {d.contrast.median():.3f}; background tiles median {d.contrast[bg].median():.3f}, p90 {d.contrast[bg].quantile(.9):.3f}, max {d.contrast.max():.2f}")
print(f"grey shift at mid-tissue (0-255): background tiles median {d['shift'][bg].median():+.1f}, p10 {d['shift'][bg].quantile(.1):+.1f}, p90 {d['shift'][bg].quantile(.9):+.1f}")
print(f"tissue-only tiles: contrast median {d.contrast[~bg].median():.3f}, shift median {d['shift'][~bg].median():+.1f}")
d["day"] = d.timestamp.str[:10]
g = d[bg].groupby(["day", "tile"]).agg(n=("z", "size"), tissue=("tissue_fraction", "median"), contrast=("contrast", "median"), shift=("shift", "median"))
print("\nwhere it differs most (days/tiles with >5% background, median per day):")
print(g[g.n > 50].sort_values("contrast", ascending=False).head(15).round(3).to_string())
print("\nat the 3x3->2x2 change (z 3270-3290):")
print(d[(d.z >= 3274) & (d.z <= 3283)].groupby("z").agg(tissue=("tissue_fraction", "median"), contrast=("contrast", "median"), shift=("shift", "median")).round(3).to_string())
