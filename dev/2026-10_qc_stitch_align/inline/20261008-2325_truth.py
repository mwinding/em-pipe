# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-08T23:25:04.971Z
# Ran: local
# Heredoc line: R=${DEV_WORK}/realsize; cat $R/out/stitch/layout.json | head -40; ${EM_PIPE_ENV}/bin/python -
# Assistant note before it: SSH to the cluster has just stopped resolving the hostname, so the Mac may have lost the institute network or VPN. I'm checking connectivity:

import json, pandas as pd, numpy as np
t = json.load(open("$R/truth.json"))
df = pd.read_csv("$R/out/stitch/tiles.csv", dtype={"tile": str})
o = {k: np.array(v, float) for k, v in t["tile_origin"].items()}
errs = []
for z, g in df.groupby("z"):
    est = {r.tile: np.array([r.tx, r.ty]) for r in g.itertuples()}
    ref = "0-0"
    for tile in est:
        errs.append(np.abs((est[tile] - est[ref]) - (o[tile] - o[ref])).max())
print("stitch max abs error vs truth (px):", max(errs))
print(pd.read_csv("$R/out/stitch/pairs.csv").head(8).to_string())
