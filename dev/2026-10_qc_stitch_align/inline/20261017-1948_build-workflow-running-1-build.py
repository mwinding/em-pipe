# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-17T19:48:10.408Z
# Ran: remote, via ssh
# Heredoc line: ssh -o BatchMode=yes -o ConnectTimeout=30 -o LogLevel=QUIET ${CLUSTER_LOGIN} 'cd ${EM_PIPE_ROOT} && ${EM_PIPE_ENV}/bin/pyt
# Assistant note before it: The build workflow is running: 1. **Build**, in parallel: the patch-grid correction into the pipeline, and the fix to the seam-measurement bug. 2. **Review:** three independent reviewers. - **Maths:** re-runs the new code on the experiment's real overlap measurements from the cluster and checks it reproduc

import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import sys, json
sys.path.insert(0, ".")
import numpy as np
from pipeline import transforms
O = os.path.expandvars("${DEV_35H}")
stitch = transforms.read_csv(f"{O}/work/stitch/tiles.csv", ["z", "tile"])
align = transforms.read_csv(f"{O}/work/align/transforms.csv", ["z"])
import pandas as pd
sl = pd.read_csv(f"{O}/work/check/slices.csv", dtype={"tile": str}, usecols=["z", "tile", "height", "width"])
meta = json.load(open(os.path.expandvars(f"{O}/work/render/${{DEV_PREFIX}}35h/render.json")))
print("old volume origin_xy", meta["origin_xy"], "canvas", meta["canvas_size_xy"])
for z, tiles in ((1000, ["1-0", "1-1", "2-0", "2-1"]), (6000, ["0-0", "0-1", "1-0", "1-1"])):
    pts = []
    for t in tiles:
        r = sl[(sl.z == z) & (sl.tile == t)].iloc[0]
        T = transforms.compose(align[z], stitch[(z, t)])
        c = transforms.apply(T, [[0, 0], [r.width - 1, 0], [0, r.height - 1], [r.width - 1, r.height - 1]])
        pts.append(c)
    # the junction: the inner corner of each tile (the corner closest to the mean of all four tiles centres)
    centre = np.mean([p.mean(axis=0) for p in pts], axis=0)
    inner = [p[np.argmin(np.linalg.norm(p - centre, axis=1))] for p in pts]
    j = np.mean(inner, axis=0)
    print("z", z, "junction (aligned px)", j.round(0).tolist(), "inner corners", np.round(inner, 0).tolist())
