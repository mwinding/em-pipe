# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-14T21:37:05.692Z
# Ran: remote, via ssh
# Heredoc line: ssh -o BatchMode=yes -o ConnectTimeout=20 -o LogLevel=QUIET ${CLUSTER_LOGIN} 'mkdir -p ${DEV_WORK} && cat > ${DEV_WORK}/stitch_range.py <<EOF
# Assistant note before it: I'll stitch slices 3366–3380 individually with the pipeline's own stitching function, as a short job on a compute node. Results print to the terminal, so nothing is written to your output folder.

"""Stitch a z range slice by slice (no output files) and compare with the interpolated tiles.csv."""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expandvars("${EM_PIPE_ROOT}"))
from pipeline.config import load_config
from pipeline import stitch
from pipeline.slices import StackCache
cfg = load_config(sys.argv[1], step_defaults=stitch.DEFAULTS)
z0, z1 = int(sys.argv[2]), int(sys.argv[3])
slices, _, gauge = stitch._load(cfg)
by_z = dict(tuple(slices.groupby("z")))
def one(z):
    rec = stitch.stitch_slice(by_z[z], StackCache(cfg["raw_dir"]), cfg["stitch"], gauge[z][1])
    if not rec["ok"]:
        return z, None, rec["reason"]
    T = {t: np.array(A)[:, 2] for t, A in rec["tiles"].items()}
    return z, {t: T[t] - T["0-0"] for t in T}, max((p["solve_residual_px"] for p in rec["pairs"] if p["used"]), default=0)
tl = pd.read_csv(stitch.step_dir(cfg, "stitch", "tiles.csv"), dtype={"tile": str})
with ProcessPoolExecutor(int(sys.argv[4])) as ex:
    for z, rel, extra in ex.map(one, range(z0, z1 + 1)):
        if rel is None:
            print(z, "FAILED", extra); continue
        g = tl[tl.z == z].set_index("tile")
        interp = {t: g.loc[t, ["tx", "ty"]].to_numpy() - g.loc["0-0", ["tx", "ty"]].to_numpy() for t in rel}
        print(z, "  ".join(f"{t}: ({rel[t][0]:8.1f},{rel[t][1]:8.1f}) diff ({rel[t][0]-interp[t][0]:6.1f},{rel[t][1]-interp[t][1]:6.1f})" for t in ["0-1", "1-0", "1-1"]), f" max res {extra:.1f}", flush=True)
