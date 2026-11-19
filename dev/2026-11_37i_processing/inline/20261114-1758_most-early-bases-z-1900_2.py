# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-11-14T17:58:39.867Z
# Ran: remote, via ssh
# Heredoc line: ssh -o LogLevel=ERROR -o ConnectTimeout=20 ${CLUSTER_LOGIN} '${EM_PIPE_ENV}/bin/python -' <<'PY'
# Assistant note before it: Most early bases (z < 1900) have too few tissue windows in the central region. Checking why:

import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import numpy as np, tensorstore as ts
S = os.path.expandvars("${DEV_37I}")
a = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": os.path.expandvars(f"{S}/${{DEV_PREFIX}}37i_8nm.ome.zarr/s0")}}, read=True, open=True).result()
st = 512
for z in (500, 1500, 2100, 9600, 16500):
    p = np.asarray(a[z, ::st, ::st].read().result())
    ys, xs = np.nonzero(p)
    t = (p > 0) & (p < 200)
    ty, tx = np.nonzero(t)
    print(z, "data y", ys.min()*st, ys.max()*st, "x", xs.min()*st, xs.max()*st, "| tissue samples", t.sum(),
          "y p10-90", int(np.percentile(ty, 10))*st, int(np.percentile(ty, 90))*st, "x p10-90", int(np.percentile(tx, 10))*st, int(np.percentile(tx, 90))*st, flush=True)
