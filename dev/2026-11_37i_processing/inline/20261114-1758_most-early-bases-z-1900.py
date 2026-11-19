# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-11-14T17:58:29.382Z
# Ran: remote, via ssh
# Heredoc line: ssh -o LogLevel=ERROR -o ConnectTimeout=20 ${CLUSTER_LOGIN} '${EM_PIPE_ENV}/bin/python -' <<'PY'
# Assistant note before it: Most early bases (z < 1900) have too few tissue windows in the central region. Checking why:

import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import numpy as np, tensorstore as ts, json
S = os.path.expandvars("${DEV_37I}")
for v in [os.path.expandvars("${DEV_PREFIX}37i_test_z1000_8nm.ome.zarr"), os.path.expandvars("${DEV_PREFIX}37i_test_z16500_8nm.ome.zarr")]:
    a = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": f"{S}/{v}/s4"}}, read=True, open=True).result()
    p = np.asarray(a[32].read().result())
    ys, xs = np.nonzero(p)
    tis = (p > 0) & (p < 200)
    ty, tx = np.nonzero(tis)
    print(v, "s4 shape", p.shape, "data bbox s0 y", ys.min()*16, ys.max()*16, "x", xs.min()*16, xs.max()*16,
          "| tissue(<200) centroid s0 y", int(ty.mean()*16), "x", int(tx.mean()*16), "p10-90 y", int(np.percentile(ty,10)*16), int(np.percentile(ty,90)*16), "x", int(np.percentile(tx,10)*16), int(np.percentile(tx,90)*16))
    s0 = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": f"{S}/{v}/s0"}}, read=True, open=True).result()
    print("  s0 shape", s0.shape)
