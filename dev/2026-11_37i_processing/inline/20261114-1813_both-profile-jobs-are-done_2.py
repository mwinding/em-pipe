# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-11-14T18:13:56.578Z
# Ran: remote, via ssh
# Heredoc line: ssh -o LogLevel=ERROR -o ConnectTimeout=20 ${CLUSTER_LOGIN} '${EM_PIPE_ENV}/bin/python -' <<'PY'
# Assistant note before it: Both profile jobs are done, and the plain volume's zoom-out levels are finished. Merging the profile results:

import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import numpy as np, tensorstore as ts
S = os.path.expandvars("${DEV_37I}")
a = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": os.path.expandvars(f"{S}/${{DEV_PREFIX}}37i_8nm.ome.zarr/s0")}}, read=True, open=True).result()
def stats(z, y, x, label):
    p = np.asarray(a[z, y:y + 2048, x:x + 2048].read().result()).astype(np.float32)
    ms, ss, hi = [], [], []
    for i in range(0, 2048, 256):
        for j in range(0, 2048, 256):
            w = p[i:i + 256, j:j + 256]
            ms.append(w.mean()); ss.append(w.std()); hi.append((w > 225).mean())
    print(f"{label} z{z}: mean p10/50/90 {np.percentile(ms,[10,50,90]).round(0)}, std {np.percentile(ss,[10,50,90]).round(1)}, >225 {np.percentile(hi,[10,50,90]).round(2)}", flush=True)
stats(16900, 13300, 12800, "resin (gap)")
stats(16900, 13300, 20500, "tissue (right piece)")
stats(10000, 16000, 6000, "tissue (left, z10000)")
