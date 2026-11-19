# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-11-14T18:25:15.281Z
# Ran: remote, via ssh
# Heredoc line: ssh -o LogLevel=ERROR -o ConnectTimeout=20 ${CLUSTER_LOGIN} '${EM_PIPE_ENV}/bin/python -' <<'PY'
# Assistant note before it: Both final volumes are finished: `37i_8nm.ome.zarr` and `37i_8nm_clahe.ome.zarr`. Each has full resolution plus 6 zoom-out levels, and all render and zoom-out jobs completed without errors. To view them, run this on the Mac, then open the neuroglancer link it prints in Chro

import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import pandas as pd, numpy as np
S = os.path.expandvars("${DEV_37I}")
OUT_ROOT = __import__('os').environ.get('DEV_OUT', 'dev_out')  # outputs: a directory you choose, never the dataset folder
sc = pd.read_csv(f"{S}/work/align/segment_changes.csv")
print(sc.to_string()[:2500])
tab = pd.read_csv(f"{OUT_ROOT}/qc/zstep_profile/profile.csv")
d = pd.read_csv(f"{S}/work/check/slices.csv")
seg = d.groupby("z").segment.first()
print(seg.diff().pipe(lambda s: s[s != 0]).index.tolist())
for zc in [1914, 2083, 4476, 5559, 8316, 11014, 14631, 15049, 17333]:
    b = tab[(tab.z >= zc - 320) & (tab.z < zc)]
    a1 = tab[(tab.z >= zc) & (tab.z < zc + 64)]
    a2 = tab[(tab.z >= zc + 64) & (tab.z < zc + 384)]
    a3 = tab[(tab.z >= zc + 1000) & (tab.z < zc + 1320)]
    f = lambda t, c: f"{t[c].median():.3f}" if len(t) else "-"
    print(f"z {zc}: k4 before {f(b,'k4')} | first 64 {f(a1,'k4')} | +64..384 {f(a2,'k4')} | +1000..1320 {f(a3,'k4')}   k1 {f(b,'k1')} {f(a1,'k1')} {f(a2,'k1')} {f(a3,'k1')}")
