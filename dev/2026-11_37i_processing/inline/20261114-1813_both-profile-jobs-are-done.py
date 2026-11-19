# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-11-14T18:13:22.499Z
# Ran: remote, via ssh
# Heredoc line: ssh -o LogLevel=ERROR -o ConnectTimeout=20 ${CLUSTER_LOGIN} '${EM_PIPE_ENV}/bin/python -' <<'PY'
# Assistant note before it: Both profile jobs are done, and the plain volume's zoom-out levels are finished. Merging the profile results:

import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import numpy as np, tensorstore as ts, cv2
S = os.path.expandvars("${DEV_37I}")
OUT_ROOT = __import__('os').environ.get('DEV_OUT', 'dev_out')  # outputs: a directory you choose, never the dataset folder
a = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": os.path.expandvars(f"{S}/${{DEV_PREFIX}}37i_8nm.ome.zarr/s3")}}, read=True, open=True).result()
print(a.shape)
tiles = []
for z in (10000, 15500, 16400, 16900, 17500):
    p = np.asarray(a[z // 8].read().result())
    p = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(p, (9728 // 8, 11264 // 8), (17920 // 8, 17408 // 8), (0, 0, 255), 6)
    cv2.putText(p, f"z {z}", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 8)
    tiles.append(cv2.resize(p, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA))
cv2.imwrite(f"{OUT_ROOT}/qc/zstep_profile/region_check.png", np.hstack(tiles))
