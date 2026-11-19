# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-08T23:28:13.129Z
# Ran: local
# Heredoc line: R=${DEV_WORK}/realsize; cd ${EM_PIPE_ROOT} && ${EM_PIPE_ENV}/bin/python - <<EOF
# Assistant note before it: Preview's peak memory dropped from 12.7 GB to 1.7 GB, and the run was twice as fast. The remaining memory-mapped reads are small thumbnails. Now align, intensity, render and pyramid at real size:

import numpy as np, cv2
from pipeline import omezarr
a = omezarr.open_scale("$R/out/render/volume.ome.zarr", 0)[0].read().result()
h, w = a.shape
cy, cx = h // 2, w // 2      # tiles meet near the centre
crop = a[cy - 300:cy + 300, cx - 300:cx + 300]
cv2.imwrite("$R/render_centre_crop.png", crop)
cv2.imwrite("$R/render_whole.png", cv2.resize(a, (w // 6, h // 6), interpolation=cv2.INTER_AREA))
# brightness step across the vertical and horizontal seams (column/row mean profiles)
col = a[cy - 2000:cy + 2000, :].astype(float).mean(0); row = a[:, cx - 2000:cx + 2000].astype(float).mean(1)
print("col-mean around x seam:", col[cx - 6:cx + 6].round(1)); print("row-mean around y seam:", row[cy - 6:cy + 6].round(1))
