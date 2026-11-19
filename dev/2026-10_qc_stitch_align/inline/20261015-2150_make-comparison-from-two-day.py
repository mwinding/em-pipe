# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-15T21:50:25.203Z
# Ran: remote, via ssh
# Heredoc line: ssh -o BatchMode=yes -o ConnectTimeout=30 -o LogLevel=QUIET ${CLUSTER_LOGIN} 'source ${EM_PIPE_ROOT}/slurm/env.sh >/dev/null 2>&1; cd ${DEV_LAB_AREA}
# Assistant note before it: I'll make the comparison from the two-day test's 8 nm no-CLAHE volume, which is the normalised data CLAHE would be applied to. First I'll find suitable regions: one at the sample edge with a lot of black off-sample area, one at the edge with less, and one inside the tissue. I'll also check what grey

import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import numpy as np
from scipy import ndimage
from PIL import Image
from pipeline import omezarr
root = os.path.expandvars("${DEV_USER_AREA}/em-pipe_${DEV_PROJECT}_test-M10-D01-D02/${DEV_PROJECT}_M10-D01-D02_8nm.ome.zarr")
img = omezarr.open_scale(root, 4)[600 // 16].read().result()
print("s4", img.shape)
Image.fromarray(img).save(os.path.expandvars("${DEV_WORK}/plane600_s4.png"))
sm = ndimage.gaussian_filter(img.astype(float), 2)
sd = np.sqrt(np.maximum(ndimage.gaussian_filter(img.astype(float) ** 2, 4) - ndimage.gaussian_filter(img.astype(float), 4) ** 2, 0))
for thr in (225, 235, 245):
    print("bright smoothed >", thr, ":", round(float((sm > thr).mean()), 3), " of which low texture (sd<6):", round(float(((sm > thr) & (sd < 6)).mean()), 3))
print("local sd percentiles in tissue (sm<220):", np.percentile(sd[(sm < 220) & (img > 0)], [5, 25, 50]).round(1))
