# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-15T21:49:56.367Z
# Ran: remote, via ssh
# Heredoc line: ssh -o BatchMode=yes -o ConnectTimeout=30 -o LogLevel=QUIET ${CLUSTER_LOGIN} 'source ${EM_PIPE_ROOT}/slurm/env.sh >/dev/null 2>&1; cd ${DEV_LAB_AREA}
# Assistant note before it: I'll make the comparison from the two-day test's 8 nm no-CLAHE volume, which is the normalised data CLAHE would be applied to. First I'll find suitable regions: one at the sample edge with a lot of black off-sample area, one at the edge with less, and one inside the tissue. I'll also check what grey

import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import numpy as np
from scipy import ndimage
from pipeline import omezarr
root = os.path.expandvars("${DEV_USER_AREA}/em-pipe_${DEV_PROJECT}_test-M10-D01-D02/${DEV_PROJECT}_M10-D01-D02_8nm.ome.zarr")
s = 3
a = omezarr.open_scale(root, s)
print("s3 shape", a.shape)
img = a[600 // 8].read().result()
v = img[img > 0]
print("nonzero fraction", round(v.size / img.size, 3), "percentiles 0.5,1,5,25,50,95:", np.percentile(v, [0.5, 1, 5, 25, 50, 95]).round(1))
h = np.bincount(v.ravel(), minlength=256)
print("histogram (bins of 8):", [int(x) for x in h.reshape(32, 8).sum(1)])
sm = ndimage.gaussian_filter(img.astype(float), 2)
for thr in (20, 30, 40, 50):
    print("smoothed < ", thr, " fraction of nonzero:", round(float(((sm < thr) & (img > 0)).sum() / v.size), 3))
# 2048 px blocks at 8 nm = 256 px at s3: dark fraction per block
B = 256
H, W = img.shape
for thr in (40,):
    dark = (sm < thr) & (img > 0)
    out = []
    for by in range(0, H - B, B // 2):
        for bx in range(0, W - B, B // 2):
            blk = img[by:by + B, bx:bx + B]
            if (blk == 0).mean() > 0.01: continue
            out.append((round(float(dark[by:by + B, bx:bx + B].mean()), 2), by * 8, bx * 8))
    out.sort()
    print("interior (least dark):", out[:3])
    mid = [o for o in out if 0.3 <= o[0] <= 0.5]
    print("edge (30-50% dark):", mid[:5])
    lo = [o for o in out if 0.1 <= o[0] <= 0.2]
    print("edge (10-20% dark):", lo[:5])
