# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-09T11:17:16.453Z
# Ran: local
# Heredoc line: O=${DEV_USER_AREA}/em-pipe_test-M10-D01-D02/output; S=${DEV_WORK}/real_rend
# Assistant note before it: The two-day test is running again from stitch, now with Janelia's montage model. Here's what came out of this round: - **Your tiles aren't plain shifted copies.** They differ by small rotations and scales, which is why translation-only stitching failed. - **The model:** Janelia's affine pulled towar

import numpy as np, cv2
from pipeline import omezarr
v = omezarr.open_scale("$O/render/volume.ome.zarr", 0)
print("shape", v.shape)
cy, cx = 3120, 3400
for k in (0, 151, 166, 302):
    crop = v[k, cy - 400:cy + 400, cx - 400:cx + 400].read().result()
    cv2.imwrite(f"$S/junction_plane{k:03d}.png", crop)
    print("plane", k, "crop mean", crop.mean().round(1), "zeros", (crop == 0).mean().round(3))
whole = v[151].read().result()
cv2.imwrite("$S/plane151_whole.png", cv2.resize(whole, (whole.shape[1] // 5, whole.shape[0] // 5), interpolation=cv2.INTER_AREA))
xz = v[:, cy, :].read().result()          # 303 x 7005: z down, x across, through the horizontal seam region
cv2.imwrite("$S/xz_y3120.png", cv2.resize(xz, (xz.shape[1] // 2, xz.shape[0] * 2), interpolation=cv2.INTER_AREA))
yz = v[:, :, 2000].read().result()        # 303 x 6257
cv2.imwrite("$S/yz_x2000.png", cv2.resize(yz, (yz.shape[1] // 2, yz.shape[0] * 2), interpolation=cv2.INTER_AREA))
print("done")
