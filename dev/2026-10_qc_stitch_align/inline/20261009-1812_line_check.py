# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-09T18:12:09.469Z
# Ran: remote, via ssh
# Heredoc line: ssh -o BatchMode=yes ${CLUSTER_LOGIN} 'cat > ${DEV_WORK}/line_check.py' 2>/dev/null <<'EOF'
# Assistant note before it: That bright, perfectly straight vertical line is the giveaway. It's not tissue, and I don't think it was there before. I'll find out where it comes from rather than guess. I'll compare the column brightness of the new TIFF (with destreak) against the `_before` TIFF (without) across all planes, locat

"""Find straight lines added by the re-render: column/row profiles of new vs before 64 nm TIFFs."""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import json, sys
import numpy as np, tifffile, cv2
from scipy.ndimage import median_filter
T = sys.argv[1]
new = tifffile.memmap(os.path.expandvars(f"{T}/${{DEV_PROJECT}}_M10-D01-D02_64nm.tif"))
old = tifffile.memmap(os.path.expandvars(f"{T}/${{DEV_PROJECT}}_M10-D01-D02_64nm_before.tif"))
print("shapes", new.shape, old.shape)
def lines(v, axis):
    prof = v.astype(np.float32).mean(axis=(0, axis))          # mean over z and the other axis
    return prof - median_filter(prof, 9, mode="nearest")
for name, axis in (("columns (vertical lines)", 1), ("rows (horizontal lines)", 2)):
    ln, lo = lines(new, axis), lines(old, axis)
    idx = np.argsort(-np.abs(ln))[:6]
    print(name, "new top:", [(int(i), round(float(ln[i]), 1), "before:", round(float(lo[i]), 1)) for i in idx])
# render geometry -> tile columns: output px (64 nm) * 8 + origin = aligned full-res px
meta = json.load(open(os.path.expandvars(f"{T}/work/render/${{DEV_PROJECT}}_M10-D01-D02/render.json")))
print("origin", meta["origin_xy"], "f", meta["downsample"])
i = int(np.argmax(np.abs(lines(new, 1))))
x_full = meta["origin_xy"][0] + i * 2 * meta["downsample"]
print("strongest vertical line at 64 nm column", i, "-> aligned full-res x ~", x_full)
z = new.shape[0] // 2
y0, y1 = 0, new.shape[1]
crop_n = new[z, :, max(0, i - 150):i + 150]
crop_o = old[z, :, max(0, i - 150):i + 150]
cv2.imwrite(f"{sys.argv[2]}/line_new.png", crop_n)
cv2.imwrite(f"{sys.argv[2]}/line_before.png", crop_o)
