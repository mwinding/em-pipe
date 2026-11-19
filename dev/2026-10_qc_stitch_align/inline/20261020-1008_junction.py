# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-20T10:08:05.551Z
# Ran: remote, via ssh
# Heredoc line: cat > ${DEV_WORK}/cluster/junction.py <<'EOF'
# Assistant note before it: A false alarm on clipping. At full resolution the 37i raw data isn't clipped. The brightest pixels are isolated single values around 65,200, with nothing piling up at a ceiling. The apparent cut-off near 64,400 came from the preview statistics, which are measured on 1/16 thumbnails, where averaging 

"""37i: is the single-tile -> 2x2 change (z 1913 -> 1914) aligned? Shifts between consecutive planes in
windows over the area both layouts cover, by phase correlation at 8 nm; the pairs either side are the
baseline. Writes junction_z1913.png (z 1913 | z 1914 | overlay) for the central window."""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from pipeline import omezarr

O = os.path.expandvars("${DEV_37I}/")
root = O + os.path.expandvars("${DEV_PREFIX}37i_8nm_raw.ome.zarr")
s0, s3 = omezarr.open_scale(root, 0), omezarr.open_scale(root, 3)
J, W = 1913, 1024
m = (s3[J // 8].read().result() > 0) & (s3[(J + 1) // 8].read().result() > 0)
m = cv2.erode(m.astype(np.uint8), np.ones((40, 40), np.uint8)) > 0   # >= 2.5k px inside both
rr, cc = np.nonzero(m)
r0, r1, c0, c1 = rr.min(), rr.max(), cc.min(), cc.max()
print(f"common area at s3: rows {r0}-{r1}, cols {c0}-{c1} ({m.sum()} px)")
cent = [(int(8 * (r0 + (r1 - r0) * fr)), int(8 * (c0 + (c1 - c0) * fc))) for fr in (0.2, 0.5, 0.8) for fc in (0.2, 0.5, 0.8)]
cent = [(r, c) for r, c in cent if m[r // 8, c // 8]]
pairs = [(z, z + 1) for z in range(J - 3, J + 3)]
win = cv2.createHanningWindow((W, W), cv2.CV_32F)
res = {p: [] for p in pairs}
for r, c in cent:
    planes = {z: s0[z, r - W // 2:r + W // 2, c - W // 2:c + W // 2].read().result().astype(np.float32)
              for z in range(J - 3, J + 4)}
    for a, b in pairs:
        (dx, dy), resp = cv2.phaseCorrelate(planes[a], planes[b], win)
        res[(a, b)].append((dx, dy, resp))
for (a, b), v in res.items():
    v = np.array(v)
    d = np.hypot(v[:, 0], v[:, 1])
    tag = "  <- layout change" if a == J else ""
    print(f"z {a}->{b}: shift median {np.median(d):.1f} px, max {d.max():.1f} px "
          f"(dx {np.median(v[:, 0]):+.1f}, dy {np.median(v[:, 1]):+.1f}; response {np.median(v[:, 2]):.2f}){tag}")
# picture: central window
r, c = cent[len(cent) // 2]
a = s0[J, r - W // 2:r + W // 2, c - W // 2:c + W // 2].read().result()
b = s0[J + 1, r - W // 2:r + W // 2, c - W // 2:c + W // 2].read().result()
ov = np.dstack([b, a, b])          # z 1913 green, z 1914 magenta: grey where they agree
def lab(img, t):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
    top = np.full((40, img.shape[1], 3), 255, np.uint8)
    cv2.putText(top, t, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return np.concatenate([top, img])
gap = np.full((W + 40, 16, 3), 255, np.uint8)
out = np.concatenate([lab(a, f"z {J} (single tile)"), gap, lab(b, f"z {J + 1} (2x2)"), gap,
                      lab(ov, "overlay: green z1913, magenta z1914")], axis=1)
cv2.imwrite(O + "qc/junction_z1913.png", out)
print(f"picture: rows {r - W // 2}-{r + W // 2}, cols {c - W // 2}-{c + W // 2} -> qc/junction_z1913.png")
