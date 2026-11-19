# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-09T15:38:16.756Z
# Ran: local
# Heredoc line: ${EM_PIPE_ENV}/bin/python - <<'EOF'
# Assistant note before it: While that runs, I'll look at how the intensity step sets the levels, to plan the per-slice correction:

import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import numpy as np, cv2
from scipy.ndimage import gaussian_filter1d
S = os.path.expandvars("${DEV_WORK}/destreak")
names = ["original", "sigma 1", "sigma 2", "sigma 4", "sigma 8"]
for f in ("resin_edge_z8530_tile0-1", "tissue_z8530_tile0-0"):
    im = cv2.imread(f"{S}/{f}.png", 0).astype(float)
    panels = [im[100:, i * 1036:i * 1036 + 1024] for i in range(5)]   # skip the label rows
    orig = panels[0]
    out = []
    for n, p in zip(names, panels):
        prof = p.mean(axis=0)
        stripes = np.std(prof - gaussian_filter1d(prof, 20))          # column-to-column stripe amplitude
        removed = p - orig
        out.append(f"{n}: stripes {stripes:.2f}  change vs original rms {np.sqrt(np.mean(removed ** 2)):.2f}")
    print(f, "\n   " + "\n   ".join(out))
