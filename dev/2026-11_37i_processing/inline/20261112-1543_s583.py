# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-11-12T15:43:47.237Z
# Ran: remote, via ssh
# Heredoc line: cat > ${DEV_WORK}/s583.py <<'EOF'
# Assistant note before it: I'll look at how check detects duplicate content first, then check the re-exported file and the Backscatter files on the cluster.

import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from pipeline.imagej_tiff import ImageJStack
RAW = Path(os.path.expandvars("${DEV_RAW_37I}"))
rows = slice(0, None, 50)
def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
for t in ["0-0", "0-1", "1-0", "1-1"]:
    n = ImageJStack(RAW / f"M11_D10_tile{t}.tif")
    e = ImageJStack(RAW / "Backscatter" / f"M11_D10_EsB_tile{t}.tif")
    s = {i: n.read(i, rows=rows).astype(np.float64) for i in (581, 582, 583, 584)}
    eb = e.read(583, rows=rows).astype(np.float64)
    same = [float((s[583] == s[i]).all(axis=1).mean()) for i in (582, 584)]
    top, bot = s[583][: len(s[583]) // 2], s[583][len(s[583]) // 2:]
    print(f"tile {t}: identical rows vs 582/584 {same[0]:.0%}/{same[1]:.0%} | corr InLens 583 with 582 {corr(s[583], s[582]):.3f}, "
          f"581-582 {corr(s[581], s[582]):.3f}, with EsB 583 {corr(s[583], eb):.3f} (top half {corr(top, eb[:len(top)]):.3f}, "
          f"bottom half {corr(bot, eb[len(top):]):.3f}) | bottom-half corr with 582 {corr(bot, s[582][len(top):]):.3f}")
