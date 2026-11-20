# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-18T21:53:51.846Z
# Ran: remote, via ssh
# Heredoc line: cat > ${DEV_WORK}/cluster/neuropil_pick.py <<'EOF'
# Assistant note before it: I'll pick the spots at random, but only inside neuropil. First I'll draw seeded random points on low-resolution views of 8 slices spread across all six segments. Then I'll keep the points that land in neuropil, judged from those views rather than by darkness, and run the clipping panels on those.

"""Seeded random spot candidates on 256 nm (s5) views of the final 35h volume, numbered, so neuropil
spots can be picked by eye for tools/clipping_check.py --at (coordinates: s0 px of the stored volume)."""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import json
import sys
from pathlib import Path
sys.path.insert(0, ".")
import cv2
import numpy as np
from pipeline import background, omezarr, render
from pipeline.config import load_config

cfg = load_config(os.path.expandvars("configs/${DEV_PREFIX}35h.yaml"), step_defaults=render.DEFAULTS)
root, wdir = render.volume_paths(cfg)
meta0 = json.loads((wdir / "render.json").read_text())
first_z = [pl[0][0] for pl in meta0["planes"]]
v5 = omezarr.open_scale(root, 5)
out = Path.home() / os.path.expandvars("dev-work") / "neuropil_pick"
out.mkdir(parents=True, exist_ok=True)
K, HALF = 32, 16                      # s5 px per s0 px; half a 1024 px crop at s5
rng = np.random.default_rng(35)
for k5 in (19, 62, 140, 219, 328, 406, 456, 515):
    i0 = K * k5 + K // 2              # s0 plane at the middle of this s5 plane
    z = first_z[i0]
    img = v5[k5].read().result()
    bg = background.mask(img) | (img == 0)
    ok = cv2.erode((~bg).astype(np.uint8), np.ones((2 * HALF + 1, 2 * HALF + 1), np.uint8)) > 0
    rr, cc = np.nonzero(ok)
    pick = rng.choice(len(rr), min(16, len(rr)), replace=False)
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    rows = []
    for n, j in enumerate(pick):
        r, c = int(rr[j]), int(cc[j])
        cv2.rectangle(rgb, (c - HALF, r - HALF), (c + HALF, r + HALF), (0, 0, 255), 1)
        cv2.putText(rgb, str(n), (c - HALF + 2, r - HALF - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        rows.append(f"{n}:{z}:{K * r + K // 2}:{K * c + K // 2}")
    cv2.putText(rgb, f"z {z} (s5 plane {k5}); boxes = 1024 px crops at 8 nm", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 160, 0), 2)
    cv2.imwrite(str(out / f"pick_z{z}.png"), rgb)
    print(f"z {z}: " + " ".join(rows), flush=True)
