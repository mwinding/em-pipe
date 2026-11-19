# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-16T09:18:13.383Z
# Ran: remote, via ssh
# Heredoc line: cd ${DEV_WORK} && cat > mask_dark.py <<'EOF'
# Assistant note before it: The new mask passes its tests: shaded and striped resin is caught, and dark and pale tissue isn't. Committing, then checking where this morning's re-run stands:

"""Background mask (pipeline.background, current) on the 35h tile slices with the most dark area."""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.expandvars("${EM_PIPE_ROOT}"))
from pipeline import background  # noqa: E402

MAIN = Path(os.path.expandvars("${DEV_35H}"))
index = pd.read_csv(MAIN / "work/preview/thumbs/index.csv", dtype={"tile": str})
index = index[index["z"] % 100 == 0]
rows = []
for npy, g in index.groupby("npy"):
    arr = np.load(MAIN / "work/preview/thumbs" / npy, mmap_mode="r")
    for r in g.itertuples():
        t8 = np.asarray(arr[int(r.i)])
        rows.append((r.z, r.tile, float((t8 < 25).mean()), t8))
res = pd.DataFrame(rows, columns=["z", "tile", "dark", "t8"]).sort_values("dark", ascending=False)
print(f"{len(res)} tile slices; with > 2% dark (< 25 of 255): {np.mean(res.dark > 0.02):.0%}; top dark fractions:",
      res.dark.head(8).round(3).tolist())
# the darkest, spread over z: at most 2 per 1000 slices
pick, used = [], {}
for r in res.itertuples():
    b = r.z // 1000
    if used.get(b, 0) < 2 and r.dark > 0.01:
        pick.append(r.Index)
        used[b] = used.get(b, 0) + 1
    if len(pick) == 18:
        break
sel = res.loc[pick]
CELL, cols = 420, 6
out = Image.new("RGB", (cols * CELL, -(-len(sel) // cols) * (CELL + 22)), (255, 255, 255))
draw = ImageDraw.Draw(out)
try:
    font = ImageFont.load_default(size=14)
except TypeError:
    font = ImageFont.load_default()
for k, r in enumerate(sel.itertuples()):
    bg = background.mask(r.t8)
    dark = r.t8 < 25
    rgb = np.repeat(r.t8[..., None], 3, axis=2).astype(np.float32)
    rgb[bg] = 0.5 * rgb[bg] + 0.5 * np.array([230, 0, 200], np.float32)
    rgb[dark & ~bg] = np.array([0, 200, 255], np.float32)      # dark but NOT background: cyan
    im = Image.fromarray(rgb.astype(np.uint8))
    im.thumbnail((CELL - 6, CELL - 6))
    x, y = (k % cols) * CELL, (k // cols) * (CELL + 22)
    out.paste(im, (x + 3, y + 22))
    draw.text((x + 4, y + 3), f"z {r.z} tile {r.tile}: dark {r.dark:.0%}, of it masked {(bg & dark).sum() / max(dark.sum(), 1):.0%}",
              fill=(0, 0, 0), font=font)
out.save(sys.argv[1])
