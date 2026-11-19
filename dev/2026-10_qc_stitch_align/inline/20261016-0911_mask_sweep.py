# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-16T09:11:54.975Z
# Ran: remote, via ssh
# Heredoc line: cd ${DEV_WORK} && cat > mask_sweep.py <<'EOF'
# Assistant note before it: All new tests pass. Running the full suite, then committing:

"""Background mask variants side by side on hard tiles: rows = tiles, columns = v1 (bright/dark and
uniform) and v2 (texture-free along y) at several thresholds. Magenta = background."""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

sys.path.insert(0, os.path.expandvars("${EM_PIPE_ROOT}"))
from pipeline import background  # noqa: E402

MAIN = Path(os.path.expandvars("${DEV_35H}"))
TILES = [(1200, "2-1"), (2100, "2-1"), (3300, "1-1"), (11700, "1-1"), (13500, "0-1"), (15300, "0-1")]
SDS = [2.0, 2.5, 3.0, 4.0]
CELL = 360


def v2(img8, max_sd_y, min_px=400, grow_px=3):
    t = np.asarray(img8, np.float32)
    m = ndimage.gaussian_filter(t, (4, 1))
    sd_y = np.sqrt(np.maximum(ndimage.gaussian_filter(t * t, (4, 1)) - m * m, 0))
    sd_y = ndimage.gaussian_filter(sd_y, 2)
    bg = ndimage.binary_opening(sd_y < max_sd_y, iterations=2)
    lab, n = ndimage.label(bg)
    if n:
        sizes = np.asarray(ndimage.sum(bg, lab, range(1, n + 1)))
        bg = np.isin(lab, 1 + np.flatnonzero(sizes >= min_px))
    return ndimage.binary_dilation(bg, iterations=grow_px) if bg.any() else bg


index = pd.read_csv(MAIN / "work/preview/thumbs/index.csv", dtype={"tile": str}).set_index(["z", "tile"])
cols = ["v1 (bright/dark, uniform)"] + [f"v2 smooth along y, sd < {s:g}" for s in SDS]
out = Image.new("RGB", (len(cols) * CELL, 24 + len(TILES) * (CELL + 20)), (255, 255, 255))
draw = ImageDraw.Draw(out)
try:
    font = ImageFont.load_default(size=13)
except TypeError:
    font = ImageFont.load_default()
for j, c in enumerate(cols):
    draw.text((j * CELL + 4, 4), c, fill=(0, 0, 0), font=font)
for i, (z, tile) in enumerate(TILES):
    r = index.loc[(z, tile)]
    t8 = np.asarray(np.load(MAIN / "work/preview/thumbs" / r["npy"], mmap_mode="r")[int(r["i"])])
    masks = [background.mask(t8)] + [v2(t8, s) for s in SDS]
    for j, bg in enumerate(masks):
        rgb = np.repeat(t8[..., None], 3, axis=2).astype(np.float32)
        rgb[bg] = 0.55 * rgb[bg] + 0.45 * np.array([230, 0, 200], np.float32)
        im = Image.fromarray(rgb.astype(np.uint8))
        im.thumbnail((CELL - 6, CELL - 6))
        x, y = j * CELL, 24 + i * (CELL + 20)
        out.paste(im, (x + 3, y + 18))
        draw.text((x + 4, y + 2), f"z {z} tile {tile}: background {bg.mean():.0%}", fill=(0, 0, 0), font=font)
out.save(sys.argv[1])
