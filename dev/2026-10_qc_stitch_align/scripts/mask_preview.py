"""Picture of the background mask on real 35h tile thumbnails: background tinted magenta.

Samples every STEP-th slice, all tiles; per tile layout picks the tiles with the most background,
some with a little, and one with none. Writes a PNG montage with labels.
"""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.expandvars("${EM_PIPE_ROOT}"))
from pipeline import background  # noqa: E402

MAIN = Path(os.path.expandvars("${DEV_35H}"))
STEP, CELL = 300, 420
stats = pd.read_csv(MAIN / "work/preview/stats.csv", dtype={"tile": str}, usecols=["z", "tile", "timestamp"])
index = pd.read_csv(MAIN / "work/preview/thumbs/index.csv", dtype={"tile": str})
seg = pd.read_csv(MAIN / "work/check/slices.csv", usecols=["z", "segment"]).drop_duplicates("z")
df = stats.merge(index, on=["z", "tile"]).merge(seg, on="z")
df = df[df["z"] % STEP == 0]
rows = []
for npy, g in df.groupby("npy"):
    arr = np.load(MAIN / "work/preview/thumbs" / npy, mmap_mode="r")
    for r in g.itertuples():
        t8 = np.asarray(arr[int(r.i)])
        bg = background.mask(t8)
        rows.append((r.z, r.tile, r.segment, r.timestamp[:16], float(bg.mean()), t8, bg))
res = pd.DataFrame(rows, columns=["z", "tile", "segment", "time", "bgfrac", "t8", "bg"])
print(f"{len(res)} tile slices sampled; background > 5% in {np.mean(res.bgfrac > 0.05):.0%}, "
      f"median background where present {res.bgfrac[res.bgfrac > 0.05].median():.0%}")
pick = []
for s, g in res.groupby("segment"):
    g = g.sort_values("bgfrac", ascending=False)
    pick += list(g.head(3).index)                                   # the most background
    mid = g[(g.bgfrac > 0.02) & (g.bgfrac < 0.25)]
    pick += list(mid.sample(min(2, len(mid)), random_state=0).index)  # a little
    pick += list(g[g.bgfrac == 0].head(1).index)                    # none
sel = res.loc[list(dict.fromkeys(pick))]
cols = 6
nrow = -(-len(sel) // cols)
out = Image.new("RGB", (cols * CELL, nrow * (CELL + 22)), (255, 255, 255))
draw = ImageDraw.Draw(out)
try:
    font = ImageFont.load_default(size=14)
except TypeError:
    font = ImageFont.load_default()
for k, r in enumerate(sel.itertuples()):
    rgb = np.repeat(r.t8[..., None], 3, axis=2).astype(np.float32)
    tint = np.array([230, 0, 200], np.float32)
    rgb[r.bg] = 0.55 * rgb[r.bg] + 0.45 * tint
    im = Image.fromarray(rgb.astype(np.uint8))
    im.thumbnail((CELL - 6, CELL - 6))
    x, y = (k % cols) * CELL, (k // cols) * (CELL + 22)
    out.paste(im, (x + 3, y + 22))
    draw.text((x + 4, y + 3), f"z {r.z} tile {r.tile} ({r.time[5:]})  background {r.bgfrac:.0%}", fill=(0, 0, 0), font=font)
out.save(sys.argv[1])
print("picked", len(sel), "->", sys.argv[1])
