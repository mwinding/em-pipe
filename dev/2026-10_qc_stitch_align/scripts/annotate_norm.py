"""Burn labels into qc/normalisation_compare.tif: a title per slice, a header per column, row labels."""
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont

path = Path(sys.argv[1])
with tifffile.TiffFile(path) as t:
    stack = t.asarray()
    labels = t.imagej_metadata["Labels"]
CROP, GAP = 700, 16
COLS = ["1  current: all pixels, 0.5-99.5%", "2  tissue only, 1-99%", "3  tissue only, 5-95%",
        "4  median-anchored (tissue median -> 140)"]
ROWS = ["crop A", "crop B"]
NOTE = {3278: "last 3x3 slice before the change to 2x2 tiles", 3279: "FIRST 2x2 slice after the change (3x3 -> big 2x2)",
        3259: "20 slices before the 3x3 -> 2x2 change", 3299: "20 slices after the 3x3 -> 2x2 change",
        12528: "last big-2x2 slice before the tiles shrink", 12529: "FIRST slice with smaller 2x2 tiles",
        12509: "20 slices before the tiles shrink", 12549: "20 slices after the tiles shrink",
        1500: "middle of the 3x3 phase", 8000: "middle of the big-2x2 phase", 15000: "smaller-tile phase",
        16500: "smaller-tile phase, near the end"}
LAYOUT = {0: "3x3 tiles", 1: "3x3 tiles", 2: "2x2 big tiles", 3: "2x2 smaller tiles", 4: "2x2 smaller tiles",
          5: "2x2 smallest tiles"}
font_path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
big, small = ImageFont.truetype(str(font_path), 34), ImageFont.truetype(str(font_path), 26)
TOP, LEFT, HEAD = 70, 130, 44
n, h, w = stack.shape
out = np.full((n, TOP + HEAD + h, LEFT + w), 255, np.uint8)
new_labels = []
for k in range(n):
    m = re.match(r"z (\d+) (.+?) \(layout (\d+)\)", labels[k])
    z, when, lay = int(m.group(1)), m.group(2), int(m.group(3))
    title = f"z {z}  |  {when}  |  {LAYOUT.get(lay, f'layout {lay}')}  |  {NOTE.get(z, '')}"
    im = Image.new("L", (LEFT + w, TOP + HEAD + h), 255)
    im.paste(Image.fromarray(stack[k]), (LEFT, TOP + HEAD))
    d = ImageDraw.Draw(im)
    d.text((10, 14), title, fill=0, font=big)
    for j, c in enumerate(COLS):
        d.text((LEFT + j * (CROP + GAP) + 6, TOP + 6), c, fill=0, font=small)
    for i, r in enumerate(ROWS):
        d.text((10, TOP + HEAD + i * (CROP + GAP) + CROP // 2 - 14), r, fill=0, font=small)
    out[k] = np.asarray(im)
    new_labels.append(title)
tifffile.imwrite(path, out, imagej=True, resolution=(125.0, 125.0),
                 metadata={"spacing": 0.008, "unit": "micron", "axes": "ZYX", "Labels": new_labels})
Image.fromarray(out[3]).resize(((LEFT + w) // 3, (TOP + HEAD + h) // 3)).save(sys.argv[2])
print("annotated", n, "slices ->", path)
