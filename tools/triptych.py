"""Side-by-side PNGs from two before/after stacks of the same slices (tools/shading_before_after.py):
before | A | B, labelled.

    python tools/triptych.py DIR_A DIR_B LABEL_A LABEL_B OUT_DIR [--z Z,Z,...] [--crop X0,Y0,X1,Y1] [--scale S]

--z: only these slices; --crop: a region of the stacks (their pixels); --scale: downscaling of the
panels (default 3 for whole slices, 1 with --crop).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import tifffile

p = argparse.ArgumentParser()
p.add_argument("a_dir", type=Path)
p.add_argument("b_dir", type=Path)
p.add_argument("label_a")
p.add_argument("label_b")
p.add_argument("out", type=Path)
p.add_argument("--z")
p.add_argument("--crop")
p.add_argument("--scale", type=int)
args = p.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
zs = {int(v) for v in args.z.split(",")} if args.z else None
crop = [int(v) for v in args.crop.split(",")] if args.crop else None
scale = args.scale or (1 if crop else 3)
for ta in sorted(args.a_dir.glob("z*.tif")):
    tb = args.b_dir / ta.name
    if not tb.exists() or (zs and int(ta.stem[1:]) not in zs):
        continue
    old, a = tifffile.imread(ta)
    _, b = tifffile.imread(tb)
    h = min(old.shape[0], a.shape[0], b.shape[0])
    w = min(old.shape[1], a.shape[1], b.shape[1])
    panels = [x[:h, :w] for x in (old, a, b)]
    if crop:
        x0, y0, x1, y1 = crop
        panels = [x[y0:y1, x0:x1] for x in panels]
    ph, pw = panels[0].shape
    gap = np.full((ph, 12 * scale), 255, np.uint8)
    img = np.concatenate([panels[0], gap, panels[1], gap, panels[2]], axis=1)
    img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale), interpolation=cv2.INTER_AREA)
    top = np.full((44, img.shape[1]), 255, np.uint8)
    img = np.concatenate([top, img])
    step = (pw + 12 * scale) // scale
    for k, text in enumerate([f"{ta.stem}: before", args.label_a, args.label_b]):
        cv2.putText(img, text, (k * step + 8, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    name = f"{ta.stem}_crop.png" if crop else f"{ta.stem}.png"
    cv2.imwrite(str(args.out / name), img)
    print(name)
