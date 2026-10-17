"""Side-by-side PNGs from two before/after stacks of the same slices (tools/shading_before_after.py):
old | A | B, downscaled, labelled.

    python tools/triptych.py DIR_A DIR_B LABEL_A LABEL_B OUT_DIR
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile

a_dir, b_dir, la, lb, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], Path(sys.argv[5])
out.mkdir(parents=True, exist_ok=True)
for ta in sorted(a_dir.glob("z*.tif")):
    tb = b_dir / ta.name
    if not tb.exists():
        continue
    old, a = tifffile.imread(ta)
    _, b = tifffile.imread(tb)
    h = min(old.shape[0], a.shape[0], b.shape[0])
    w = min(old.shape[1], a.shape[1], b.shape[1])
    gap = np.full((h, 16), 255, np.uint8)
    img = np.concatenate([old[:h, :w], gap, a[:h, :w], gap, b[:h, :w]], axis=1)
    img = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3), interpolation=cv2.INTER_AREA)
    ws = w // 3
    for k, text in enumerate([f"{ta.stem}: before", la, lb]):
        cv2.putText(img, text, (k * (ws + 5) + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 3)
        cv2.putText(img, text, (k * (ws + 5) + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 1)
    cv2.imwrite(str(out / f"{ta.stem}.png"), img)
    print(ta.stem)
