"""2-frame stacks for tools/triptych.py from single 32 nm planes (experiment, Oct 2026).

    python tools/experiments/clahe_stack.py SRC_DIR PLANE_DIR OUT_DIR Z[,Z...] LABEL

OUT_DIR/z<z>.tif = [frame 0 of SRC_DIR/z<z>.tif (the original volume), PLANE_DIR/z<z>.tif].
"""
import sys
from pathlib import Path

import numpy as np
import tifffile

src, planes, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
zs, label = [int(v) for v in sys.argv[4].split(",")], sys.argv[5]
out.mkdir(parents=True, exist_ok=True)
for z in zs:
    before = tifffile.imread(src / f"z{z}.tif")[0]
    p = tifffile.imread(planes / f"z{z}.tif")
    h, w = min(before.shape[0], p.shape[0]), min(before.shape[1], p.shape[1])
    tifffile.imwrite(out / f"z{z}.tif", np.stack([before[:h, :w], p[:h, :w]]), imagej=True,
                     resolution=(1 / 0.032, 1 / 0.032),
                     metadata={"spacing": 0.032, "unit": "micron", "axes": "ZYX",
                               "Labels": [f"z {z}: original volume", f"z {z}: {label}"]})
    print("done", z, flush=True)
