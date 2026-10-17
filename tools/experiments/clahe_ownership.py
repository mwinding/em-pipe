"""Which tile owns each 32 nm pixel, and where each pair of tiles meets (experiment, Oct 2026).

    python tools/experiments/clahe_ownership.py CONFIG OLD_VOLUME Z[,Z...] OUT_DIR

Uses tools/seam_steps.ownership (the real render with every tile a constant). Writes OUT_DIR/z<z>.tif
(uint8 labels, 0 = no data) and OUT_DIR/z<z>.json: the tile of every label, and per pair of tiles
sharing a seam its pixel count and bounding box (x0, y0, x1, y1) at 32 nm.
"""
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import render, shading  # noqa: E402
from pipeline.config import deep_merge, load_config  # noqa: E402
from seam_steps import ownership  # noqa: E402

cfg = load_config(sys.argv[1], step_defaults=deep_merge(render.DEFAULTS, shading.DEFAULTS))
old, zs, out = sys.argv[2], [int(v) for v in sys.argv[3].split(",")], Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
for z in zs:
    lab, tiles = ownership(cfg, old, z)
    tifffile.imwrite(out / f"z{z}.tif", lab)
    seams = {}
    for a in range(1, len(tiles) + 1):
        grown = ndimage.binary_dilation(lab == a)
        for b in range(a + 1, len(tiles) + 1):
            m = grown & (lab == b)
            if m.sum() < 50:
                continue
            ys, xs = np.nonzero(m)
            seams[f"{tiles[a - 1]}|{tiles[b - 1]}"] = {"px": int(m.sum()), "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]}
    (out / f"z{z}.json").write_text(json.dumps({"tiles": tiles, "seams": seams}, indent=1))
    print(z, json.dumps(seams), flush=True)
