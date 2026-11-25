"""A square of tissue in a rendered volume at given planes, as a render.bbox for a test slab.

    python dev/2026-11_zstep_correction/scripts/pick_box.py VOLUME.ome.zarr RENDER.json SIZE Z [Z ...]

For each plane Z: on scale 5 (32x coarser), the SIZE px (scale 0) square with no empty voxels, the least resin
(greys above 200) and the most texture (SD), preferring squares near the canvas centre. Prints its centre in the
volume's voxels (x, y) and the square in aligned px (render.bbox: [xmin, ymin, xmax, ymax]), using the frame
in the volume's render.json (work/render/<volume>/render.json: origin_xy, canvas_size_xy, settings.rotate_deg).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pipeline import omezarr  # noqa: E402

volume, frame_path, size = sys.argv[1], sys.argv[2], int(sys.argv[3])
frame = json.loads(Path(frame_path).read_text())
(ox, oy), (W, H) = frame["origin_xy"], frame["canvas_size_xy"]
a = omezarr.open_scale(volume.rstrip("/"), 5)
c = size // 32
for z in (int(v) for v in sys.argv[4:]):
    p = np.asarray(a[min(z // 32, a.shape[0] - 1)].read().result()).astype(np.float32)
    best = None
    for i in range(0, p.shape[0] - c + 1, c // 4):
        for j in range(0, p.shape[1] - c + 1, c // 4):
            b = p[i:i + c, j:j + c]
            if b.min() == 0:
                continue
            far = np.hypot((i + c / 2) / p.shape[0] - 0.5, (j + c / 2) / p.shape[1] - 0.5)
            score = b.std() * (1 - (b > 200).mean()) ** 4 * (1 - far)
            if best is None or score > best[0]:
                best = (score, i, j, float((b > 200).mean()), float(b.mean()), float(b.std()))
    if best is None:
        print(f"z {z}: no square without empty voxels")
        continue
    _, i, j, resin, mean, sd = best
    x0, y0 = j * 32, i * 32                               # volume voxels
    if frame["settings"].get("rotate_deg", 0) == 180:
        ax0, ay0 = ox + (W - 1 - (x0 + size - 1)), oy + (H - 1 - (y0 + size - 1))
    else:
        ax0, ay0 = ox + x0, oy + y0
    print(f"z {z}: centre x {x0 + size // 2} y {y0 + size // 2} (volume voxels); mean {mean:.0f}, SD {sd:.0f}, "
          f"resin {resin:.2f}; render.bbox [{ax0}, {ay0}, {ax0 + size}, {ay0 + size}]")
