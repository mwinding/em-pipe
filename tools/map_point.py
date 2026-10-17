"""Map a point from one rendered volume to another of the same dataset (e.g. a test render to the
full volume): the plane by acquisition time, x/y by template matching at a coarse scale.

    python tools/map_point.py CONFIG_A VOLUME_A CONFIG_B VOLUME_B X Y Z [SCALE]

CONFIG/VOLUME pairs name the two renders (volume folders in each config's output_dir). X Y Z are
in VOLUME_A's voxels (as neuroglancer shows them). Matches a 1024 px square of VOLUME_A's plane at
SCALE (default 3, i.e. 8x coarser) within VOLUME_B's corresponding plane; prints the point in
VOLUME_B's voxels and the match score (normalised correlation; > 0.5 is a confident match).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import omezarr  # noqa: E402
from pipeline.config import load_config  # noqa: E402
from pipeline.slices import load_slices  # noqa: E402


def plane_z(cfg, volume, plane):
    """Global z of a render's output plane (single-slice planes: no downsampling in z)."""
    stem = volume.removesuffix(".zarr").removesuffix(".ome")
    meta = json.loads((Path(cfg["output_dir"]) / "work" / "render" / stem / "render.json").read_text())
    return meta["planes"], meta["planes"][plane][0][0]


def main():
    cfg_a, vol_a, cfg_b, vol_b = load_config(sys.argv[1]), sys.argv[2], load_config(sys.argv[3]), sys.argv[4]
    x, y, z = (float(v) for v in sys.argv[5:8])
    scale = int(sys.argv[8]) if len(sys.argv) > 8 else 3
    f = 2 ** scale
    _, z_a = plane_z(cfg_a, vol_a, int(round(z)))
    when = load_slices(cfg_a).drop_duplicates("z").set_index("z").loc[z_a, "timestamp"]
    sl_b = load_slices(cfg_b).drop_duplicates("z")
    z_b = int(sl_b.loc[sl_b["timestamp"] == when, "z"].iloc[0])
    planes_b, _ = plane_z(cfg_b, vol_b, 0)
    plane_b = next(i for i, p in enumerate(planes_b) if p[0][0] == z_b)
    a = omezarr.open_scale(Path(cfg_a["output_dir"]) / vol_a, scale)
    b = omezarr.open_scale(Path(cfg_b["output_dir"]) / vol_b, scale)
    pa = a[int(round(z)) // f].read().result()
    pb = b[plane_b // f].read().result()
    half = 512
    cx, cy = int(x / f), int(y / f)
    x0, y0 = max(cx - half, 0), max(cy - half, 0)
    tmpl = pa[y0:y0 + 2 * half, x0:x0 + 2 * half].astype(np.float32)
    res = cv2.matchTemplate(pb.astype(np.float32), tmpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, (bx, by) = cv2.minMaxLoc(res)
    dx, dy = (bx - x0) * f, (by - y0) * f
    print(f"{vol_a} plane {int(round(z))} = slice z {z_a} ({when}) -> {vol_b} plane {plane_b}")
    print(f"offset (x, y) {dx:+d}, {dy:+d} voxels; match score {score:.2f}")
    print(f"point in {vol_b}: [{x + dx:.0f}, {y + dy:.0f}, {plane_b}]")


if __name__ == "__main__":
    main()
