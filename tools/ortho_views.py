"""x-y, x-z and y-z views through the same point of two or more rendered volumes, at full resolution: e.g.
slice-to-slice flicker, which shows as horizontal stripes in x-z and y-z.

For every point (z, y, x), in voxels of scale 0: a ``size`` px square x-y view at z centred on (y, x); an x-z
view along the line y (x across, z down, every plane of the volume or ``z_range``); a y-z view along x (y
across, z down). One PNG per view and point, the volumes stacked, each under a label bar; 1 px per voxel and
the volumes' own greys (no contrast stretch, no interpolation).

Spec (JSON): {"volumes": [{"label": "...", "source": "..."}, ...], "points": [[z, y, x], ...],
              "size": 2048, "z_range": [z0, z1]}

  python tools/ortho_views.py SPEC.json OUT_DIR     # OUT_DIR/ortho_z<z>_y<y>_x<x>_{xy,xz,yz}.png
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from volume_compare import Volume  # noqa: E402

BAR = 44      # label bar height, px
GAP = 8       # white gap between panels, px


def window(c, size, n):
    """(start, stop) of a ``size`` window centred on c, kept inside 0..n."""
    s = min(size, n)
    a = int(np.clip(c - s // 2, 0, n - s))
    return a, a + s


def views(vol, z, y, x, size, z_range=None):
    """{'xy': (y, x) array, 'xz': (z, x), 'yz': (z, y)} at scale 0."""
    nz, ny, nx = vol.shape(0)
    y0, y1 = window(y, size, ny)
    x0, x1 = window(x, size, nx)
    za, zb = (max(0, int(z_range[0])), min(nz, int(z_range[1]))) if z_range else (0, nz)
    a = vol.scale(0)
    if vol.zarr:
        xz = np.asarray(a[za:zb, y, x0:x1].read().result())
        yz = np.asarray(a[za:zb, y0:y1, x].read().result())
    else:
        xz = np.asarray(a[x0:x1, y, za:zb, 0].read().result()).T
        yz = np.asarray(a[x, y0:y1, za:zb, 0].read().result()).T
    return {"xy": vol.plane(0, z, y0, y1, x0, x1), "xz": xz, "yz": yz}, (y0, x0, za)


def stack(panels, labels):
    """Panels (uint8, same width) one under the other, each under a label bar."""
    w = max(p.shape[1] for p in panels)
    rows = []
    for p, label in zip(panels, labels):
        bar = np.full((BAR, w), 255, np.uint8)
        cv2.putText(bar, label, (8, BAR - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2, cv2.LINE_AA)
        img = np.full((p.shape[0], w), 255, np.uint8)
        img[:, :p.shape[1]] = p
        rows += [bar, img, np.full((GAP, w), 255, np.uint8)]
    return np.vstack(rows[:-1])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("spec", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args(argv)
    spec = json.loads(args.spec.read_text())
    vols = [(v["label"], Volume(v["source"])) for v in spec["volumes"]]
    size = int(spec.get("size", 2048))
    args.out.mkdir(parents=True, exist_ok=True)
    for z, y, x in spec["points"]:
        got = [(label, *views(v, int(z), int(y), int(x), size, spec.get("z_range"))) for label, v in vols]
        y0, x0, za = got[0][2]
        where = {"xy": f"z {z}, y {y0}-{y0 + size}, x {x0}-{x0 + size}",
                 "xz": f"y {y}, x {x0}-{x0 + size}, z down from {za}",
                 "yz": f"x {x}, y {y0}-{y0 + size}, z down from {za}"}
        for view in ("xy", "xz", "yz"):
            img = stack([np.asarray(g[1][view], np.uint8) for g in got],
                        [f"{g[0]}   ({view}: {where[view]})" for g in got])
            out = args.out / f"ortho_z{z}_y{y}_x{x}_{view}.png"
            cv2.imwrite(str(out), img)
            print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
