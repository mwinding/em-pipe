"""Quick-look TIFF of the rendered volume for Fiji, next to it in output_dir.

Writes ``<volume stem>_<voxel>nm.tif`` from the finest pyramid scale whose uncompressed size is at
most ``export.max_gb``, so it stays openable however large the dataset is. Run after pyramid.
"""

import json
import logging
import math
import sys
from pathlib import Path

import tifffile

from . import omezarr
from .cli import atomic_write, base_parser, setup
from .render import DEFAULTS as RENDER_DEFAULTS, volume_paths

log = logging.getLogger(__name__)

DEFAULTS = {
    "export": {
        "max_gb": 4.0,   # largest TIFF to write (uint8, uncompressed); picks the finest scale that fits
    },
}


def choose_scale(root, max_bytes):
    """(scale index, shape, voxel (z, y, x) nm) of the finest scale of at most ``max_bytes``."""
    datasets = json.loads((Path(root) / "zarr.json").read_text())["attributes"]["ome"]["multiscales"][0]["datasets"]
    for s, ds in enumerate(datasets):
        shape = tuple(omezarr.open_scale(root, s).shape)
        if math.prod(shape) <= max_bytes or s == len(datasets) - 1:
            return s, shape, ds["coordinateTransformations"][0]["scale"]


def main(argv=None):
    p = base_parser(__doc__.splitlines()[0])
    args = p.parse_args(argv)
    cfg = setup(args, {**RENDER_DEFAULTS, **DEFAULTS})
    root, _ = volume_paths(cfg)
    if not (root / "zarr.json").exists():
        log.error("%s not found: run render (and pyramid) first", root)
        return 1
    s, shape, (vz, vy, vx) = choose_scale(root, float(cfg["export"]["max_gb"]) * 1e9)
    stem = root.name.removesuffix(".zarr").removesuffix(".ome")
    # Headers give e.g. 7.99986 nm: name the file by the rounded voxel size (64nm, not 63.9988nm).
    out = root.parent / f"{stem}_{round(vx)}nm.tif"
    log.info("exporting scale s%d %s at %g nm to %s", s, shape, vx, out)
    data = omezarr.open_scale(root, s).read().result()
    atomic_write(out, lambda tmp: tifffile.imwrite(
        tmp, data, imagej=True, resolution=(1000 / vx, 1000 / vy),
        metadata={"spacing": vz / 1000, "unit": "micron", "axes": "ZYX"}))
    for old in root.parent.glob(f"{stem}_*nm.tif"):   # an earlier export at another scale
        if old != out:
            old.unlink()
    log.info("wrote %s (%.2f GB)", out, out.stat().st_size / 1e9)
    return 0


if __name__ == "__main__":
    sys.exit(main())
