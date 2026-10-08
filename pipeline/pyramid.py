"""Lower-resolution scales s1..s{N-1} of the rendered OME-Zarr volume by 2×2×2 mean pooling.

render init creates every scale's array and render run fills s0. ``run --scale s`` fills
scale s from scale s-1, one whole shard per write, so scale s-1 must be complete: one job
per scale, each after the previous; its array tasks take the shards strided.
Each finished shard leaves ``render/done/s{s}_{index:06d}`` (index in omezarr.shard_boxes
order). A marker counts only if it is newer than its scale array (so render init redoes the
pyramid) and than the s-1 shard files it was pooled from (so shards whose source was
re-rendered, or written late, are redone without --overwrite).
"""

import itertools
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

from . import omezarr
from .cli import atomic_write, base_parser, my_chunks, setup, task_info
from .config import step_dir

try:  # render's defaults supply render.name; pyramid also works on a volume made without it
    from .render import DEFAULTS as RENDER_DEFAULTS
except ImportError:
    RENDER_DEFAULTS = {}

log = logging.getLogger(__name__)

# No settings of its own: scale count, chunk and shard shapes are fixed by render init.
DEFAULTS = {"pyramid": {}}


def volume_path(cfg):
    """output_dir/render/<render.name>."""
    return step_dir(cfg, "render", cfg.get("render", {}).get("name", "volume.ome.zarr"))


def downsample(block):
    """2×2×2 mean of a uint8 block [z, y, x], rounded to nearest (halves up).

    An odd trailing plane, row or column is edge-padded: every voxel that exists is then
    counted equally often, so those output voxels are the mean of the voxels that exist.
    """
    pad = [(0, n % 2) for n in block.shape]
    if any(p for _, p in pad):
        block = np.pad(block, pad, mode="edge")
    s = np.add(block[0::2], block[1::2], dtype=np.uint16)
    s = s[:, 0::2] + s[:, 1::2]
    s = s[:, :, 0::2] + s[:, :, 1::2]
    return ((s + 4) >> 3).astype(np.uint8)


def _mtime(path):
    """Modification time of ``path``, or -inf if it does not exist."""
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return -math.inf


def _shard_file(root, scale, pos):
    """File of the shard at grid position ``pos`` (zarr v3 default chunk key encoding)."""
    return Path(root) / f"s{scale}" / "c" / "/".join(map(str, pos))


def _slices(box):
    return tuple(slice(a, b) for a, b in box)


def run(cfg, args):
    root = volume_path(cfg)
    s = args.scale
    if s < 1 or not (root / f"s{s}" / "zarr.json").exists():
        log.error("%s has no scale s%d to build (s >= 1; scales are created by render init)", root, s)
        return 1
    src, dst = omezarr.open_scale(root, s - 1), omezarr.open_scale(root, s)
    src_shard = omezarr.shard_shape(src)
    done = step_dir(cfg, "render", "done")
    created = _mtime(root / f"s{s}" / "zarr.json")
    boxes = omezarr.shard_boxes(dst.shape, omezarr.shard_shape(dst))
    task_id, num_tasks = task_info(args)
    mine = my_chunks(list(enumerate(boxes)), task_id, num_tasks)

    todo = []
    for i, box in mine:
        src_box = tuple((2 * a, min(2 * b, n)) for (a, b), n in zip(box, src.shape))
        sources = list(itertools.product(*(range(a // c, -(-b // c)) for (a, b), c in zip(src_box, src_shard))))
        # A source shard rewritten as all zeros is not stored, so only --overwrite catches that.
        if args.overwrite or _mtime(done / f"s{s}_{i:06d}") <= max(
                [created] + [_mtime(_shard_file(root, s - 1, p)) for p in sources]):
            todo.append((i, box, src_box, sources))

    if s >= 2:  # only the s-1 shards this task reads; for s = 1, render's s0 is complete by job order
        prev = _mtime(root / f"s{s - 1}" / "zarr.json")
        grid = [-(-n // c) for n, c in zip(src.shape, src_shard)]
        need = sorted({int(np.ravel_multi_index(p, grid)) for *_, sources in todo for p in sources})
        missing = [j for j in need if _mtime(done / f"s{s - 1}_{j:06d}") <= prev]
        if missing:
            log.error("scale s%d is incomplete (%d of the %d shards this task reads are not done, e.g. %d): "
                      "build it first", s - 1, len(missing), len(need), missing[0])
            return 1

    log.info("s%d %s: task %d/%d has %d of %d shards, %d to do",
             s, dst.shape, task_id, num_tasks, len(mine), len(boxes), len(todo))
    for i, box, src_box, _ in todo:
        t0 = time.time()
        block = src[_slices(src_box)].read().result()
        dst[_slices(box)].write(downsample(block)).result()
        atomic_write(done / f"s{s}_{i:06d}", lambda p: Path(p).write_text(f"{box}\n"))
        log.info("s%d shard %d %s: %.1f s", s, i, box, time.time() - t0)
    log.info("s%d: wrote %d shards, %d already done", s, len(todo), len(mine) - len(todo))
    return 0


def main(argv=None):
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("command", choices=["run"])
    p.add_argument("--scale", type=int, required=True, help="scale to build from the one above (1..num_scales-1)")
    args = p.parse_args(argv)
    cfg = setup(args, {**RENDER_DEFAULTS, **DEFAULTS})
    return run(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
