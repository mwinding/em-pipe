"""Export a rendered volume (OME-Zarr, every pyramid scale) as a Neuroglancer precomputed volume.

Precomputed is what Zetta's and the CloudVolume tools read. ``init`` writes the ``info`` (one scale
per zarr scale: size, resolution, chunk size, encoding, sharding) and a ``provenance`` file;
``run`` (array) writes whole shards, each read from the zarr in one block, so tasks never share a
file and a shard that exists is skipped (re-runs continue); ``verify`` compares random chunk-sized
blocks of every scale with the zarr and checks that every shard exists.

Sharded (default): one file per 2^``shard_chunks_log2`` chunks (64: a 4 x 4 x 4 block of 128^3
chunks, 512^3 voxels), so a volume of terabytes is tens of thousands of files instead of tens of
millions of chunks. With an identity hash, a shard holds an aligned range of compressed Morton
codes, i.e. an aligned block of chunks (see ``shard_blocks``). Output: output_dir/<name>.precomputed.
"""

import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorstore as ts

from . import omezarr
from .cli import atomic_write, base_parser, my_chunks, setup, task_info
from .render import DEFAULTS as RENDER_DEFAULTS, volume_paths

log = logging.getLogger(__name__)

DEFAULTS = {
    "precomputed": {
        "enabled": False,           # run_pipeline.sh: export after pyramid
        "name": None,               # folder in output_dir; null: <volume stem>.precomputed
        "chunk": [128, 128, 128],   # chunk size, x y z voxels
        "encoding": "raw",          # raw (lossless, gzip in the shards) | jpeg (lossy, smaller)
        "jpeg_quality": 90,
        "sharded": True,
        "shard_chunks_log2": 6,     # chunks per shard = 2^this (6: 4 x 4 x 4)
        "minishard_chunks_log2": 3,  # chunks per minishard = 2^this
        "scales": None,             # zarr scales to export, e.g. [0, 1, 2]; null: all
        "description": "",          # provenance
        "owners": [],               # provenance, e.g. ["name@crick.ac.uk"]
    }
}
ENCODINGS = ("raw", "jpeg")


def paths(cfg):
    """(zarr volume, precomputed folder)."""
    zarr_root, _ = volume_paths(cfg)
    name = cfg["precomputed"]["name"] or zarr_root.name.removesuffix(".zarr").removesuffix(".ome") + ".precomputed"
    return zarr_root, Path(cfg["output_dir"]) / name


def _bits(n):
    return max(0, math.ceil(math.log2(n))) if n > 1 else 0


def _order(grid):
    """Compressed Morton code bit order: [(dim, level)], lowest bit first."""
    bits = [_bits(g) for g in grid]
    return [(d, i) for i in range(max(bits, default=0)) for d in range(len(grid)) if i < bits[d]]


def shard_blocks(grid, k):
    """[(shard number, chunk origin, chunk block shape)] covering a chunk grid (x, y, z) exactly once,
    each the aligned range of 2^k compressed Morton codes one shard holds (identity hash)."""
    order = _order(grid)
    k = min(k, len(order))
    block = [1] * len(grid)
    for d, _ in order[:k]:
        block[d] *= 2
    out = []
    for s in range(2 ** (len(order) - k)):
        origin = [0] * len(grid)
        for p, (d, i) in enumerate(order[k:], start=k):
            origin[d] |= ((s >> (p - k)) & 1) << i
        if all(o < g for o, g in zip(origin, grid)):
            out.append((s, origin, [min(b, g - o) for b, o, g in zip(block, origin, grid)]))
    return out


def sharding_spec(grid, p):
    """neuroglancer_uint64_sharded_v1 for a chunk grid, or None if unsharded."""
    if not p["sharded"]:
        return None
    total = len(_order(grid))
    k = min(int(p["shard_chunks_log2"]), total)
    pre = min(int(p["minishard_chunks_log2"]), k)
    return {"@type": "neuroglancer_uint64_sharded_v1", "hash": "identity", "preshift_bits": pre,
            "minishard_bits": k - pre, "shard_bits": total - k, "minishard_index_encoding": "gzip",
            "data_encoding": "raw" if p["encoding"] == "jpeg" else "gzip"}


def shard_file(scale_key, sharding, s):
    width = max(1, math.ceil(sharding["shard_bits"] / 4))
    return f"{scale_key}/{s:0{width}x}.shard"


def make_info(zarr_root, p):
    """The precomputed ``info`` for the zarr's scales (x, y, z order)."""
    ms = json.loads((Path(zarr_root) / "zarr.json").read_text())["attributes"]["ome"]["multiscales"][0]
    datasets = ms["datasets"]
    scales = p["scales"] if p["scales"] is not None else list(range(len(datasets)))
    chunk = [int(c) for c in p["chunk"]]
    out = []
    for s in scales:
        z, y, x = omezarr.open_scale(zarr_root, s).shape
        vz, vy, vx = datasets[s]["coordinateTransformations"][0]["scale"]
        size = [int(x), int(y), int(z)]
        grid = [math.ceil(n / c) for n, c in zip(size, chunk)]
        scale = {"key": "_".join(str(round(v)) for v in (vx, vy, vz)), "size": size,
                 "resolution": [float(vx), float(vy), float(vz)], "voxel_offset": [0, 0, 0],
                 "chunk_sizes": [chunk], "encoding": p["encoding"]}
        if p["encoding"] == "jpeg":
            scale["jpeg_quality"] = int(p["jpeg_quality"])
        sharding = sharding_spec(grid, p)
        if sharding:
            scale["sharding"] = sharding
        out.append({"zarr_scale": int(s), **scale})
    info = {"@type": "neuroglancer_multiscale_volume", "type": "image", "data_type": "uint8", "num_channels": 1,
            "scales": [{k: v for k, v in sc.items() if k != "zarr_scale"} for sc in out]}
    return info, [sc["zarr_scale"] for sc in out]


def _check(p):
    if p["encoding"] not in ENCODINGS:
        raise ValueError(f"precomputed.encoding must be one of {ENCODINGS}, not {p['encoding']!r}")
    if len(p["chunk"]) != 3 or min(p["chunk"]) < 1:
        raise ValueError("precomputed.chunk must be three positive sizes (x, y, z)")


def init(cfg, overwrite=False):
    p = cfg["precomputed"]
    _check(p)
    zarr_root, dest = paths(cfg)
    if not (zarr_root / "zarr.json").exists():
        log.error("%s not found: run render and pyramid first", zarr_root)
        return 1
    info, zarr_scales = make_info(zarr_root, p)
    old = dest / "info"
    if old.exists() and not overwrite:
        have = json.loads(old.read_text())
        if have != info:
            log.error("%s differs from the current settings or volume: re-run init with --overwrite "
                      "(the old shards are deleted)", old)
            return 1
        log.info("%s matches: kept", old)
        return 0
    if dest.exists() and overwrite:
        import shutil
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    atomic_write(dest / "info", lambda t: Path(t).write_text(json.dumps(info, indent=1)))
    prov = {"description": p["description"] or f"em-pipe export of {zarr_root.name}",
            "owners": list(p["owners"]), "sources": [str(zarr_root)],
            "processing": [{"method": {"task": "em-pipe precomputed", "zarr_scales": zarr_scales,
                                       "encoding": p["encoding"], "chunk": p["chunk"]},
                            "date": datetime.now(timezone.utc).isoformat(timespec="seconds")}]}
    atomic_write(dest / "provenance", lambda t: Path(t).write_text(json.dumps(prov, indent=1)))
    for sc in info["scales"]:
        n = len(shard_blocks([math.ceil(a / b) for a, b in zip(sc["size"], sc["chunk_sizes"][0])],
                             int(p["shard_chunks_log2"]))) if "sharding" in sc else None
        log.info("scale %s: size %s, %s", sc["key"], sc["size"], f"{n} shards" if n else "unsharded")
    return 0


def _units(info, p):
    """[(scale index, shard number, chunk origin, block)] over every scale."""
    out = []
    for i, sc in enumerate(info["scales"]):
        grid = [math.ceil(a / b) for a, b in zip(sc["size"], sc["chunk_sizes"][0])]
        k = int(p["shard_chunks_log2"]) if "sharding" in sc else 6   # unsharded: same blocks as work units
        out += [(i, s, o, b) for s, o, b in shard_blocks(grid, k)]
    return out


def run(cfg, task_id, num_tasks, overwrite=False):
    p = cfg["precomputed"]
    zarr_root, dest = paths(cfg)
    if not (dest / "info").exists():
        log.error("%s/info not found: run 'precomputed init' first", dest)
        return 1
    info = json.loads((dest / "info").read_text())
    want, zarr_scales = make_info(zarr_root, p)
    if want != info:
        log.error("%s/info differs from the current settings or volume: run 'precomputed init --overwrite'", dest)
        return 1
    units = my_chunks(_units(info, p), task_id, num_tasks)
    log.info("task %d/%d: %d shards", task_id, num_tasks, len(units))
    stores, sources = {}, {}
    done = skipped = 0
    for i, s, origin, block in units:
        sc = info["scales"][i]
        if "sharding" in sc and not overwrite and (dest / shard_file(sc["key"], sc["sharding"], s)).exists():
            skipped += 1
            continue
        if i not in stores:
            stores[i] = ts.open({"driver": "neuroglancer_precomputed",
                                 "kvstore": {"driver": "file", "path": str(dest) + "/"}, "scale_index": i},
                                open=True, write=True).result()
            sources[i] = omezarr.open_scale(zarr_root, zarr_scales[i])
        c = sc["chunk_sizes"][0]
        lo = [o * cc for o, cc in zip(origin, c)]
        hi = [min((o + b) * cc, n) for o, b, cc, n in zip(origin, block, c, sc["size"])]
        data = sources[i][lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]].read().result()
        txn = ts.Transaction()
        stores[i].with_transaction(txn)[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2], 0].write(
            np.ascontiguousarray(np.transpose(data, (2, 1, 0)))).result()
        txn.commit_async().result()
        done += 1
    log.info("task %d: wrote %d shards, %d already done", task_id, done, skipped)
    return 0


def verify(cfg, samples=24, seed=0):
    """Every shard file present; random chunk-sized blocks of every scale equal to the zarr."""
    p = cfg["precomputed"]
    zarr_root, dest = paths(cfg)
    info = json.loads((dest / "info").read_text())
    _, zarr_scales = make_info(zarr_root, p)
    rng = np.random.default_rng(seed)
    bad = 0
    for i, sc in enumerate(info["scales"]):
        if "sharding" in sc:
            grid = [math.ceil(a / b) for a, b in zip(sc["size"], sc["chunk_sizes"][0])]
            missing = [s for s, _, _ in shard_blocks(grid, int(p["shard_chunks_log2"]))
                       if not (dest / shard_file(sc["key"], sc["sharding"], s)).exists()]
            if missing:
                log.error("scale %s: %d shard files missing, e.g. %s", sc["key"], len(missing), missing[:5])
                bad += 1
                continue
        store = ts.open({"driver": "neuroglancer_precomputed", "kvstore": {"driver": "file", "path": str(dest) + "/"},
                         "scale_index": i}, open=True).result()
        src = omezarr.open_scale(zarr_root, zarr_scales[i])
        c, size = sc["chunk_sizes"][0], sc["size"]
        diffs = 0
        for _ in range(samples):
            lo = [int(rng.integers(0, max(1, n - cc + 1))) for n, cc in zip(size, c)]
            hi = [min(a + cc, n) for a, cc, n in zip(lo, c, size)]
            a = store[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2], 0].read().result()
            b = np.transpose(src[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]].read().result(), (2, 1, 0))
            tol = 0 if sc["encoding"] == "raw" else 255
            if a.shape != b.shape or np.abs(a.astype(int) - b.astype(int)).max() > tol:
                diffs += 1
        if sc["encoding"] == "jpeg":
            log.info("scale %s: %d samples read (jpeg: lossy, not compared exactly)", sc["key"], samples)
        elif diffs:
            log.error("scale %s: %d of %d sampled blocks differ from the zarr", sc["key"], diffs, samples)
            bad += 1
        else:
            log.info("scale %s: %d sampled blocks identical to the zarr", sc["key"], samples)
    return 1 if bad else 0


def main(argv=None):
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("command", choices=["init", "run", "verify"])
    args = p.parse_args(argv)
    cfg = setup(args, {**RENDER_DEFAULTS, **DEFAULTS})
    if args.command == "init":
        return init(cfg, args.overwrite)
    if args.command == "run":
        return run(cfg, *task_info(args), overwrite=args.overwrite)
    return verify(cfg)


if __name__ == "__main__":
    sys.exit(main())
