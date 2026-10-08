"""OME-Zarr 0.5 multiscale volumes (Zarr v3 with sharding) via tensorstore.

Layout: ``<root>/zarr.json`` (group with OME metadata) and one array per scale,
``<root>/s0``, ``<root>/s1``, ... Each scale halves z, y and x (mean pooling).
Arrays are sharded so a ~TB volume is thousands of files, not millions: a shard is
``shard`` voxels holding ``chunk``-sized inner chunks that neuroglancer fetches
individually with HTTP range requests. Writers must write whole shards; render
writes z-slabs of exactly ``shard[0]`` planes over the full xy extent, pyramid writes
one shard at a time.
"""

import json
import math
import os
from pathlib import Path

import tensorstore as ts

COMPRESSORS = {
    "none": [],
    "gzip": [{"name": "gzip", "configuration": {"level": 3}}],
    "zstd": [{"name": "zstd", "configuration": {"level": 3, "checksum": False}}],
    "blosc": [{"name": "blosc", "configuration": {"cname": "lz4", "clevel": 5, "shuffle": "noshuffle",
                                                  "typesize": 1, "blocksize": 0}}],
}


def scale_shapes(shape0, num_scales):
    """Shapes of s0..s{num_scales-1}; each scale is ceil(previous / 2) per axis."""
    shapes = [tuple(int(v) for v in shape0)]
    for _ in range(1, num_scales):
        shapes.append(tuple(max(1, math.ceil(v / 2)) for v in shapes[-1]))
    return shapes


def _array_spec(path, shape, chunk, shard, compression):
    return {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(path)},
        "metadata": {
            "shape": list(shape),
            "data_type": "uint8",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(shard)}},
            "chunk_key_encoding": {"name": "default"},
            "codecs": [{
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": list(chunk),
                    "codecs": [{"name": "bytes"}, *COMPRESSORS[compression]],
                    "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}},
                                     {"name": "crc32c"}],
                    "index_location": "end",
                },
            }],
            "fill_value": 0,
            "dimension_names": ["z", "y", "x"],
        },
    }


def create(root, shape0, voxel_nm, num_scales=7, chunk=(64, 64, 64), shard=(64, 1024, 1024),
           compression="gzip", name="volume"):
    """Create the group and empty arrays for every scale (deleting any existing ones).

    voxel_nm: (z, y, x) voxel size of s0 in nanometres.
    Returns the list of scale shapes.
    """
    root = Path(root)
    if any(s % c for s, c in zip(shard, chunk)):
        raise ValueError(f"shard {shard} must be a multiple of chunk {chunk}")
    root.mkdir(parents=True, exist_ok=True)
    shapes = scale_shapes(shape0, num_scales)
    datasets = []
    for s, shape in enumerate(shapes):
        ts.open(_array_spec(root / f"s{s}", shape, chunk, shard, compression),
                create=True, delete_existing=True).result()
        f = 2 ** s
        datasets.append({
            "path": f"s{s}",
            "coordinateTransformations": [
                {"type": "scale", "scale": [v * f for v in voxel_nm]},
                # Mean pooling puts a downsampled voxel's centre between its source voxels.
                {"type": "translation", "translation": [v * (f - 1) / 2 for v in voxel_nm]},
            ],
        })
    group = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {"ome": {"version": "0.5", "multiscales": [{
            "name": name,
            "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
            "datasets": datasets,
        }]}},
    }
    (root / "zarr.json").write_text(json.dumps(group, indent=2))
    return shapes


def open_scale(root, scale):
    """Open an existing scale array for reading and writing."""
    # Encoding threads default to every core of the node; keep them to the Slurm allocation.
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    context = ts.Context({"data_copy_concurrency": {"limit": int(cpus)}} if cpus else {})
    return ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(Path(root) / f"s{scale}")}},
                   open=True, context=context).result()


def shard_shape(arr):
    """Shard (write-chunk) shape of an opened array."""
    return tuple(int(v) for v in arr.chunk_layout.write_chunk.shape)


def shard_boxes(shape, shard):
    """All shard boxes of an array as tuples of (start, stop) per axis, in C order."""
    ranges = [range(0, n, s) for n, s in zip(shape, shard)]
    boxes = []
    for z in ranges[0]:
        for y in ranges[1]:
            for x in ranges[2]:
                boxes.append(tuple((a, min(a + s, n)) for a, s, n in zip((z, y, x), shard, shape)))
    return boxes
