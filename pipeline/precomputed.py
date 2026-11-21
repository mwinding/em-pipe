"""Export a rendered volume (OME-Zarr, every pyramid scale) as a Neuroglancer precomputed volume, into a local
folder or straight into a Google Cloud Storage bucket.

Precomputed is what CloudVolume and the segmentation tools built on it read. ``init`` plans the export: the
``info`` (one scale per zarr scale: size, resolution, chunk size, encoding, sharding) and ``provenance`` are kept
in the work folder (output_dir/work/precomputed/<name>/) until the end. ``run`` (array) writes whole shards, each
read from the zarr in one block, so tasks never share a file; every written (or empty) shard is recorded in the
work folder, so re-runs continue, and a shard recorded but missing at the destination is written again.
``verify`` checks that every shard was written and is there, compares random chunk-sized blocks of every scale
with the zarr, and only then writes ``info`` and ``provenance`` to the destination: a volume only looks complete
when it is.

Destination (``precomputed.destination``): null, a folder output_dir/<name>; ``gs://bucket/prefix``, written
directly by tensorstore (no local copy, no upload step), with the service-account key at
``precomputed.credentials`` (handed over as GOOGLE_APPLICATION_CREDENTIALS unless that is set; never read here);
or ``file:///path``. A destination that already holds a volume is only replaced with ``init --overwrite``: a local
folder is deleted first; a bucket's files are rewritten in place (only when its info is identical, i.e. the same
geometry and settings), and shards that have no data now are deleted.

Sharded (default): one file per 2^``shard_chunks_log2`` chunks (64: a 4 x 4 x 4 block of 128^3
chunks, 512^3 voxels), so a volume of terabytes is tens of thousands of files instead of tens of
millions of chunks. With an identity hash, a shard holds an aligned range of compressed Morton
codes, i.e. an aligned block of chunks (see ``shard_blocks``).
"""

import json
import logging
import math
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorstore as ts

from . import omezarr
from .cli import atomic_write, base_parser, key_file, my_chunks, setup, task_info
from .render import DEFAULTS as RENDER_DEFAULTS, volume_paths

log = logging.getLogger(__name__)

DEFAULTS = {
    "precomputed": {
        "enabled": False,           # run_pipeline.sh: export after pyramid
        "name": None,               # folder in output_dir (and work folder name); null: <volume stem>.precomputed
        "destination": None,        # null: output_dir/<name>; gs://bucket/prefix (written directly); file:///path
        "credentials": "~/.cloudvolume/secrets/google-secret.json",   # gs:// destinations: service-account key
        "chunk": [128, 128, 128],   # chunk size, x y z voxels
        "encoding": "raw",          # raw (lossless, gzip in the shards) | jpeg (lossy, smaller)
        "jpeg_quality": 90,
        "sharded": True,
        "shard_chunks_log2": 6,     # chunks per shard = 2^this (6: 4 x 4 x 4)
        "minishard_chunks_log2": 3,  # chunks per minishard = 2^this
        "scales": None,             # zarr scales to export, e.g. [0, 1, 2]; null: all
        "description": "",          # provenance
        "owners": [],               # provenance, e.g. ["name@example.org"]
    }
}
ENCODINGS = ("raw", "jpeg")


def paths(cfg):
    """(zarr volume, local precomputed folder: the default destination)."""
    zarr_root, _ = volume_paths(cfg)
    name = cfg["precomputed"]["name"] or zarr_root.name.removesuffix(".zarr").removesuffix(".ome") + ".precomputed"
    return zarr_root, Path(cfg["output_dir"]) / name


def work_dir(cfg):
    """The export's plan and progress records."""
    return Path(cfg["output_dir"]) / "work" / "precomputed" / paths(cfg)[1].name


class Destination:
    """Where the export goes, as a tensorstore key-value store: a local folder, file:///path or gs://bucket/prefix.
    Keys are relative to the volume (``info``, ``8_8_8/0a1.shard``)."""

    def __init__(self, where, credentials=None):
        self.url = str(where)
        self.local = None   # a plain folder (the default): init --overwrite may delete it
        if self.url.startswith("gs://"):
            bucket, _, prefix = self.url.removeprefix("gs://").partition("/")
            if not bucket:
                raise ValueError(f"precomputed.destination {self.url!r}: no bucket")
            prefix = prefix.strip("/")
            self.spec = {"driver": "gcs", "bucket": bucket, "path": prefix + "/" if prefix else ""}
            if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_file(credentials, "precomputed.credentials"))
        elif "://" in self.url and not self.url.startswith("file://"):
            raise ValueError(f"precomputed.destination must be a folder, file:///path or gs://bucket/prefix, "
                             f"not {self.url!r}")
        else:
            path = self.url.removeprefix("file://")
            if not self.url.startswith("file://"):
                self.local = Path(path)
            self.spec = {"driver": "file", "path": str(path).rstrip("/") + "/"}
        self._kv = None

    @property
    def kv(self):
        if self._kv is None:
            self._kv = ts.KvStore.open(self.spec).result()
        return self._kv

    def read(self, key):
        r = self.kv.read(key).result()
        return r.value if r.state == "value" else None

    def write(self, key, data):
        self.kv.write(key, data).result()

    def delete(self, key):
        self.kv.delete_range(ts.KvStore.KeyRange(inclusive_min=key, exclusive_max=key + "\x00")).result()

    def keys(self, prefix):
        """Every key starting with ``prefix``."""
        end = prefix[:-1] + chr(ord(prefix[-1]) + 1)
        return {k.decode() for k in self.kv.list(ts.KvStore.KeyRange(inclusive_min=prefix, exclusive_max=end)).result()}

    def store(self, sc, write=False):
        """One scale as a tensorstore array (x, y, z, channel), from the planned metadata: no info file needed."""
        meta = {k: sc[k] for k in ("key", "size", "resolution", "voxel_offset", "encoding", "sharding", "jpeg_quality")
                if k in sc}
        meta["chunk_size"] = sc["chunk_sizes"][0]
        spec = {"driver": "neuroglancer_precomputed", "kvstore": self.spec,
                "multiscale_metadata": {"type": "image", "data_type": "uint8", "num_channels": 1},
                "scale_metadata": meta}
        return ts.open(spec, open=True, assume_metadata=True, read=True, write=write).result()


def destination(cfg):
    p = cfg["precomputed"]
    return Destination(p["destination"] or paths(cfg)[1], p["credentials"])


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


def _plan(wdir):
    f = Path(wdir) / "export.json"
    return json.loads(f.read_text()) if f.exists() else None


def _rendered(cfg):
    """Which render of the volume this is (render.json's digest and creation stamp), or None if it has none:
    a volume re-rendered with the same geometry is a different source, and its export starts again."""
    f = volume_paths(cfg)[1] / "render.json"
    if not f.exists():
        return None
    meta = json.loads(f.read_text())
    return {"digest": meta.get("digest"), "created": meta.get("created")}


def _records(wdir, export_id):
    """{(scale index, shard number): "data" | "empty"} recorded by run tasks for this export."""
    out = {}
    for f in sorted((Path(wdir) / "done").glob("*.txt")):
        for line in f.read_text().splitlines():
            parts = line.split()
            if len(parts) == 4 and parts[0] == export_id:
                out[(int(parts[1]), int(parts[2]))] = parts[3]
    return out


def init(cfg, overwrite=False):
    p = cfg["precomputed"]
    _check(p)
    zarr_root, _ = paths(cfg)
    if not (zarr_root / "zarr.json").exists():
        log.error("%s not found: run render and pyramid first", zarr_root)
        return 1
    info, zarr_scales = make_info(zarr_root, p)
    dest = destination(cfg)
    wdir = work_dir(cfg)
    plan = _plan(wdir)
    source = _rendered(cfg)
    if plan and not overwrite:
        if plan["info"] == info and plan["destination"] == dest.url and plan.get("source") == source:
            log.info("export of %s to %s already planned with these settings: kept", zarr_root.name, dest.url)
            return 0
        log.error("%s plans an export of another render, with other settings or to another destination: re-run "
                  "init with --overwrite", wdir / "export.json")
        return 1
    have = dest.read("info")
    if have is not None:
        if not overwrite:
            log.error("%s already holds a volume: re-run init with --overwrite to replace it (%s)", dest.url,
                      "the folder is deleted first" if dest.local else "its files are rewritten in place")
            return 1
        if dest.local is None and json.loads(have) != info:
            log.error("%s holds a volume of another size or other export settings: replacing it in place would "
                      "leave some of its files behind; remove it first or choose another destination", dest.url)
            return 1
    replace = have is not None
    if overwrite and dest.local is not None and dest.local.exists():
        shutil.rmtree(dest.local)
        replace = False
    shutil.rmtree(wdir / "done", ignore_errors=True)
    (wdir / "done").mkdir(parents=True, exist_ok=True)
    prov = {"description": p["description"] or f"em-pipe export of {zarr_root.name}",
            "owners": list(p["owners"]), "sources": [str(zarr_root)],
            "processing": [{"method": {"task": "em-pipe precomputed", "zarr_scales": zarr_scales,
                                       "encoding": p["encoding"], "chunk": p["chunk"]},
                            "date": datetime.now(timezone.utc).isoformat(timespec="seconds")}]}
    plan = {"id": uuid.uuid4().hex, "destination": dest.url, "replace": replace, "source": source,
            "zarr_scales": zarr_scales, "info": info, "provenance": prov}
    atomic_write(wdir / "export.json", lambda t: Path(t).write_text(json.dumps(plan, indent=1)))
    for sc in info["scales"]:
        n = len(shard_blocks([math.ceil(a / b) for a, b in zip(sc["size"], sc["chunk_sizes"][0])],
                             int(p["shard_chunks_log2"]))) if "sharding" in sc else None
        log.info("scale %s: size %s, %s", sc["key"], sc["size"], f"{n} shards" if n else "unsharded")
    log.info("export to %s planned%s; verify writes info and provenance there last", dest.url,
             " (replacing the volume there in place)" if replace else "")
    return 0


def _units(info, p):
    """[(scale index, shard number, chunk origin, block)] over every scale."""
    out = []
    for i, sc in enumerate(info["scales"]):
        grid = [math.ceil(a / b) for a, b in zip(sc["size"], sc["chunk_sizes"][0])]
        k = int(p["shard_chunks_log2"]) if "sharding" in sc else 6   # unsharded: same blocks as work units
        out += [(i, s, o, b) for s, o, b in shard_blocks(grid, k)]
    return out


def _loaded(cfg):
    """(plan, destination) after checks that the plan matches the settings and volume, else None."""
    p = cfg["precomputed"]
    zarr_root, _ = paths(cfg)
    wdir = work_dir(cfg)
    plan = _plan(wdir)
    if plan is None:
        log.error("%s not found: run 'precomputed init' first", wdir / "export.json")
        return None
    dest = destination(cfg)
    if (make_info(zarr_root, p)[0] != plan["info"] or dest.url != plan["destination"]
            or plan.get("source") != _rendered(cfg)):
        log.error("the settings, volume or destination changed since %s was written: run 'precomputed init "
                  "--overwrite'", wdir / "export.json")
        return None
    return plan, dest


def _box(origin, block, sc):
    c = sc["chunk_sizes"][0]
    lo = [o * cc for o, cc in zip(origin, c)]
    hi = [min((o + b) * cc, n) for o, b, cc, n in zip(origin, block, c, sc["size"])]
    return lo, hi


def run(cfg, task_id, num_tasks, overwrite=False):
    p = cfg["precomputed"]
    loaded = _loaded(cfg)
    if loaded is None:
        return 1
    plan, dest = loaded
    zarr_root, _ = paths(cfg)
    wdir = work_dir(cfg)
    info = plan["info"]
    units = my_chunks(_units(info, p), task_id, num_tasks)
    log.info("task %d/%d: %d shards to %s", task_id, num_tasks, len(units), dest.url)
    recorded = {} if overwrite else _records(wdir, plan["id"])
    present, stores, sources = {}, {}, {}
    done = skipped = empty = 0
    with open(wdir / "done" / f"task{task_id:04d}-of-{num_tasks:04d}.txt", "a", buffering=1) as rec:
        for i, s, origin, block in units:
            sc = info["scales"][i]
            key = shard_file(sc["key"], sc["sharding"], s) if "sharding" in sc else None
            state = recorded.get((i, s))
            if state == "empty" or (state == "data" and key is None):
                skipped += 1
                continue
            if state == "data":   # recorded: skip it if it is (still) at the destination
                if i not in present:
                    present[i] = dest.keys(sc["key"] + "/")
                if key in present[i]:
                    skipped += 1
                    continue
            if i not in stores:
                stores[i] = dest.store(sc, write=True)
                sources[i] = omezarr.open_scale(zarr_root, plan["zarr_scales"][i])
            lo, hi = _box(origin, block, sc)
            data = sources[i][lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]].read().result()
            if not data.any():   # no data here (outside the imaged area): nothing to store, reads as zeros
                if key is not None and plan["replace"]:
                    dest.delete(key)   # the replaced volume's shard, if any
                rec.write(f"{plan['id']} {i} {s} empty\n")
                empty += 1
                continue
            txn = ts.Transaction()
            stores[i].with_transaction(txn)[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2], 0].write(
                np.ascontiguousarray(np.transpose(data, (2, 1, 0)))).result()
            txn.commit_async().result()
            rec.write(f"{plan['id']} {i} {s} data\n")
            done += 1
    log.info("task %d: wrote %d shards, %d with no data (not stored), %d already done", task_id, done, empty, skipped)
    return 0


def verify(cfg, samples=24, seed=0):
    """Every shard written by run and present at the destination (all-zero blocks are not stored and read as
    zeros: outside the imaged area; checked against the zarr), random chunk-sized blocks of every scale equal to
    the zarr; then info and provenance are written to the destination, last."""
    p = cfg["precomputed"]
    loaded = _loaded(cfg)
    if loaded is None:
        return 1
    plan, dest = loaded
    zarr_root, _ = paths(cfg)
    info = plan["info"]
    recorded = _records(work_dir(cfg), plan["id"])
    rng = np.random.default_rng(seed)
    bad = 0
    for i, sc in enumerate(info["scales"]):
        units = [u for u in _units(info, p) if u[0] == i]
        todo = [s for _, s, _, _ in units if (i, s) not in recorded]
        if todo:
            log.error("scale %s: %d of %d shards not written yet (run every task)", sc["key"], len(todo), len(units))
            bad += 1
            continue
        src = omezarr.open_scale(zarr_root, plan["zarr_scales"][i])
        if "sharding" in sc:
            present = dest.keys(sc["key"] + "/")
            missing = [s for _, s, _, _ in units if recorded[(i, s)] == "data"
                       and shard_file(sc["key"], sc["sharding"], s) not in present]
            left = [s for _, s, _, _ in units if recorded[(i, s)] == "empty"
                    and shard_file(sc["key"], sc["sharding"], s) in present]
            if missing or left:
                if missing:
                    log.error("scale %s: %d shard files missing at the destination, e.g. %s (run again)",
                              sc["key"], len(missing), missing[:5])
                if left:
                    log.error("scale %s: %d shards with no data still have a file at the destination, e.g. %s",
                              sc["key"], len(left), left[:5])
                bad += 1
                continue
        lost = []
        for _, s, origin, block in units:   # cheap where the zarr has no data: its empty chunks aren't stored either
            if recorded[(i, s)] == "empty":
                lo, hi = _box(origin, block, sc)
                if src[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]].read().result().any():
                    lost.append(s)
        if lost:
            log.error("scale %s: %d shards recorded as empty have data in the zarr, e.g. %s (init --overwrite)",
                      sc["key"], len(lost), lost[:5])
            bad += 1
            continue
        n_empty = sum(1 for _, s, _, _ in units if recorded[(i, s)] == "empty")
        if n_empty:
            log.info("scale %s: %d shards with no data in the zarr (not stored, read as zeros)", sc["key"], n_empty)
        store = dest.store(sc)
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
    if bad:
        log.error("%s: not complete; info and provenance not written", dest.url)
        return 1
    dest.write("info", json.dumps(info, indent=1).encode())
    dest.write("provenance", json.dumps(plan["provenance"], indent=1).encode())
    log.info("%s: verified; info and provenance written, the volume is complete", dest.url)
    return 0


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
