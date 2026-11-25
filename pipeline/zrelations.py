"""Tile z-relations: the brightness relation of each tile to the same tile in the next selected slices,
for intensity's joint balance of tiles and slices (``intensity.balance_z_distance`` > 0).

After the intensity correction of Janelia's render (saalfeldlab/render, ``newsolver`` branch, run with
``--zDistance``), which goes back to TrakEM2's "Match intensities (layer-wise)": besides the overlaps
within a slice, every tile is matched to itself in the neighbouring slices, and all tiles of all slices
are balanced together. Written here from that description (no code taken from either).

Per selected slice z1, tile and distance d = 1 .. ``intensity.balance_z_distance``: the tile's preview
thumbnail at z1 and at z2, the d-th next selected slice of the same segment, each turned into the values
render corrects the tile to before its window, u (flat-field and, with patchgrid, the tile position's
shape, as ``patchgrid.tile_fields``, at thumbnail resolution): the values intensity's in-slice relations
are on. Only pixels that are tissue in both thumbnails count (pipeline/background.py; resin and
off-sample areas left out, as they don't follow the tissue). The two slices' tissue distributions are
matched by their percentiles ``QUANTILES`` (percentile matching: over a few 8 nm slices the tissue in a
tile hardly changes, the acquisition's brightness jumps do), and a line through the pairs (reduced major
axis, symmetric in the two slices) gives

    u_z2 ~ gain * u_z1 + offset,   centre = the mean of the z1 percentiles (where the relation is matched)

The uint8 thumbnails are dithered (a uniform step within each of their grey levels, a fixed random
pattern per (z, tile)) before the percentiles, so the percentiles don't snap to the thumbnail's grey
steps (~2-3 display greys on A1001). Pairs with less than ``min_tissue`` of the thumbnail tissue in both
slices are left out. The same relation on the raw thumbnail values (no flat-field, no shape) is kept as
gain_raw, offset_raw, centre_raw for comparison.

    run   (array over z-chunks of ``chunk_slices``): work/zrelations/chunks/chunk_<z0>-<z1>.csv, each with
          a .json of what it was measured from (settings, distance, thumbnails, shading, patchgrid)
    merge: work/zrelations/relations.csv (COLUMNS), relations.json (distance, settings, the shading and
           patchgrid in use) and qc/zrelations.png

Cheap (thumbnails only, ~0.1 s per tile slice): one task handles A1001 (487 slices x 9 tiles) in minutes;
35h- and 37i-sized runs (~18,000 slices) are spread over an array.
"""

import hashlib
import json
import logging
import sys
import zlib
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import background, patchgrid, shading  # noqa: E402
from .cli import atomic_write, base_parser, my_chunks, setup, task_info  # noqa: E402
from .config import deep_merge, qc_path, step_dir, step_path  # noqa: E402
from .preview import TILE_COLORS, chunk_name, z_chunks  # noqa: E402
from .slices import load_slices  # noqa: E402

log = logging.getLogger(__name__)

DEFAULTS = {
    "zrelations": {
        "min_tissue": 0.05,     # fraction of the thumbnail that must be tissue in both slices for a relation
        "chunk_slices": 100,    # selected z per array-task chunk (fixed blocks of global z)
    },
}

QUANTILES = np.arange(5.0, 95.0 + 1e-9, 2.5)   # percentiles of the tissue paired between the two slices
COLUMNS = ["z1", "z2", "tile", "segment", "distance", "gain", "offset", "centre", "n",
           "gain_raw", "offset_raw", "centre_raw"]


def match(ua, ub):
    """(gain, offset, centre) with ua ~ gain * ub + offset from the paired percentiles QUANTILES of the
    two samples (reduced major axis through the pairs; centre: the mean of ub's percentiles, where the
    line is exact whatever the gain's error), or None when either has no spread."""
    qa, qb = np.percentile(ua, QUANTILES), np.percentile(ub, QUANTILES)
    sa, sb = qa.std(), qb.std()
    if not (sa > 0 and sb > 0):
        return None
    gain = sa / sb
    centre = float(qb.mean())
    return float(gain), float(qa.mean() - gain * centre), centre


def dithered(t8, lo, hi, seed):
    """Raw values of a uint8 thumbnail autoscaled to [lo, hi], each pixel at a uniform random place
    within its grey level (preview.thumb_values puts every pixel at its level's centre)."""
    rng = np.random.default_rng(seed)
    return (lo + (np.asarray(t8, np.float64) + rng.random(t8.shape)) * ((hi - lo) / 255.0)).astype(np.float32)


def _seed(z, tile):
    return [int(z), zlib.crc32(str(tile).encode())]


def u_fields(key, tile_shape, thumb_shape, fields, shapes, R):
    """(gain, offset) arrays of the thumbnail's shape with u = gain * raw + offset as render corrects the
    tile (flat-field, then the patchgrid shape); None when neither is in use."""
    field = fields.get(key)
    if R is not None:
        gain, offset = patchgrid.tile_fields(shapes.get(key) or patchgrid.identity(1), R, field, tile_shape)
    elif field is not None:
        gain, offset = (1.0 / field).astype(np.float32), np.zeros(field.shape, np.float32)
    else:
        return None
    if gain.shape != tuple(thumb_shape):
        size = (thumb_shape[1], thumb_shape[0])
        gain = cv2.resize(gain, size, interpolation=cv2.INTER_LINEAR)
        offset = cv2.resize(offset, size, interpolation=cv2.INTER_LINEAR)
    return gain, offset


class Thumbs:
    """A tile slice's dithered raw thumbnail values, its corrected values u and its tissue mask."""

    def __init__(self, cfg):
        self.dir = step_path(cfg, "preview", "thumbs")
        idx = pd.read_csv(self.dir / "index.csv", dtype={"tile": str})
        stats = pd.read_csv(step_path(cfg, "preview", "stats.csv"), dtype={"tile": str})[["z", "tile", "p0_5", "p99_5"]]
        self.rows = idx.merge(stats, on=["z", "tile"]).set_index(["z", "tile"])
        self.fields = shading.load(cfg)
        self.shapes, self.R = patchgrid.load(cfg)
        self._npy, self._u = {}, {}

    def has(self, z, tile):
        return (z, tile) in self.rows.index

    def get(self, z, tile, key, tile_shape):
        """(raw values, u, tissue mask), float32 / bool."""
        r = self.rows.loc[(z, tile)]
        if r["npy"] not in self._npy:
            self._npy[r["npy"]] = np.load(self.dir / r["npy"], mmap_mode="r")
        t8 = np.asarray(self._npy[r["npy"]][int(r["i"])])
        v = dithered(t8, float(r["p0_5"]), float(r["p99_5"]), _seed(z, tile))
        k = (key, tile_shape, t8.shape)
        if k not in self._u:
            self._u[k] = u_fields(key, tile_shape, t8.shape, self.fields, self.shapes, self.R)
        f = self._u[k]
        u = v if f is None else f[0] * v + f[1]
        return v, u, ~background.mask(t8)

    def identity(self, keys):
        """Digest of the thumbnails (file, index, autoscale range) of (z, tile) ``keys``."""
        sub = self.rows.loc[sorted(k for k in keys if k in self.rows.index)]
        return hashlib.sha256(sub.reset_index().to_csv(index=False).encode()).hexdigest()


def neighbours(slices, distance):
    """{z1: [(d, z2), ...]}: for every selected z, the d-th next selected slices (d = 1 .. distance) of the
    same segment."""
    per_z = slices.drop_duplicates("z").sort_values("z")
    out = {}
    for _, g in per_z.groupby("segment"):
        zs = g["z"].tolist()
        for i, z in enumerate(zs):
            out[z] = [(d, zs[i + d]) for d in range(1, int(distance) + 1) if i + d < len(zs)]
    return out


def distance_of(cfg):
    d = int((cfg.get("intensity") or {}).get("balance_z_distance") or 0)
    if d < 1:
        raise ValueError("zrelations measures for intensity.balance_z_distance > 0 (it is 0: off)")
    return d


def measure_pair(a, b, min_px):
    """A row's relation fields (dict) between (raw, u, tissue) a at z2 and b at z1 of one tile, or None."""
    (va, ua, ta), (vb, ub, tb) = a, b
    if ua.shape != ub.shape:
        return None
    tissue = ta & tb
    n = int(tissue.sum())
    if n < min_px:
        return None
    rel, raw = match(ua[tissue], ub[tissue]), match(va[tissue], vb[tissue])
    if rel is None or raw is None:
        return None
    return {"gain": rel[0], "offset": rel[1], "centre": rel[2], "n": n,
            "gain_raw": raw[0], "offset_raw": raw[1], "centre_raw": raw[2]}


def _identity(cfg, zs, near, slices, thumbs):
    z_all = set(zs) | {z2 for z in zs for _, z2 in near.get(z, [])}
    sub = slices[slices["z"].isin(z_all)]
    keys = [(int(z), t) for z, t in zip(sub["z"], sub["tile"])]
    return {"settings": {**DEFAULTS["zrelations"], **(cfg.get("zrelations") or {})}, "distance": distance_of(cfg),
            "quantiles": [float(q) for q in QUANTILES], "thumbs": thumbs.identity(keys),
            "shading": shading.digest(cfg), "patchgrid": patchgrid.digest(cfg)}


def run(cfg, args):
    s = cfg["zrelations"]
    D = distance_of(cfg)
    slices = load_slices(cfg)
    near = neighbours(slices, D)
    thumbs = Thumbs(cfg)
    recs = {(int(r["z"]), r["tile"]): r for r in slices.to_dict("records")}
    tiles_of = slices.groupby("z")["tile"].apply(sorted).to_dict()
    mine = my_chunks(z_chunks(slices, s["chunk_slices"]), *task_info(args))
    log.info("%d chunks for this task (distance %d)", len(mine), D)
    for zs in mine:
        name = chunk_name(zs)
        out = step_dir(cfg, "zrelations", "chunks", f"chunk_{name}.csv")
        meta = out.with_suffix(".json")
        ident = _identity(cfg, zs, near, slices, thumbs)
        if out.exists() and meta.exists() and not args.overwrite and json.loads(meta.read_text()) == ident:
            log.info("chunk %s is current, skipping", name)
            continue
        out.unlink(missing_ok=True)
        rows, cache = [], {}

        def values(z, tile):
            if (z, tile) not in cache:
                r = recs[(z, tile)]
                cache[(z, tile)] = thumbs.get(z, tile, (int(r["segment"]), tile), (int(r["height"]), int(r["width"])))
            return cache[(z, tile)]

        for z1 in zs:
            for tile in [t for t in tiles_of.get(z1, []) if thumbs.has(z1, t)]:
                for d, z2 in near.get(z1, []):
                    if (z2, tile) not in recs or not thumbs.has(z2, tile):
                        continue
                    a, b = values(z2, tile), values(z1, tile)
                    min_px = float(s["min_tissue"]) * a[0].size
                    got = measure_pair(a, b, min_px)
                    if got is not None:
                        rows.append({"z1": z1, "z2": z2, "tile": tile, "segment": int(recs[(z1, tile)]["segment"]),
                                     "distance": d, **got})
            for k in [k for k in cache if k[0] <= z1]:
                del cache[k]
        atomic_write(meta, lambda p: Path(p).write_text(json.dumps(ident)))
        atomic_write(out, lambda p: pd.DataFrame(rows, columns=COLUMNS).to_csv(p, index=False, float_format="%.8g"))
        log.info("chunk %s: %d relations over %d slices", name, len(rows), len(zs))
    return 0


def merge(cfg, args):
    s = cfg["zrelations"]
    D = distance_of(cfg)
    slices = load_slices(cfg)
    near = neighbours(slices, D)
    thumbs = Thumbs(cfg)
    parts, bad = [], []
    for zs in z_chunks(slices, s["chunk_slices"]):
        name = chunk_name(zs)
        path = step_path(cfg, "zrelations", "chunks", f"chunk_{name}.csv")
        meta = path.with_suffix(".json")
        if (not path.exists() or not meta.exists()
                or json.loads(meta.read_text()) != _identity(cfg, zs, near, slices, thumbs)):
            bad.append(name)
            continue
        parts.append(pd.read_csv(path, dtype={"tile": str}))
    if bad:
        raise RuntimeError(f"{len(bad)} z-relation chunks missing or out of date (first: {bad[0]}); run 'zrelations run'")
    rel = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLUMNS)
    atomic_write(step_dir(cfg, "zrelations", "relations.csv"), lambda p: rel.to_csv(p, index=False, float_format="%.8g"))
    info = {"distance": D, "settings": {**DEFAULTS["zrelations"], **(cfg.get("zrelations") or {})},
            "shading": shading.digest(cfg), "patchgrid": patchgrid.digest(cfg), "relations": int(len(rel))}
    atomic_write(step_dir(cfg, "zrelations", "relations.json"), lambda p: Path(p).write_text(json.dumps(info, indent=1)))
    plot(rel, qc_path(cfg, "zrelations.png"))
    log.info("%d z-relations (distance up to %d) over %d slices", len(rel), D, rel["z1"].nunique() if len(rel) else 0)
    return 0


def read(cfg, distance):
    """work/zrelations/relations.csv, the relations up to ``distance``; raises when it is missing, measured
    for a shorter distance, or warns when measured with other flat-fields or shapes than in use now."""
    path = step_path(cfg, "zrelations", "relations.csv")
    info_path = path.with_suffix(".json")
    if not path.exists() or not info_path.exists():
        raise FileNotFoundError(f"{path} not found: intensity.balance_z_distance > 0 needs the zrelations step "
                                "(run, merge)")
    info = json.loads(info_path.read_text())
    if int(info.get("distance", 0)) < int(distance):
        raise ValueError(f"{path} holds relations up to distance {info.get('distance')}, intensity.balance_z_distance "
                         f"is {distance}: re-run zrelations (run, merge)")
    for name, now in (("shading", shading.digest(cfg)), ("patchgrid", patchgrid.digest(cfg))):
        if info.get(name) != now:
            log.warning("%s was measured with other %s fields than in use now: re-run zrelations", path, name)
    rel = pd.read_csv(path, dtype={"tile": str})
    return rel[rel["distance"] <= int(distance)].reset_index(drop=True)


def plot(rel, path):
    """Per tile: each slice-to-next relation's change at its centre (% of the centre), on the corrected
    values and on the raw thumbnails."""
    if rel.empty:
        return
    r1 = rel[rel["distance"] == 1]
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 7), layout="constrained")
    for i, (tile, g) in enumerate(r1.groupby("tile")):
        style = {"color": TILE_COLORS[i % 8], "ls": ["-", "--", ":"][i // 8 % 3], "lw": 0.6}
        change = 100 * ((g["gain"] - 1) * g["centre"] + g["offset"]) / g["centre"]
        axes[0].plot(g["z1"], change, label=tile, **style)
        axes[1].plot(g["z1"], 100 * (g["gain"] - 1), **style)
    axes[0].set_ylabel("change at centre to next slice (%)")
    axes[1].set_ylabel("gain to next slice - 1 (%)")
    axes[1].set_xlabel("z")
    axes[0].legend(title="tile", fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Each tile against itself in the next slice (tissue, percentile matched, corrected values u)")
    atomic_write(path, lambda p: fig.savefig(p, dpi=90))
    plt.close(fig)


def main(argv=None):
    from . import intensity
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("command", choices=["run", "merge"])
    args = p.parse_args(argv)
    defaults = DEFAULTS
    for d in (intensity.DEFAULTS, shading.DEFAULTS, patchgrid.DEFAULTS):
        defaults = deep_merge(defaults, d)
    cfg = setup(args, defaults)
    return run(cfg, args) if args.command == "run" else merge(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
