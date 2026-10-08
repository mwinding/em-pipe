"""Synthetic FIB-SEM datasets with known ground truth, written like the real exports.

A smooth random 3D volume is sampled through a grid of overlapping tiles that drifts
from slice to slice. Each tile is written per calendar day as an ImageJ stack named
``M{MM}_D{DD}_tile{r}-{c}[_partN].tif`` with per-slice labels
``G460-0186_{yy-mm-dd}_{HHMMSS}_0-{r}-{c}_InLens_raw.tif``.

Ground truth (in the conventions of docs/design.md, with filename r = y, c = x):
- stitch: tile (r, c) pixel -> montage is a translation by ``tile_origin[tile]``
- align: montage of slice z -> volume frame is a translation by ``drift[z]`` (+ a constant)
"""

import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import tifffile
import yaml
from scipy import ndimage

LABEL = "G460-0186_{ts:%y-%m-%d_%H%M%S}_0-{r}-{c}_InLens_raw.tif"


@dataclass
class Synth:
    raw_dir: Path
    timestamps: list            # datetime per z
    tiles: list                 # ["0-0", "0-1", ...]
    tile_shape: tuple           # (height, width)
    tile_origin: dict           # tile -> (x, y) in montage coordinates
    montage_shape: tuple        # (height, width)
    drift: np.ndarray           # (Z, 2) integer (dx, dy) per z
    margin: int
    volume: np.ndarray          # float32 [Zv, H, W], values in [0, 1]
    z_positions: np.ndarray     # axial sampling position of each slice (volume voxel units)
    files: dict = field(default_factory=dict)   # filename -> list of z it contains

    def montage(self, z):
        """True content of slice z in montage coordinates (float, [0, 1])."""
        dx, dy = self.drift[z]
        H, W = self.montage_shape
        y0, x0 = self.margin + dy, self.margin + dx
        return _sample_z(self.volume, self.z_positions[z])[y0:y0 + H, x0:x0 + W]


def _sample_z(volume, pos):
    i = int(np.floor(pos))
    f = pos - i
    if f < 1e-9 or i + 1 >= len(volume):
        return volume[min(i, len(volume) - 1)]
    return (1 - f) * volume[i] + f * volume[i + 1]


def make_volume(shape, seed=0, sigma=(1.0, 2.0, 2.0), n_blobs=None):
    """Smooth noise plus sparse blobs, normalised to [0, 1]. Adjacent z are similar."""
    rng = np.random.default_rng(seed)
    vol = ndimage.gaussian_filter(rng.standard_normal(shape).astype(np.float32), sigma)
    vol /= vol.std() + 1e-9
    zz, yy, xx = shape
    n_blobs = n_blobs if n_blobs is not None else max(1, (yy * xx) // 900)
    blobs = np.zeros(shape, np.float32)
    for _ in range(n_blobs):
        cz, cy, cx = rng.uniform(0, zz), rng.uniform(0, yy), rng.uniform(0, xx)
        blobs[min(int(cz), zz - 1), min(int(cy), yy - 1), min(int(cx), xx - 1)] = rng.choice([-1, 1]) * 40
    vol += ndimage.gaussian_filter(blobs, (1.5, 3.0, 3.0))
    vol -= vol.min()
    vol /= vol.max() + 1e-9
    return vol


def make_dataset(root, *, n_slices=24, grid=(2, 2), tile_shape=(192, 224), overlap=(24, 28),
                 start=datetime(2026, 9, 24, 23, 35, 0), interval_s=142.0, drift_step=1.5,
                 drift_jumps=None, gaps=None, parts=None, z_positions=None, tile_gain=None,
                 intensity_drift=0.0, streaks=0.0, noise=0.01, seed=0, faults=None,
                 lo=26000, hi=41000):
    """Write a synthetic dataset under ``root`` and return its ground truth.

    gaps:   {z: extra_seconds} time gap inserted before slice z (no file split)
    parts:  [z, ...] acquisition restarts: a new ``_partN`` file starts at z (also adds a gap)
    drift_jumps: {z: (dx, dy)} extra jump in drift at z
    z_positions: axial sampling position per slice (default 0, 1, 2, ...)
    tile_gain: {tile: gain} multiplicative intensity per tile
    streaks: amplitude of vertical curtaining stripes (fraction of range)
    faults: {"duplicate": [(day_prefix, dst_tile, src_tile)],
             "truncate": [(day_prefix, tile, n_bytes_removed)],
             "mislabel": [(day_prefix, tile, label_tile)],
             "missing":  [(day_prefix, tile)]}
      day_prefix like "M09_D25" (part suffix is matched as part of the file name).
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + 1)
    R, C = grid
    th, tw = tile_shape
    oy, ox = overlap
    H, W = R * th - (R - 1) * oy, C * tw - (C - 1) * ox
    tiles = [f"{r}-{c}" for r in range(R) for c in range(C)]
    tile_origin = {f"{r}-{c}": (c * (tw - ox), r * (th - oy)) for r in range(R) for c in range(C)}

    steps = np.rint(rng.normal(0, drift_step, (n_slices, 2))).astype(int)
    steps[0] = 0
    for z, (dx, dy) in (drift_jumps or {}).items():
        steps[z] += (dx, dy)
    drift = np.cumsum(steps, axis=0)
    drift -= drift.min(axis=0)
    margin = int(np.abs(drift).max()) + 4

    z_positions = np.arange(n_slices, dtype=float) if z_positions is None else np.asarray(z_positions, float)
    nz = int(np.ceil(z_positions.max())) + 2
    volume = make_volume((nz, H + 2 * margin + drift[:, 1].max(), W + 2 * margin + drift[:, 0].max()), seed)

    parts = sorted(parts or [])
    gaps = dict(gaps or {})
    for z in parts:
        gaps[z] = gaps.get(z, 0) + 300
    timestamps, t = [], start
    for z in range(n_slices):
        if z > 0:
            t = t + timedelta(seconds=interval_s + gaps.get(z, 0) + float(rng.uniform(-2, 2)))
        timestamps.append(t.replace(microsecond=0))

    truth = Synth(root, timestamps, tiles, tile_shape, tile_origin, (H, W), drift, margin,
                  volume, z_positions)

    # Group z by (calendar day, part number).
    part_of = np.searchsorted(parts, np.arange(n_slices), side="right") if parts else np.zeros(n_slices, int)
    groups = {}
    for z, ts in enumerate(timestamps):
        groups.setdefault((ts.date(), int(part_of[z])), []).append(z)
    days = {}
    for (day, part), zs in groups.items():
        days.setdefault(day, []).append((part, zs))

    gains = tile_gain or {}
    stripe = None
    if streaks:
        stripe = (rng.standard_normal(tw) * streaks).astype(np.float32)
        stripe = ndimage.gaussian_filter1d(stripe, 0.7)
    for day, plist in sorted(days.items()):
        prefix = f"M{day.month:02d}_D{day.day:02d}"
        for pi, (part, zs) in enumerate(sorted(plist)):
            suffix = f"_part{pi + 1}" if len(plist) > 1 else ""
            for tile in tiles:
                r, c = map(int, tile.split("-"))
                x0, y0 = tile_origin[tile]
                stack = np.empty((len(zs), th, tw), np.float32)
                for i, z in enumerate(zs):
                    m = truth.montage(z)[y0:y0 + th, x0:x0 + tw]
                    img = m * gains.get(tile, 1.0) * (1 + intensity_drift * z / max(n_slices - 1, 1))
                    if stripe is not None:
                        img = img + stripe[None, :]
                    img = img + rng.normal(0, noise, img.shape)
                    stack[i] = img
                data = np.clip(lo + stack * (hi - lo), 0, 65535).astype(np.uint16)
                labels = [LABEL.format(ts=timestamps[z], r=r, c=c) for z in zs]
                name = f"{prefix}_tile{tile}{suffix}.tif"
                write_imagej(root / name, data, labels)
                truth.files[name] = list(zs)

    _apply_faults(root, faults or {}, truth)
    return truth


def write_imagej(path, data, labels, voxel_um=0.008):
    tifffile.imwrite(path, data, imagej=True, byteorder=">", resolution=(1 / voxel_um, 1 / voxel_um),
                     metadata={"spacing": voxel_um, "unit": "micron", "Labels": labels})


def _match(root, prefix, tile):
    hits = sorted(p for p in root.glob(f"{prefix}_tile{tile}*.tif"))
    if not hits:
        raise ValueError(f"no file for {prefix} tile {tile}")
    return hits


def _apply_faults(root, faults, truth):
    for prefix, dst, src in faults.get("duplicate", []):
        for src_path in _match(root, prefix, src):
            dst_path = Path(str(src_path).replace(f"_tile{src}", f"_tile{dst}"))
            shutil.copyfile(src_path, dst_path)
    for prefix, tile, n in faults.get("truncate", []):
        for path in _match(root, prefix, tile):
            size = path.stat().st_size
            with open(path, "r+b") as fh:
                fh.truncate(size - n)
    for prefix, tile, label_tile in faults.get("mislabel", []):
        for path in _match(root, prefix, tile):
            with tifffile.TiffFile(path) as tif:
                data = tif.asarray()
                labels = tif.imagej_metadata["Labels"]
            r, c = label_tile.split("-")
            labels = [l.replace(f"_0-{tile}_", f"_0-{r}-{c}_") for l in labels]
            write_imagej(path, data, labels)
    for prefix, tile in faults.get("missing", []):
        for path in _match(root, prefix, tile):
            path.unlink()
            truth.files.pop(path.name, None)


def write_config(path, raw_dir, output_dir, **sections):
    """Write a dataset YAML for tests. ``sections`` are merged in as top-level keys."""
    cfg = {"name": "synthetic", "raw_dir": str(raw_dir), "output_dir": str(output_dir), **sections}
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False))
    return Path(path)
