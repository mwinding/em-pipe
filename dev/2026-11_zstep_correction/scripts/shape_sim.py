"""Does shape_measure's anisotropy read a known stretch along z? A synthetic check.

    python dev/2026-11_zstep_correction/scripts/shape_sim.py

An isotropic random texture (blobs of a few voxels plus pixel noise) is resampled along z as slices of a given
thickness relative to the voxel would record it: slices 0.8x as thick draw structures out along z by 1 / 0.8.
Prints the median anisotropy (L_z / sqrt(L_x L_y)) for each thickness; it should follow 1 / thickness.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shape_measure  # noqa: E402

rng = np.random.default_rng(0)
base = ndimage.gaussian_filter(rng.standard_normal((160, 400, 400)).astype(np.float32), 3.0)
base = base / base.std()
for thickness in (1.0, 0.8, 1.25):
    n = int(base.shape[0] / thickness)
    v = ndimage.map_coordinates(base, np.meshgrid(np.arange(n) * thickness, np.arange(400), np.arange(400), indexing="ij"),
                                order=1)
    v = np.clip(128 + 40 * v + 12 * rng.standard_normal(v.shape), 0, 255).astype(np.uint8)
    g, m = shape_measure.band_pass(v, 2.0, 6.0)
    lx, ly, lz = shape_measure.lengths(g)
    aniso = (lz / np.sqrt(lx * ly))[m:-m]
    print(f"slices {thickness:.2f}x the voxel: L_x {np.nanmedian(lx):.2f} L_y {np.nanmedian(ly):.2f} L_z {np.nanmedian(lz):.2f}, "
          f"anisotropy {np.nanmedian(aniso):.3f} (expected {1 / thickness:.3f})")
