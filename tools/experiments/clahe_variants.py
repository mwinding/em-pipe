"""CLAHE on top of the current pipeline's 32 nm test slices (experiment, Oct 2026).

    python tools/experiments/clahe_variants.py SRC_DIR QC_DIR Z[,Z...]

SRC_DIR holds z<z>.tif stacks from tools/shading_before_after.py (frame 0: the original volume,
frame 1: the current pipeline; 32 nm, 0 = no data). For every clip limit in CLIPS and CLAHE tile
size in TILES (full-resolution px, so tile / 4 px at 32 nm) this writes
QC_DIR/clahe_<clip>_<tile>/z<z>.tif = [frame 0 unchanged, render._clahe(frame 1)], and
QC_DIR/clahe_before_2.0_512/z<z>.tif = [frame 0, render._clahe(frame 0)] (CLAHE alone on the
original volume). Valid pixels: the non-zero pixels with their holes filled, i.e. the slice's
footprint (tissue clipped to 0 inside the slice is data, as in the render's coverage mask).
In an array job SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT split the z list.
"""
import os
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline import render  # noqa: E402

CLIPS = (1.5, 2.0, 3.0)
TILES = (256, 512, 1024)
F = 4   # 32 nm planes: full-resolution px per plane px


def footprint(img):
    """Valid pixels of a plane where 0 = no data: the non-zero pixels with their holes filled."""
    return ndimage.binary_fill_holes(img > 0)


def clahe(img, clip, tile_px):
    """render._clahe of a 32 nm plane with a CLAHE tile of tile_px full-resolution px."""
    return render._clahe(img.copy(), footprint(img), clip, max(1.0, tile_px / F))


def write(path, frames, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.tif")
    tifffile.imwrite(tmp, np.stack(frames), imagej=True, resolution=(1 / 0.032, 1 / 0.032),
                     metadata={"spacing": 0.032, "unit": "micron", "axes": "ZYX", "Labels": labels})
    os.replace(tmp, path)


def main():
    src, qc = Path(sys.argv[1]), Path(sys.argv[2])
    zs = [int(v) for v in sys.argv[3].split(",")]
    k, n = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)), int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    for z in zs[k::n]:
        before, cur = tifffile.imread(src / f"z{z}.tif")[:2]
        for clip in CLIPS:
            for tile in TILES:
                write(qc / f"clahe_{clip}_{tile}" / f"z{z}.tif", [before, clahe(cur, clip, tile)],
                      [f"z {z}: original volume", f"z {z}: current pipeline + CLAHE clip {clip}, tile {tile} px (at 32 nm)"])
        write(qc / "clahe_before_2.0_512" / f"z{z}.tif", [before, clahe(before, 2.0, 512)],
              [f"z {z}: original volume", f"z {z}: original volume + CLAHE clip 2.0, tile 512 px (at 32 nm)"])
        print("done", z, flush=True)


if __name__ == "__main__":
    main()
