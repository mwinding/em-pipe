"""CLAHE variants side by side on regions of a rendered (no-CLAHE) volume, as ImageJ TIFF stacks.

Columns: no CLAHE | current 2D (render._clahe: clip 2, 512 px) | background-masked 2D (clip 2, 512)
| masked, gentler (clip 1.5, 1024) | masked, 3D-consistent (histograms pooled over +-POOL slices,
clip 2, 512). Background = large uniform areas (bright resin, black no-data or off-sample): they are
left out of the CLAHE histograms. CLAHE runs on a CONTEXT px square around each region, cropped to
CROP px. Writes <volume folder>/qc/clahe_compare/region_*.tif.
"""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile
from scipy import ndimage

sys.path.insert(0, os.path.expandvars("${EM_PIPE_ROOT}"))
from pipeline import omezarr, render  # noqa: E402

VOL = Path(sys.argv[1])
OUT = VOL.parent / "qc" / "clahe_compare"
OUT.mkdir(parents=True, exist_ok=True)
REGIONS = {"A_lumen_edge": (11760, 11200), "B_sample_edge": (23200, 5200),
           "C_interior": (6400, 19200), "D_nodata_edge": (27040, 16000)}   # centre (x, y), full-res px
Z0, NZ, POOL = 590, 21, 16
CROP, CONTEXT, BAR = 2048, 4096, 16
COLUMNS = ["no CLAHE", "current 2D (clip 2, 512 px)", "masked 2D (clip 2, 512 px)",
           "masked gentle (clip 1.5, 1024 px)", f"masked 3D +-{POOL} slices (clip 2, 512 px)"]


def background(img):
    """Large uniform areas: no data, or very bright / very dark with little texture (8x coarser)."""
    small = cv2.resize(img, (img.shape[1] // 8, img.shape[0] // 8), interpolation=cv2.INTER_AREA).astype(float)
    m = ndimage.gaussian_filter(small, 2)
    sd = np.sqrt(np.maximum(ndimage.gaussian_filter(small ** 2, 3) - ndimage.gaussian_filter(small, 3) ** 2, 0))
    bg = ((m > 232) | (m < 25)) & (sd < 6)
    bg = ndimage.binary_opening(bg, iterations=2)
    bg = cv2.resize(bg.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return bg | (img == 0)


def hists(img, use, tile):
    gy, gx = img.shape[0] // tile, img.shape[1] // tile
    cy = np.minimum(np.arange(img.shape[0]) // tile, gy - 1)
    cx = np.minimum(np.arange(img.shape[1]) // tile, gx - 1)
    cell = cy[:, None] * gx + cx[None, :]
    key = (cell * 256 + img)[use]
    return np.bincount(key, minlength=gy * gx * 256).reshape(gy, gx, 256).astype(float)


def luts(h, clip, min_frac=0.05, tile_area=None):
    n = h.sum(-1, keepdims=True)
    few = n[..., 0] < min_frac * tile_area
    if few.any():   # borrow from the 3x3 neighbourhood of cells
        nb = ndimage.uniform_filter(h, size=(3, 3, 1), mode="nearest") * 9
        h = np.where(few[..., None], nb, h)
        n = h.sum(-1, keepdims=True)
    limit = np.maximum(clip * n / 256, 1)
    c = np.minimum(h, limit)
    c += (h - c).sum(-1, keepdims=True) / 256
    lut = np.clip(np.round(np.cumsum(c, -1) * 255 / np.maximum(n, 1)), 0, 255)
    empty = n[..., 0] < 1
    lut[empty] = np.arange(256)
    return lut.astype(np.float32)


def apply(img, lut, tile):
    gy, gx = lut.shape[:2]

    def coords(n, g):
        p = (np.arange(n) + 0.5) / tile - 0.5
        a = np.clip(np.floor(p).astype(int), 0, g - 1)
        b = np.clip(a + 1, 0, g - 1)
        w = np.clip(p - a, 0, 1).astype(np.float32)
        w[p < 0] = 0
        return a, b, w
    y0, y1, wy = coords(img.shape[0], gy)
    x0, x1, wx = coords(img.shape[1], gx)
    out = np.empty(img.shape, np.float32)
    for r0 in range(0, img.shape[0], 512):
        r = slice(r0, r0 + 512)
        I = img[r]
        a = lut[y0[r][:, None], x0[None, :], I] * (1 - wx) + lut[y0[r][:, None], x1[None, :], I] * wx
        b = lut[y1[r][:, None], x0[None, :], I] * (1 - wx) + lut[y1[r][:, None], x1[None, :], I] * wx
        out[r] = a * (1 - wy[r][:, None]) + b * wy[r][:, None]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


arr = omezarr.open_scale(VOL, 0)
H, W = arr.shape[1:]
check_done = False
for name, (cx, cy) in REGIONS.items():
    x0 = int(np.clip(cx - CONTEXT // 2, 0, W - CONTEXT))
    y0 = int(np.clip(cy - CONTEXT // 2, 0, H - CONTEXT))
    zs = np.arange(Z0 - POOL, Z0 + NZ + POOL)
    stack = arr[int(zs[0]):int(zs[-1]) + 1, y0:y0 + CONTEXT, x0:x0 + CONTEXT].read().result()
    bgs = [background(s) for s in stack]
    h512 = np.array([hists(s, ~b, 512) for s, b in zip(stack, bgs)])
    pooled = np.cumsum(np.concatenate([np.zeros_like(h512[:1]), h512]), axis=0)
    c0 = (CONTEXT - CROP) // 2
    crop = (slice(c0, c0 + CROP), slice(c0, c0 + CROP))
    out = np.full((NZ, CROP, len(COLUMNS) * CROP + (len(COLUMNS) - 1) * BAR), 128, np.uint8)
    labels = []
    for i in range(NZ):
        k = POOL + i
        img, bg = stack[k], bgs[k]
        valid = img > 0
        cols = [img,
                render._clahe(img.copy(), valid, 2.0, 512),
                apply(img, luts(h512[k], 2.0, tile_area=512 ** 2), 512),
                apply(img, luts(hists(img, ~bg, 1024), 1.5, tile_area=1024 ** 2), 1024),
                apply(img, luts(pooled[k + POOL + 1] - pooled[k - POOL], 2.0, tile_area=512 ** 2 * (2 * POOL + 1)), 512)]
        for j in range(2, 5):
            cols[j][~valid] = 0
        if not check_done:   # our CLAHE, unmasked, against OpenCV's on the same plane
            ref = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(CONTEXT // 512, CONTEXT // 512)).apply(img)
            ours = apply(img, luts(hists(img, np.ones_like(valid), 512), 2.0, tile_area=512 ** 2), 512)
            d = np.abs(ref.astype(int) - ours.astype(int))
            print(f"check vs OpenCV (unmasked): mean |diff| {d.mean():.2f}, p99 {np.percentile(d, 99):.0f} grey levels", flush=True)
            check_done = True
        for j, c in enumerate(cols):
            out[i, :, j * (CROP + BAR):j * (CROP + BAR) + CROP] = c[crop]
        labels.append(f"z {Z0 + i} | {name} | columns: " + " | ".join(COLUMNS))
    tifffile.imwrite(OUT / f"region_{name}.tif", out, imagej=True, resolution=(125.0, 125.0),
                     metadata={"spacing": 0.008, "unit": "micron", "axes": "ZYX", "Labels": labels})
    print(f"{name}: context x {x0} y {y0}, background {np.mean(bgs[POOL]):.1%} of the context", flush=True)
print("->", OUT)
