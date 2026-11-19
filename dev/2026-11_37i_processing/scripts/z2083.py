import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import io
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.config import load_config, step_dir
from pipeline.slices import load_slices, StackCache
from pipeline.preview import mosaic

cfg = load_config(os.path.expandvars("configs/lab/${DEV_PREFIX}37i.yaml"))
ZS = [2081, 2082, 2083, 2084, 2085]
REF = 2084
CROP_TILE, CROP = "0-0", 1000
f = (cfg.get("preview") or {}).get("factor", 16)

sl = load_slices(cfg)
idx = pd.read_csv(step_dir(cfg, "preview", "thumbs", "index.csv"), dtype={"tile": str})
thumbs = step_dir(cfg, "preview", "thumbs")
cache = StackCache(cfg["raw_dir"])


def thumb(z, tile):
    r = idx[(idx.z == z) & (idx.tile == tile)].iloc[0]
    return np.load(thumbs / r.npy, mmap_mode="r")[r.i].astype(float)


def shift(a, b):
    """(dy, dx) such that a[y, x] ~ b[y + dy, x + dx]; phase correlation on the common shape, plus peak corr."""
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a, b = a[:h, :w] - a[:h, :w].mean(), b[:h, :w] - b[:h, :w].mean()
    win = np.outer(np.hanning(h), np.hanning(w))
    F = np.fft.fft2(a * win).conj() * np.fft.fft2(b * win)
    c = np.fft.ifft2(F / (np.abs(F) + 1e-9)).real
    dy, dx = np.unravel_index(np.argmax(c), c.shape)
    dy, dx = (dy - h if dy > h // 2 else dy), (dx - w if dx > w // 2 else dx)
    # correlation of the overlapping parts after the shift
    ya, yb = max(0, -dy), max(0, dy)
    xa, xb = max(0, -dx), max(0, dx)
    hh, ww = h - abs(dy), w - abs(dx)
    pa, pb = a[ya:ya + hh, xa:xa + ww], b[yb:yb + hh, xb:xb + ww]
    return dy, dx, float(np.corrcoef(pa.ravel(), pb.ravel())[0, 1])


# Crop centre: the most textured window of CROP_TILE in the REF thumbnail.
ref = thumb(REF, CROP_TILE)
w = CROP // f
best, cy, cx = -1, 0, 0
for y in range(w, ref.shape[0] - 2 * w, w // 2):
    for x in range(w, ref.shape[1] - 2 * w, w // 2):
        s = ref[y:y + w, x:x + w].std()
        if s > best:
            best, cy, cx = s, y, x

for z in ZS:
    for t in ("0-0", "0-1", "1-0", "1-1"):
        dy, dx, c = shift(thumb(REF, t), thumb(z, t))
        print(f"z {z} tile {t}: shift vs z {REF} ({dy * f}, {dx * f}) px, thumbnail corr {c:.3f}", file=sys.stderr)
    for a, b in ((z - 1, z),):
        if a in ZS:
            for t in ("0-0", "1-1"):
                dy, dx, c = shift(thumb(a, t), thumb(b, t))
                print(f"   z {a} -> {b} tile {t}: shift ({dy * f}, {dx * f}) px, corr {c:.3f}", file=sys.stderr)

fig, axes = plt.subplots(2, len(ZS), figsize=(4 * len(ZS), 8.6), layout="constrained")
for j, z in enumerate(ZS):
    rows = idx[idx.z == z].merge(sl[sl.z == z][["z", "tile", "tile_row", "tile_col", "timestamp", "file", "index"]],
                                 on=["z", "tile"])
    img, labels = mosaic(rows, thumbs)
    ax = axes[0, j]
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    for (x, y), t in labels:
        ax.text(x, y, t, color="white", fontsize=7, va="top", bbox={"facecolor": "black", "alpha": 0.5, "pad": 1, "lw": 0})
    ts = rows["timestamp"].iloc[0]
    ax.set_title(f"z {z}   {pd.Timestamp(ts):%d %b %H:%M:%S} UTC", fontsize=10, color="red" if z == 2083 else "black")
    ax.axis("off")
    dy, dx, _ = shift(ref, thumb(z, CROP_TILE))
    y0, x0 = (cy + dy) * f, (cx + dx) * f
    ax.add_patch(plt.Rectangle((cx + dx, cy + dy), w, w, fill=False, ec="red", lw=1))
    rec = rows[rows.tile == CROP_TILE].iloc[0]
    crop = cache.read(rec, slice(y0, y0 + CROP), slice(x0, x0 + CROP)).astype(float)
    lo, hi = np.percentile(crop, [1, 99])
    ax = axes[1, j]
    ax.imshow(np.clip((crop - lo) / (hi - lo), 0, 1), cmap="gray")
    ax.set_title(f"tile {CROP_TILE}, full res (red box above), x {x0}, y {y0}", fontsize=8)
    ax.axis("off")
fig.suptitle("37i around z 2083 (first slice after the 18-19 Oct pause). Top: each tile as acquired (not stitched). "
             "Bottom: the same tissue at full resolution", fontsize=11)
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110)
sys.stdout.buffer.write(buf.getvalue())
