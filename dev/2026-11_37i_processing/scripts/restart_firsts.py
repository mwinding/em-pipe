import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import sys

import numpy as np
import pandas as pd

from pipeline.config import load_config, step_dir
from pipeline.slices import load_slices

cfg = load_config(os.path.expandvars("configs/lab/${DEV_PREFIX}37i.yaml"))
F = 16
sl = load_slices(cfg)
idx = pd.read_csv(step_dir(cfg, "preview", "thumbs", "index.csv"), dtype={"tile": str})
thumbs = step_dir(cfg, "preview", "thumbs")
cache = {}


def thumb(z, tile):
    r = idx[(idx.z == z) & (idx.tile == tile)]
    if r.empty:
        return None
    r = r.iloc[0]
    if r.npy not in cache:
        cache[r.npy] = np.load(thumbs / r.npy, mmap_mode="r")
    return cache[r.npy][r.i].astype(float)


def shift(a, b):
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a, b = a[:h, :w] - a[:h, :w].mean(), b[:h, :w] - b[:h, :w].mean()
    win = np.outer(np.hanning(h), np.hanning(w))
    X = np.fft.fft2(a * win).conj() * np.fft.fft2(b * win)
    c = np.fft.ifft2(X / (np.abs(X) + 1e-9)).real
    dy, dx = np.unravel_index(np.argmax(c), c.shape)
    dy, dx = (dy - h if dy > h // 2 else dy), (dx - w if dx > w // 2 else dx)
    ya, yb, xa, xb = max(0, -dy), max(0, dy), max(0, -dx), max(0, dx)
    hh, ww = h - abs(dy), w - abs(dx)
    pa, pb = a[ya:ya + hh, xa:xa + ww], b[yb:yb + hh, xb:xb + ww]
    return dy * F, dx * F, float(np.corrcoef(pa.ravel(), pb.ravel())[0, 1])


def halves(a, b):
    """dx of the left and right halves (a stretch shows as a difference)."""
    w = min(a.shape[1], b.shape[1]) // 2
    return shift(a[:, :w], b[:, :w])[1], shift(a[:, w:2 * w], b[:, w:2 * w])[1]


per = sl.groupby("z").agg(seam=("seam", "any"), segment=("segment", "first"), ts=("timestamp", "first"))
firsts = sorted(set(per.index[per["seam"]]) | set(per.index[per["segment"].diff() > 0]))
print("first slices after restarts / gaps / segment changes:", firsts, file=sys.stderr)
for z0 in firsts:
    tiles = sorted(sl.loc[sl.z == z0, "tile"])
    lines = []
    for t in tiles:
        cells = []
        for a, b in ((z0, z0 + 1), (z0 + 1, z0 + 2)):
            A, B = thumb(a, t), thumb(b, t)
            if A is None or B is None:
                cells.append("-")
                continue
            dy, dx, c = shift(B, A)
            l, r = halves(B, A)
            cells.append(f"({dy:+d},{dx:+d}) c{c:.3f} L/R dx {l:+d}/{r:+d}")
        lines.append(f"    {t}: first vs next {cells[0]}   | next vs next+1 {cells[1]}")
    print(f"z {z0} ({per.at[z0, 'ts']:%d %b %H:%M}, seg {per.at[z0, 'segment']}):", file=sys.stderr)
    print("\n".join(lines), file=sys.stderr)
