import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import glob
import json
import sys

import numpy as np

from pipeline.config import load_config
from pipeline.slices import load_slices, StackCache

cfg = load_config(os.path.expandvars("configs/lab/${DEV_PREFIX}37i.yaml"))
sl = load_slices(cfg)
cache = StackCache(cfg["raw_dir"])
F = 4
NB = 8


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


def band(z, tile, i):
    rec = sl[(sl.z == z) & (sl.tile == tile)].iloc[0]
    H = int(rec["height"])
    y0, y1 = H * i // NB, H * (i + 1) // NB
    img = cache.read(rec, slice(y0, y1)).astype(np.float32)
    h, w = (img.shape[0] // F) * F, (img.shape[1] // F) * F
    return img[:h, :w].reshape(h // F, F, w // F, F).mean(axis=(1, 3)), y0


for tile in ("0-0", "1-1"):
    for a, b in ((2083, 2084), (2084, 2085)):
        out = []
        for i in range(NB):
            A, y0 = band(a, tile, i)
            B, _ = band(b, tile, i)
            dy, dx, c = shift(B, A)
            out.append(f"y{y0}: ({dy:+d},{dx:+d}) {c:.2f}")
        print(f"tile {tile}, z {a} vs {b} (dy, dx) per band of rows:", " | ".join(out), file=sys.stderr)

D = cfg["output_dir"] + "/work/stitch/samples"
for z in (2082, 2083, 2084, 2085, 2086):
    p = glob.glob(f"{D}/z*{z}.json")
    if not p:
        print(f"stitch sample z {z}: none", file=sys.stderr)
        continue
    r = json.load(open(p[0]))
    t = {k: np.ravel(v)[[2, 5]].round(1).tolist() for k, v in r["tiles"].items()}
    pairs = [(q["tile_a"], q["tile_b"], q["n_inliers"], q["used"], round(q.get("residual_px") or -1, 1)) for q in r["pairs"]]
    print(f"stitch sample z {z} (ok {r['ok']}, seg {r['segment']}): tiles {t}\n   pairs {pairs}", file=sys.stderr)

lay = glob.glob(cfg["output_dir"] + "/work/stitch/*.json")
print("stitch files:", [p.rsplit('/', 1)[-1] for p in lay], file=sys.stderr)
