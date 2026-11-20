"""Real-size synthetic dataset: 2x2 tiles of 12751 x 13875 uint16, a few slices, known overlap/drift.

Texture: multi-octave noise (EM-like blobs at several scales) that changes slowly with z.
Usage: python make_realsize.py OUT_DIR [n_slices]
"""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import tifffile

TH, TW = 12751, 13875
OY, OX = 640, 520                     # true overlaps (px)
H, W = 2 * TH - OY, 2 * TW - OX


def octaves(rng, shape, scales=(64, 16, 4, 1), weights=(1.0, 0.7, 0.5, 0.25)):
    out = np.zeros(shape, np.float32)
    for s, w in zip(scales, weights):
        small = rng.standard_normal((shape[0] // s + 2, shape[1] // s + 2)).astype(np.float32)
        if s > 1:
            small = cv2.GaussianBlur(small, (0, 0), 0.8)
            big = cv2.resize(small, (shape[1] // s * s + 2 * s, shape[0] // s * s + 2 * s),
                             interpolation=cv2.INTER_CUBIC)[:shape[0], :shape[1]]
        else:
            big = cv2.GaussianBlur(small[:shape[0], :shape[1]], (0, 0), 1.2)
        out += w * big
    return out


def main(out, n):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    margin = 64
    base = octaves(rng, (H + 2 * margin, W + 2 * margin))
    steps = np.rint(rng.normal(0, 3, (n, 2))).astype(int)
    steps[0] = 0
    drift = np.cumsum(steps, axis=0)
    drift -= drift.min(axis=0)
    assert drift.max() < margin
    start = datetime(2026, 9, 24, 23, 50, 0)
    ts = [start + timedelta(seconds=142 * z) for z in range(n)]
    origin = {f"{r}-{c}": (c * (TW - OX), r * (TH - OY)) for r in (0, 1) for c in (0, 1)}
    files = {}
    for day in sorted({t.date() for t in ts}):
        zs = [z for z, t in enumerate(ts) if t.date() == day]
        stacks = {tile: np.empty((len(zs), TH, TW), ">u2") for tile in origin}
        for i, z in enumerate(zs):
            evolve = octaves(np.random.default_rng(100 + z), base.shape, scales=(16, 4), weights=(0.15, 0.1))
            dx, dy = drift[z]
            mont = (base + evolve)[margin + dy:margin + dy + H, margin + dx:margin + dx + W]
            mont = 33000 + 2500 * mont + np.random.default_rng(200 + z).normal(0, 300, mont.shape).astype(np.float32)
            for tile, (x0, y0) in origin.items():
                stacks[tile][i] = np.clip(mont[y0:y0 + TH, x0:x0 + TW], 0, 65535).astype(np.uint16)
            print(f"slice {z} done", flush=True)
        prefix = f"M{day.month:02d}_D{day.day:02d}"
        for tile, data in stacks.items():
            r, c = tile.split("-")
            labels = [os.path.expandvars(f"INSTRUMENT_{ts[z]:%y-%m-%d_%H%M%S}_0-{r}-{c}_InLens_raw.tif") for z in zs]
            name = f"{prefix}_tile{tile}.tif"
            tifffile.imwrite(out / name, data, imagej=True, byteorder=">", resolution=(125.0, 125.0),
                             metadata={"spacing": 0.008, "unit": "micron", "Labels": labels})
            files[name] = zs
            print("wrote", name, flush=True)
    truth = {"tile_origin": origin, "drift": drift.tolist(), "files": files, "overlap": [OY, OX]}
    (out.parent / "truth.json").write_text(json.dumps(truth))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8)
