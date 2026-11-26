"""Check the reference tissue mask (pipeline/refmask.py) of a rendered dataset, by eye and by numbers.

    python tools/refmask_check.py CONFIG VOLUME OUT_DIR [--name NAME] [--overlay Z ...]
        [--stats Z0 ...] [--windows N] [--boxes BOXES.json] [--seed S]

The mask is made from the preview thumbnails (as it would be before the intensity step) and placed in
the rendered VOLUME (a folder in the config's output_dir) with render's frozen plan
(work/render/<stem>/tiles.csv, render.json).

--overlay Z: OUT_DIR/<NAME>_z<Z>.png, the volume at 128 nm (--scale 4, the plane holding z) with the mask
  tinted and outlined in yellow, the background (no score) tinted blue, and --boxes within 64 planes
  of z as magenta squares.
--stats Z0: windows of 1024 px at 8 nm, wholly inside the mask of planes Z0+16 and Z0+48 (Z0 a multiple
  of 64: one chunk read gives both), --windows positions per Z0 (random, not overlapping), and as
  many half again wholly in tissue (background left out) for comparison; per window the neuropil
  metrics used to match neuropil contrast between volumes: lc8 (mean SD in 32 px windows at 8 nm), lc32 (mean SD in
  8 px windows at 32 nm), noise8 (MAD of horizontal neighbour differences / sqrt 2), std, p5, p50, p95.
--boxes BOXES.json: [{"z", "x0", "y0", "x1", "y1", ...}] in the volume's voxels (as stored): the
  fraction of each box inside the mask (e.g. hand-picked neuropil patches).
Results: OUT_DIR/<NAME>_check.json (per window and box, and medians).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import omezarr, preview, refmask  # noqa: E402
from pipeline.config import load_config, step_path  # noqa: E402
from pipeline.stats import robust_sd  # noqa: E402

METRICS = ["lc8", "lc32", "noise8", "std", "p5", "p50", "p95"]
WIN = 1024      # stats window, 8 nm px
PAIR = (16, 48)  # planes of a 64-plane chunk measured


def metrics(x):
    """Neuropil metrics as used to match contrast between volumes (lc8, lc32, noise8, std, p5, p50, p95)."""
    x = x.astype(np.float32)
    m = ndimage.uniform_filter(x, 32)
    lc8 = np.sqrt(np.maximum(ndimage.uniform_filter(x * x, 32) - m * m, 0)).mean()
    h, w = (s // 4 * 4 for s in x.shape)
    y = x[:h, :w].reshape(h // 4, 4, w // 4, 4).mean(axis=(1, 3))
    m = ndimage.uniform_filter(y, 8)
    lc32 = np.sqrt(np.maximum(ndimage.uniform_filter(y * y, 8) - m * m, 0)).mean()
    d = np.diff(x, axis=1).ravel()
    noise = robust_sd(d) / math.sqrt(2)
    p5, p50, p95 = np.percentile(x, [5, 50, 95])
    return {"lc8": float(lc8), "lc32": float(lc32), "noise8": float(noise), "std": float(x.std()),
            "p5": float(p5), "p50": float(p50), "p95": float(p95)}


class Dataset:
    def __init__(self, cfg, volume):
        self.cfg, self.volume = cfg, volume
        stem = volume.removesuffix(".zarr").removesuffix(".ome")
        rdir = step_path(cfg, "render", stem)
        self.meta = json.loads((rdir / "render.json").read_text())
        self.plan = pd.read_csv(rdir / "tiles.csv", dtype={"tile": str}).set_index(["z", "tile"])
        self.thumbs_dir = step_path(cfg, "preview", "thumbs")
        idx = pd.read_csv(self.thumbs_dir / "index.csv", dtype={"tile": str})
        st = pd.read_csv(step_path(cfg, "preview", "stats.csv"), dtype={"tile": str},
                         usecols=["z", "tile", "p0_5", "p99_5"])
        self.index = idx.merge(st, on=["z", "tile"])
        self.root = Path(cfg["output_dir"]) / volume
        self.rotated = self.meta["settings"].get("rotate_deg", 0) == 180
        self.planes = {int(p[0][0]): i for i, p in enumerate(self.meta["planes"])}
        self.settings = {**refmask.DEFAULTS["refmask"], **cfg.get("refmask", {})}

    def tile_masks(self, z):
        rows = self.index[self.index.z == z]
        tiles = {r.tile: (np.asarray(np.load(self.thumbs_dir / r.npy, mmap_mode="r")[int(r.i)]), r.p0_5, r.p99_5)
                 for r in rows.itertuples()}
        masks, scores, thr = refmask.slice_masks(tiles, self.settings)
        c = self.settings["cell_px"]
        out = {t: (refmask.to_pixels(masks[t], tiles[t][0].shape, c),
                   refmask.to_pixels(np.isnan(scores[t]), tiles[t][0].shape, c)) for t in tiles}
        return out, thr

    def to_volume(self, z, tile, thumb_shape, f):
        """2x3: thumbnail px -> stored volume px at downsample f."""
        p = self.plan.loc[(z, tile)]
        T = preview.thumb_to_tile(thumb_shape, (p["height"], p["width"]))
        A = np.array([[p.a, p.b, p.tx], [p.c, p.d, p.ty]])
        M = np.c_[A[:, :2] @ T[:, :2], A[:, :2] @ T[:, 2] + A[:, 2]]
        M[:, 2] -= self.meta["origin_xy"]
        if self.rotated:
            W, H = self.meta["canvas_size_xy"]
            M = np.array([[-1.0, 0, W - 1], [0, -1.0, H - 1]]) @ np.r_[M, [[0, 0, 1]]]
        M = M / f
        M[:, 2] -= (f - 1) / (2 * f)   # full-res px centre c -> px (c - (f - 1) / 2) / f at downsample f
        return M

    def volume_mask(self, z, f):
        """(mask, background, covered) of slice z in the stored volume's plane grid at downsample f."""
        W, H = (math.ceil(v / f) for v in self.meta["canvas_size_xy"])
        mask, bg, cov = (np.zeros((H, W), bool) for _ in range(3))
        tiles, thr = self.tile_masks(z)
        for tile, (m, b) in tiles.items():
            M = self.to_volume(z, tile, m.shape, f)
            warp = [cv2.warpAffine(a.astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST, borderValue=v)
                    for a, v in ((m, 0), (b, 0), (np.ones_like(m), 0))]
            mask |= warp[0] > 0
            bg |= warp[1] > 0
            cov |= warp[2] > 0
        return mask, bg & ~mask, cov, thr


def overlay(ds, z, out_png, name, boxes=(), scale=4):
    f = 2 ** scale
    mask, bg, cov, thr = ds.volume_mask(z, f)
    arr = omezarr.open_scale(ds.root, scale)
    img = np.asarray(arr[ds.planes[z] // f].read().result())
    h, w = min(img.shape[0], mask.shape[0]), min(img.shape[1], mask.shape[1])
    img, mask, bg, cov = img[:h, :w], mask[:h, :w], bg[:h, :w], cov[:h, :w]
    rgb = np.repeat(img[:, :, None], 3, axis=2).astype(np.float32)
    rgb[mask] = 0.82 * rgb[mask] + 0.18 * np.array([0, 230, 255])        # BGR yellow tint
    rgb[bg & cov] = 0.8 * rgb[bg & cov] + 0.2 * np.array([255, 120, 0])  # BGR blue tint
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgb, cnt, -1, (0, 230, 255), 2 if f <= 8 else 1)
    near = {(b["y0"], b["y1"], b["x0"], b["x1"]) for b in boxes if abs(int(b["z"]) - z) <= 64}
    for y0, y1, x0, x1 in near:   # boxes of nearby planes (e.g. hand-picked patches), magenta
        cv2.rectangle(rgb, (round(x0 / f), round(y0 / f)), (round(x1 / f), round(y1 / f)), (255, 0, 255), 2)
    tissue = cov & ~bg
    frac = mask.sum() / max(tissue.sum(), 1)
    label = f"{name} z {z}  {8 * f} nm  {ds.settings['method']}: neuropil mask (yellow) {100 * frac:.0f}% of tissue, threshold {thr:.2f}; blue: background"
    fs = 1.4 * 8 / f
    cv2.putText(rgb, label, (20, int(50 * 8 / f)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(rgb, label, (20, int(50 * 8 / f)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_png), rgb, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return {"z": z, "threshold": thr, "mask_fraction_of_tissue": float(frac), "png": out_png.name}


def place(ok, n, size, rng):
    """Up to n top-left corners (y, x) of size x size squares wholly inside ``ok``, not overlapping."""
    S = np.pad(np.cumsum(np.cumsum(ok.astype(np.int64), 0), 1), ((1, 0), (1, 0)))
    count = S[size:, size:] - S[:-size, size:] - S[size:, :-size] + S[:-size, :-size]
    ys, xs = np.nonzero(count == size * size)
    picked = []
    for i in rng.permutation(len(ys)):
        y, x = ys[i], xs[i]
        if all(abs(y - a) >= size or abs(x - b) >= size for a, b in picked):
            picked.append((int(y), int(x)))
            if len(picked) == n:
                break
    return picked


def stats(ds, z0s, n, rng):
    f = 16
    s = WIN // f
    arr = omezarr.open_scale(ds.root, 0)
    rows = []
    for z0 in z0s:
        zs = [z0 + d for d in PAIR]
        ms = [ds.volume_mask(z, f) for z in zs]
        neuropil = ms[0][0] & ms[1][0]
        tissue = ms[0][2] & ms[1][2] & ~ms[0][1] & ~ms[1][1]
        for kind, ok, k in (("mask", neuropil, n), ("tissue", tissue, (n + 1) // 2)):
            for y, x in place(ok, k, s, rng):
                Y, X = y * f, x * f
                block = np.asarray(arr[ds.planes[zs[0]]:ds.planes[zs[1]] + 1:zs[1] - zs[0], Y:Y + WIN, X:X + WIN]
                                   .read().result())
                for z, win in zip(zs, block):
                    rows.append({"kind": kind, "z": z, "y": Y, "x": X, "zero_frac": float((win == 0).mean()),
                                 **metrics(win)})
        print(f"z {z0}: {sum(r['z'] == zs[0] and r['kind'] == 'mask' for r in rows)} mask windows", flush=True)
    return rows


def recall(ds, boxes):
    f = 16
    out, cache = [], {}
    for b in boxes:
        z = int(b["z"])
        if z not in cache:
            cache[z] = ds.volume_mask(z, f)
        mask, bg, cov, _ = cache[z]
        y0, y1, x0, x1 = (int(round(b[k] / f)) for k in ("y0", "y1", "x0", "x1"))
        sub = mask[y0:y1, x0:x1]
        out.append({**b, "in_mask": float(sub.mean()), "background": float(bg[y0:y1, x0:x1].mean())})
    return out


def summary(rows):
    d = pd.DataFrame(rows)
    return {kind: {"n": int(len(g)), **{m: float(g[m].median()) for m in METRICS}} for kind, g in d.groupby("kind")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("volume")
    ap.add_argument("out_dir")
    ap.add_argument("--name")
    ap.add_argument("--overlay", type=int, nargs="*", default=[])
    ap.add_argument("--stats", type=int, nargs="*", default=[])
    ap.add_argument("--windows", type=int, default=25)
    ap.add_argument("--boxes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", help="the dataset's output_dir, if not the config's (e.g. mounted elsewhere)")
    ap.add_argument("--method", choices=["classifier", "score"], help="refmask.method (default: the config's, else classifier)")
    ap.add_argument("--scale", type=int, default=4, help="overlay pyramid scale (4: 128 nm, 3: 64 nm)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, output_dir=a.output_dir)
    if a.method:
        cfg.setdefault("refmask", {})["method"] = a.method
    ds = Dataset(cfg, a.volume)
    name = a.name or cfg.get("name", Path(a.config).stem)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = {"name": name, "volume": a.volume, "settings": ds.settings}
    given = json.loads(Path(a.boxes).read_text()) if a.boxes else []
    res["overlays"] = [overlay(ds, z, out / f"{name}_z{z}.png", name, given, a.scale) for z in a.overlay]
    for o in res["overlays"]:
        print(f"overlay z {o['z']}: mask {100 * o['mask_fraction_of_tissue']:.0f}% of tissue, threshold "
              f"{o['threshold']:.2f}", flush=True)
    if a.boxes:
        boxes = recall(ds, given)
        cover = np.array([b["in_mask"] for b in boxes])
        res["boxes"] = boxes
        res["recall"] = {"n": len(boxes), "mean_in_mask": float(cover.mean()),
                         "frac_boxes_over_half_in_mask": float((cover > 0.5).mean())}
        print("recall:", res["recall"], flush=True)
    if a.stats:
        rows = stats(ds, a.stats, a.windows, np.random.default_rng(a.seed))
        res["windows"] = rows
        res["stats"] = summary(rows)
        for kind, s in res["stats"].items():
            print(f"{kind:<7} n {s['n']:4d} " + " ".join(f"{m} {s[m]:.2f}" for m in METRICS), flush=True)
    (out / f"{name}_check.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
