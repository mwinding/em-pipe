"""Render test slices at 32 nm with Janelia-style patch intensity maps (janelia_solve.py), as
tools/shading_before_after.py does: ImageJ stacks [frame 0: OLD_VOLUME's 32 nm plane, frame 1: new].

    python tools/experiments/janelia_render.py CONFIG OLD_VOLUME Z[,Z...]|@ZFILE NAME=PREFIX [NAME=PREFIX ...]

PREFIX_maps.csv (z, tile, p, g, c) and PREFIX_windows.csv (z, lo, hi) from janelia_solve.py. Each
tile slice is read, divided by its flat-field, mapped by g(x, y) (v - R) + R + c(x, y) (bilinear
between patch centres, clamped at the tile border) and normalised with the slice's common window
(lo, hi). Raw tile reads are cached per slice, so all variants come from one read. Writes
<output_dir>/qc/<NAME>/z<z>.tif and z<z>.png. Array tasks take every n-th slice.
"""
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import omezarr, render, shading, transforms  # noqa: E402
from pipeline.config import deep_merge, load_config  # noqa: E402
import janelia_solve as js  # noqa: E402


def apply_map(img, gc, h, w, rows, cols, block=1024):
    """g (v - R) + R + c on tile pixels rows x cols of an h x w tile, in float32, by row blocks."""
    g, c = gc
    img = img.astype(np.float32, copy=False) if img.dtype == np.float32 else img.astype(np.float32)
    cidx = np.arange(cols.start, cols.stop)
    for b0 in range(0, img.shape[0], block):
        b1 = min(img.shape[0], b0 + block)
        ridx = np.arange(rows.start + b0, rows.start + b1)
        blk = img[b0:b1]
        blk -= np.float32(js.R)
        blk *= js.map_rows(g, ridx, cidx, h, w)
        blk += js.map_rows(c, ridx, cidx, h, w) + np.float32(js.R)
    return img


class CachedReads:
    """StackCache whose reads are kept until clear(): every variant of a slice reads the tiles once."""

    def __init__(self, cache):
        self.cache, self.kept = cache, {}

    def read(self, row, rows=slice(None), cols=slice(None)):
        key = (row["file"], int(row["index"]), rows.start, rows.stop, cols.start, cols.stop)
        if key not in self.kept:
            self.kept[key] = self.cache.read(row, rows, cols)
        return self.kept[key]

    def clear(self):
        self.kept = {}


class PatchRenderer(render._Renderer):
    """render._Renderer with the patch map applied after the flat-field (``maps``: {(z, tile): (g, c)})."""

    maps = {}

    def _tile(self, t):
        # Copy of render._Renderer._tile; the only change is the marked patch-map step.
        f, (H, W) = self.f, self.shape
        T = transforms.from_row(t)
        h, w = int(t["height"]), int(t["width"])
        canvas = self.origin + f * np.array([[0, 0], [W, 0], [0, H], [W, H]], float)
        box = transforms.apply(transforms.invert(T), canvas)
        c0, r0 = (int(q) for q in np.maximum(np.floor(box.min(axis=0)) - f - 1, 0))
        c1, r1 = (int(q) for q in np.minimum(np.ceil(box.max(axis=0)) + f + 1, (w, h)))
        is_shift = transforms.is_translation(T)
        if f > 1:
            if is_shift and render._whole(T[:, 2]):
                c0 += int((self.origin[0] - T[0, 2] - c0) % f)
                r0 += int((self.origin[1] - T[1, 2] - r0) % f)
            c1 -= (c1 - c0) % f
            r1 -= (r1 - r0) % f
        if c1 <= c0 or r1 <= r0:
            return None
        img = self.cache.read(t, slice(r0, r1), slice(c0, c1))
        if self.destreak is not None:
            img = self.destreak(img)
        field = self.shading.get((self.segment.get(t["z"]), t["tile"]))
        if field is not None:
            img = np.divide(img, shading.gain_window(field, (h, w), slice(r0, r1), slice(c0, c1)), dtype=np.float32)
        # --- patch map (Janelia-style intensity correction) ---
        gc = self.maps.get((int(t["z"]), t["tile"]))
        if gc is not None:
            img = apply_map(img, gc, h, w, slice(r0, r1), slice(c0, c1))
        # ------------------------------------------------------
        v = render._normalise(img, t["lo"], t["hi"])
        if f > 1:
            v = cv2.resize(v, ((c1 - c0) // f, (r1 - r0) // f), interpolation=cv2.INTER_AREA)
        hv, wv = v.shape
        mode = t.get("blend", self.mode)
        cut = mode == "cut"
        dr, dc = render._edge_distance(r0, hv, h, f), render._edge_distance(c0, wv, w, f)
        if cut:
            wt = np.add.outer(dr, dc).astype(np.float32)
        elif mode == "seam":
            wt = render._seam_weight(dr, dc, self.seam)
        else:
            wt = np.outer(render._ramp(r0, hv, h, f, self.blend), render._ramp(c0, wv, w, f, self.blend))
        if not cut:
            v *= wt
        L, c = T[:, :2], (f - 1) / 2
        M = np.hstack([L, ((L @ (np.array([c0, r0]) + c) + T[:, 2] - self.origin - c) / f)[:, None]])
        if is_shift and render._whole(M[:, 2]):
            x, y = (int(round(m)) for m in M[:, 2])
            xa, xb, ya, yb = max(x, 0), min(x + wv, W), max(y, 0), min(y + hv, H)
            if xa >= xb or ya >= yb:
                return None
            src = (slice(ya - y, yb - y), slice(xa - x, xb - x))
            return (slice(ya, yb), slice(xa, xb)), v[src], wt[src], cut
        p = transforms.apply(M, [[-0.5, -0.5], [wv - 0.5, -0.5], [-0.5, hv - 0.5], [wv - 0.5, hv - 0.5]])
        xa, ya = (int(q) for q in np.maximum(np.floor(p.min(axis=0)), 0))
        xb, yb = (int(q) for q in np.minimum(np.ceil(p.max(axis=0)) + 1, (W, H)))
        if xa >= xb or ya >= yb:
            return None
        Mw = M - [[0, 0, xa], [0, 0, ya]]

        def warp(a):
            return cv2.warpAffine(a, Mw, (xb - xa, yb - ya), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if cut:
            cover = warp(np.ones_like(v))
            vals = np.divide(warp(v), cover, out=np.zeros_like(cover), where=cover > 0)
            return (slice(ya, yb), slice(xa, xb)), vals, np.where(cover > 0.5, warp(wt), 0), cut
        return (slice(ya, yb), slice(xa, xb)), warp(v), warp(wt), cut


def load_variant(prefix, zs):
    m = pd.read_csv(f"{prefix}_maps.csv", dtype={"tile": str})
    m = m[m["z"].isin(zs)]
    maps = {}
    for (z, t), g in m.groupby(["z", "tile"]):
        g = g.sort_values("p")
        G = int(round(math.sqrt(len(g))))
        maps[(int(z), t)] = (g["g"].to_numpy(np.float64).reshape(G, G), g["c"].to_numpy(np.float64).reshape(G, G))
    win = pd.read_csv(f"{prefix}_windows.csv").set_index("z")
    return maps, win


def main():
    CFG, OLD = sys.argv[1], sys.argv[2]
    zs = js.read_zs(sys.argv[3])
    variants = dict(a.split("=", 1) for a in sys.argv[4:])
    tid = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)) - int(os.environ.get("SLURM_ARRAY_TASK_MIN", 0))
    zs = zs[tid::int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))]
    cfg = load_config(CFG, step_defaults=deep_merge(render.DEFAULTS, shading.DEFAULTS))
    out = Path(cfg["output_dir"])
    old_stem = OLD.removesuffix(".zarr").removesuffix(".ome")
    old_meta = json.loads((out / "work" / "render" / old_stem / "render.json").read_text())
    f = 4
    w, h = old_meta["canvas_size_xy"]
    meta = {**old_meta, "downsample": f, "shape": [1, math.ceil(h / f), math.ceil(w / f)],
            "settings": {**old_meta["settings"], "downsample": f}, "shading": shading.digest(cfg)}
    plan = render._tile_plan(cfg)
    plan = plan[plan["z"].isin(zs)]
    old = omezarr.open_scale(out / OLD, 2)
    loaded = {name: load_variant(prefix, zs) for name, prefix in variants.items()}
    for name in variants:
        (out / "qc" / name).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(8) as pool:
        r = PatchRenderer(cfg, meta, plan, pool)
        r.cache = CachedReads(r.cache)
        for z in zs:
            o = old[z // 4].read().result()
            for name, (maps, win) in loaded.items():
                pz = plan[plan["z"] == z].copy()
                missing = [t for t in pz["tile"] if (z, t) not in maps]
                if z not in win.index:
                    print(f"{name} z {z}: no window, skipped", flush=True)
                    continue
                pz["lo"], pz["hi"] = win.loc[z, "lo"], win.loc[z, "hi"]
                r.tiles = {z: pz.to_dict("records")}
                r.maps = maps
                new = r.plane([(z, 1.0)])
                hh, ww = min(o.shape[0], new.shape[0]), min(o.shape[1], new.shape[1])
                a, b = o[:hh, :ww], new[:hh, :ww]
                qc = out / "qc" / name
                tifffile.imwrite(qc / f"z{z}.tif", np.stack([a, b]), imagej=True, resolution=(1 / 0.032, 1 / 0.032),
                                 metadata={"spacing": 0.032, "unit": "micron", "axes": "ZYX",
                                           "Labels": [f"z {z}: old volume (32 nm plane {z // 4})",
                                                      f"z {z}: {name} patch maps + flat-field"]})
                side = np.concatenate([a, np.full((hh, 20), 255, np.uint8), b], axis=1)
                side = cv2.resize(side, (side.shape[1] // 3, side.shape[0] // 3), interpolation=cv2.INTER_AREA)
                cv2.putText(side, f"z {z}   left: before   right: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)
                cv2.imwrite(str(qc / f"z{z}.png"), side)
                print(f"{name} z {z} done" + (f" (no map for {missing})" if missing else ""), flush=True)
            r.cache.clear()


if __name__ == "__main__":
    main()
