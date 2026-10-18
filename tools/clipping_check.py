"""How much tissue the display windows clip to black (0) or white (255) at full resolution, and what
the darkest structures (e.g. dense granules) look like without the clip.

    python tools/clipping_check.py CONFIG Z[,Z...] [--spots 3] [--crop 1024] [--headroom 15,240]

CONFIG names the rendered volume (render.name; its work/render/<stem>/render.json and tiles.csv).
The display windows come from tissue percentiles measured on 16x downsampled previews, where small
dark objects and pixel noise are averaged away, so at full resolution more tissue can fall outside
them than the percentiles suggest.

1. For each Z: the volume's s0 plane; tissue = pipeline.background.mask on a 16x downsampled copy
   (no-data, which is also 0, is background); the fraction of tissue pixels at 0, <= 2, >= 253 and
   255, and the 1024 px blocks with the most tissue at 0.
2. For the --spots blocks with the most black-clipped tissue (from different slices): the block is
   rendered again from the raw tiles with the volume's plan (transforms, flat-field, patch shapes,
   elastic field) and each tile's window widened so that its black and white points land on
   --headroom (default 15 and 240) instead of 0 and 255; and once with the volume's own windows,
   which must reproduce the volume (a check of this renderer). Panels: volume | CLAHE volume (if
   CONFIG's render.name with _clahe exists) | widened | widened + CLAHE (render.clahe settings of
   the CLAHE config, default clip 1.5, 256 px).

Writes <output_dir>/qc/clipping/clipping.csv, spot_<k>_z<z>.png and prints a summary. Runs on NEMO.
"""
import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import background, omezarr, patchgrid, render, shading  # noqa: E402
from pipeline.config import deep_merge, load_config  # noqa: E402

F = 16   # downsampling for the tissue mask, as the preview step


def tissue_mask(plane):
    """Tissue pixels of a uint8 plane: background.mask on a 16x area-downsampled copy, upsampled."""
    h, w = plane.shape
    small = cv2.resize(plane, (w // F, h // F), interpolation=cv2.INTER_AREA)
    valid = small > 0
    lo, hi = np.percentile(small[valid], (0.5, 99.5)) if valid.any() else (0, 255)
    s8 = np.clip((small.astype(float) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    s8[~valid] = 0
    bg = background.mask(s8) | ~valid
    return cv2.resize((~bg).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0


def render_crop(cfg, meta0, tiles, z, rows, cols, widen):
    """The volume's plane z on rotated-volume pixels rows x cols, rendered from the raw tiles; tile
    windows widened so black/white points land on ``widen`` (None: the volume's own windows)."""
    H, W = meta0["shape"][1:]
    rotated = meta0["settings"].get("rotate_deg", 0) == 180
    if rotated:   # rotated pixel (r, c) is unrotated (H - 1 - r, W - 1 - c)
        rows, cols = slice(H - rows.stop, H - rows.start), slice(W - cols.stop, W - cols.start)
    ox, oy = meta0["origin_xy"]
    h, w = rows.stop - rows.start, cols.stop - cols.start
    meta = {**meta0, "origin_xy": [ox + cols.start, oy + rows.start], "canvas_size_xy": [w, h], "shape": [1, h, w],
            "settings": {**meta0["settings"], "clahe": {**meta0["settings"]["clahe"], "enabled": False}}}
    plan = tiles[tiles["z"] == z].copy()
    if widen is not None:
        a, b = widen
        span = plan["hi"] - plan["lo"]
        new_span = span * 255.0 / (b - a)
        plan["lo"] = plan["lo"] - a / 255.0 * new_span
        plan["hi"] = plan["lo"] + new_span
    with ThreadPoolExecutor(8) as pool:
        out = render._Renderer(cfg, meta, plan, pool).plane([(z, 1.0)])
    return out[::-1, ::-1] if rotated else out


def label(img, text):
    top = np.full((40, img.shape[1]), 255, np.uint8)
    cv2.putText(top, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    return np.concatenate([top, img])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("z")
    p.add_argument("--spots", type=int, default=3)
    p.add_argument("--crop", type=int, default=1024)
    p.add_argument("--headroom", default="15,240")
    args = p.parse_args()
    widen = tuple(float(v) for v in args.headroom.split(","))
    cfg = load_config(args.config, step_defaults=deep_merge(deep_merge(render.DEFAULTS, shading.DEFAULTS), patchgrid.DEFAULTS))
    out = Path(cfg["output_dir"])
    root, wdir = render.volume_paths(cfg)
    meta0 = json.loads((wdir / "render.json").read_text())
    tiles = pd.read_csv(wdir / "tiles.csv", dtype={"tile": str, "file": str})
    vol = omezarr.open_scale(root, 0)
    clahe_root = root.parent / root.name.replace(".ome.zarr", "_clahe.ome.zarr")
    cvol = omezarr.open_scale(clahe_root, 0) if (clahe_root / "s0" / "zarr.json").exists() else None
    clahe_cfg = (cfg["render"]["clahe"] if cfg["render"]["clahe"]["enabled"] else {"clip_limit": 1.5, "tile_px": 256})
    qc = out / "qc" / "clipping"
    qc.mkdir(parents=True, exist_ok=True)
    zs = [int(v) for v in args.z.split(",")]
    plane_of = {z: next(i for i, pl in enumerate(meta0["planes"]) if pl[0][0] == z) for z in zs}
    c = args.crop
    rows_out, spots = [], []
    for z in zs:
        plane = vol[plane_of[z]].read().result()
        tissue = tissue_mask(plane)
        t = plane[tissue]
        n = t.size
        row = {"z": z, "tissue_px": n, "at_0_pct": 100 * (t == 0).sum() / n, "le_2_pct": 100 * (t <= 2).sum() / n,
               "ge_253_pct": 100 * (t >= 253).sum() / n, "at_255_pct": 100 * (t == 255).sum() / n}
        rows_out.append(row)
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
        H, W = plane.shape
        for r0 in range(0, H - c + 1, c):
            for c0 in range(0, W - c + 1, c):
                m = tissue[r0:r0 + c, c0:c0 + c]
                if m.mean() < 0.7:
                    continue
                spots.append((float((plane[r0:r0 + c, c0:c0 + c][m] == 0).mean()), z, r0, c0))
        del plane, tissue
    df = pd.DataFrame(rows_out)
    df.to_csv(qc / "clipping.csv", index=False)
    print("\ntissue pixels clipped (median over slices): black %.2f %%, white %.2f %%" % (df.at_0_pct.median(), df.at_255_pct.median()))
    spots.sort(reverse=True)
    chosen, used = [], set()
    for s in spots:
        if s[1] not in used:
            chosen.append(s)
            used.add(s[1])
        if len(chosen) == args.spots:
            break
    for k, (frac, z, r0, c0) in enumerate(chosen):
        rows, cols = slice(r0, r0 + c), slice(c0, c0 + c)
        cur = vol[plane_of[z], r0:r0 + c, c0:c0 + c].read().result()
        same = render_crop(cfg, meta0, tiles, z, rows, cols, None)
        wide = render_crop(cfg, meta0, tiles, z, rows, cols, widen)
        valid = wide > 0
        diff = np.abs(same.astype(int) - cur.astype(int))
        wide_clahe = render._clahe(wide, valid, clahe_cfg["clip_limit"], clahe_cfg["tile_px"])
        panels = [label(cur, f"z {z}: volume (black-clipped tissue {100 * frac:.1f} %)")]
        if cvol is not None:
            panels.append(label(cvol[plane_of[z], r0:r0 + c, c0:c0 + c].read().result(), "CLAHE volume"))
        panels += [label(wide, f"unclipped: black/white points at {widen[0]:g}/{widen[1]:g}"),
                   label(wide_clahe, f"unclipped + CLAHE {clahe_cfg['clip_limit']}/{clahe_cfg['tile_px']}")]
        gap = np.full((panels[0].shape[0], 16), 255, np.uint8)
        img = panels[0]
        for pn in panels[1:]:
            img = np.concatenate([img, gap, pn], axis=1)
        name = f"spot_{k}_z{z}.png"
        cv2.imwrite(str(qc / name), img)
        below = (wide[valid] < widen[0]).mean() * 100
        print(f"{name}: rows {r0}-{r0 + c}, cols {c0}-{c0 + c} (rotated volume px); re-render vs volume: "
              f"max |diff| {diff.max()}, {100 * (diff > 1).mean():.3f} % px > 1; below the old black point: "
              f"{below:.1f} % of px, darkest {int(wide[valid].min())} (old black point = {widen[0]:g})", flush=True)


if __name__ == "__main__":
    main()
