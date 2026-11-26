"""Are structures as deep (z) as they are wide (x, y)? Correlation lengths per plane, for slabs rendered with and
without z-correction, and what the correction's interpolation does to in-plane detail.

    python dev/2026-11_zstep_correction/scripts/shape_measure.py OUT.json LABEL=VOLUME.ome.zarr [LABEL=VOLUME ...]
        [--crop 2048] [--sigma 2] [--positions positions.csv --z0 Z]

Each volume (scale 0, central ``crop`` px square, all planes) is smoothed with a 3-D Gaussian (``sigma`` voxels,
the same along every axis, so pixel noise does not set the result). Per plane z: the correlation of the plane
with itself shifted k px along x and along y, and with plane z + k; the correlation length along each axis is
the shift at which that falls to 0.5. anisotropy = L_z / sqrt(L_x L_y): 1 when structures are as deep as wide,
> 1 where slices are thinner than the voxel (structures drawn out along z), < 1 where they are thicker.
Uneven slice spacing shows as anisotropy that changes from plane to plane (its spread along z), not in the mean.

detail: SD of the 1-2 px band-pass of the unsmoothed plane over the plane's SD (lower = blurrier / less noise).
blend: with the volume's render plan (work/render/<volume>/render.json), detail of the planes that are almost one
slice (largest source weight >= 0.9) against those that are an even mix of two (<= 0.6).
With --positions (zcorrect's positions.csv) and --z0 (global z of the volume's first plane, uncorrected volume):
the correlation between the first volume's per-plane anisotropy and the estimated spacing of its planes.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pipeline import omezarr  # noqa: E402

K = 12


def ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else np.nan


def length(curve, level=0.5):
    """Shift at which the correlation (curve[k - 1] at shift k, 1 at shift 0) first falls to ``level``."""
    c = np.concatenate([[1.0], curve])
    below = np.where(c < level)[0]
    if not len(below) or not np.isfinite(c[:below[0] + 1]).all():
        return np.nan
    i = below[0]
    return float(i - 1 + (c[i - 1] - level) / (c[i - 1] - c[i]))


def measure(path, crop, sigma):
    a = omezarr.open_scale(str(path).rstrip("/"), 0)
    nz, ny, nx = (int(v) for v in a.shape)
    c = min(crop, ny, nx)
    y0, x0 = (ny - c) // 2, (nx - c) // 2
    v = np.asarray(a[:, y0:y0 + c, x0:x0 + c].read().result())
    detail = []
    for p in v:
        f = p.astype(np.float32)
        band = cv2.GaussianBlur(f, (0, 0), 1.0) - cv2.GaussianBlur(f, (0, 0), 2.0)
        detail.append(float(band[8:-8, 8:-8].std() / max(f.std(), 1e-6)))
    empty = float((v == 0).mean())
    # How much each plane is a blend: the largest weight among its source slices (1 = one slice, 0.5 = an even
    # mix of two), from the render plan next to the volume (<output_dir>/work/render/<volume>/render.json).
    root = Path(str(path).rstrip("/"))
    plan = root.parent / "work" / "render" / root.name.replace(".ome.zarr", "") / "render.json"
    weight = None
    if plan.exists():
        planes = json.loads(plan.read_text())["planes"]
        if len(planes) == nz:
            weight = np.array([max(w for _, w in q) for q in planes])
    m = int(np.ceil(3 * sigma))
    g = ndimage.gaussian_filter(v.astype(np.float32), sigma)[:, m:-m, m:-m]
    lx, ly, lz = [], [], []
    for z in range(nz):
        p = g[z]
        lx.append(length([ncc(p[:, :-k], p[:, k:]) for k in range(1, K + 1)]))
        ly.append(length([ncc(p[:-k], p[k:]) for k in range(1, K + 1)]))
        lz.append(length([ncc(p, g[z + k]) for k in range(1, K + 1)]) if z + K < nz else np.nan)
    lx, ly, lz = (np.array(x) for x in (lx, ly, lz))
    aniso = lz / np.sqrt(lx * ly)
    ok = np.isfinite(aniso)
    ok[:m] = False                                    # planes the z-smoothing could not fill evenly
    ok[-m:] = False
    q = np.percentile(aniso[ok], [5, 25, 50, 75, 95])
    by_weight = None
    if weight is not None:
        d = np.array(detail)
        one, mix = weight >= 0.9, weight <= 0.6
        by_weight = {"mean_largest_weight": round(float(weight.mean()), 3),
                     "planes_from_one_slice": int((weight > 0.999).sum()),
                     "detail_weight_above_0.9": [int(one.sum()), round(float(np.median(d[one])), 4) if one.any() else None],
                     "detail_weight_below_0.6": [int(mix.sum()), round(float(np.median(d[mix])), 4) if mix.any() else None],
                     "r_detail_weight": round(float(np.corrcoef(d, weight)[0, 1]), 3) if weight.std() > 0 else None}
    return {"volume": Path(str(path).rstrip("/")).name, "blend": by_weight, "shape": [nz, ny, nx], "crop_px": c, "empty_fraction": round(empty, 4),
            "planes_measured": int(ok.sum()),
            "L_x_px": round(float(np.nanmedian(lx[ok])), 3), "L_y_px": round(float(np.nanmedian(ly[ok])), 3),
            "L_z_planes": round(float(np.nanmedian(lz[ok])), 3),
            "anisotropy": {"median": round(float(q[2]), 4), "p5": round(float(q[0]), 4), "p95": round(float(q[4]), 4),
                           "robust_sd": round(float((q[3] - q[1]) / 1.349), 4), "sd": round(float(aniso[ok].std()), 4),
                           "min": round(float(aniso[ok].min()), 4), "max": round(float(aniso[ok].max()), 4)},
            "detail": {"median": round(float(np.median(detail)), 4), "p5": round(float(np.percentile(detail, 5)), 4),
                       "p95": round(float(np.percentile(detail, 95)), 4)},
            "per_plane": {"anisotropy": [None if not np.isfinite(x) else round(float(x), 4) for x in aniso],
                          "detail": [round(x, 4) for x in detail]}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path)
    p.add_argument("volumes", nargs="+", metavar="LABEL=VOLUME")
    p.add_argument("--crop", type=int, default=2048)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--positions", type=Path)
    p.add_argument("--z0", type=int)
    args = p.parse_args(argv)
    if args.out.exists():
        sys.exit(f"{args.out} exists: choose a new name")
    res = {}
    for item in args.volumes:
        label, path = item.split("=", 1)
        res[label] = measure(path, args.crop, args.sigma)
        print(label, json.dumps({k: v for k, v in res[label].items() if k != "per_plane"}), flush=True)
    if args.positions and args.z0 is not None:
        import pandas as pd
        pos = pd.read_csv(args.positions).set_index("z")["position_nm"]
        first = next(iter(res.values()))
        an = np.array([np.nan if x is None else x for x in first["per_plane"]["anisotropy"]])
        zs = args.z0 + np.arange(len(an))
        # Depth of the next 6 slices from plane z, relative to nominal: the spacing the z-length is read over.
        span = np.array([(pos.get(z + 6, np.nan) - pos.get(z, np.nan)) for z in zs])
        span = span / np.nanmedian(np.diff(pos.to_numpy())) / 6
        m = np.isfinite(an) & np.isfinite(span)
        res["anisotropy_vs_spacing"] = {"r": round(float(np.corrcoef(an[m], 1 / span[m])[0, 1]), 3), "planes": int(m.sum()),
                                        "spacing_ratio_6_slices": {"min": round(float(span[m].min()), 3),
                                                                   "max": round(float(span[m].max()), 3)}}
        print("anisotropy_vs_spacing", json.dumps(res["anisotropy_vs_spacing"]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
