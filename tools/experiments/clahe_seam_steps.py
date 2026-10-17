"""tools/seam_steps.py with its own output path, local contrast and CLAHE artefact measures
(experiment, Oct 2026).

    python tools/experiments/clahe_seam_steps.py CONFIG OLD_VOLUME Z[,Z...] OUT_PREFIX NAME=DIR [NAME=DIR ...]

The seam steps are exactly those of tools/seam_steps.py (same ownership render, masks, bands,
stretches and medians; the loops only work on the bounding box of each seam's bands, which does not
change the pixels used). DIRs hold z<z>.tif: 2-frame stacks (frame 0: before, frame 1: the
variant; "before" comes from the first DIR) or single planes (the variant). Added:
  <name>_iqr  per seam: median over its stretches of the mean of the inter-quartile ranges of the
              class's pixels on the two sides (the local contrast the step is seen against).
Writes OUT_PREFIX_seams.csv, and OUT_PREFIX_artefacts.csv with one row per slice and image:
  tissue_p25/p50/p75/iqr  of all tissue pixels;
  resin_noise   robust std (1.4826 MAD) of pixel minus its 3x3 mean, in resin 3 px from its edge;
  resin_spread  p90 - p10 of the medians of 32 x 32 px blocks (1 um) that are >= 80 % resin;
  halo          median resin 2-10 px from tissue minus median resin 30-80 px from tissue.
In an array job SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT split the z list, and the files are
OUT_PREFIX_seams.<task>.csv etc. (concatenate them).
--clean (anywhere in the arguments): drop stray ownership labels first. Where the render resamples
(non-integer transforms, elastic warps) the constant tile codes are interpolated at the data's edge
and along seams, and the rounded code of such a pixel can be another tile's label: e.g. a 1-px line
of "1-0" along the bottom edge of tile 2-0 at z 1000, which puts 2-0 pixels near the slice's edge
into the 1-0|2-0 band. Pixels that a 5 x 5 opening of their own label removes become no data.
Runs on NEMO (reads the render plan): sbatch --cpus-per-task=8 --mem=64G.
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import render, shading  # noqa: E402
from pipeline.config import deep_merge, load_config  # noqa: E402
from seam_steps import NEAR, FAR, STRETCH, MIN_PX, classes, ownership  # noqa: E402

BLOCK = 32


def _iqr(v):
    q1, q3 = np.percentile(v, (25, 75))
    return float(q3 - q1)


def steps(lab, tiles, images, tissue, resin, z):
    rows = []
    dist = {k: ndimage.distance_transform_edt(lab != k) for k in range(1, len(tiles) + 1)}
    for a in range(1, len(tiles) + 1):
        for b in range(a + 1, len(tiles) + 1):
            sa = (lab == a) & (dist[b] >= NEAR) & (dist[b] <= FAR)
            sb = (lab == b) & (dist[a] >= NEAR) & (dist[a] <= FAR)
            if sa.sum() < 2000 or sb.sum() < 2000:
                continue
            ys, xs = np.nonzero(sa | sb)
            along = ys if np.ptp(ys) > np.ptp(xs) else xs
            axis = 0 if np.ptp(ys) > np.ptp(xs) else 1
            y0, x0 = ys.min(), xs.min()
            box = np.s_[y0:ys.max() + 1, x0:xs.max() + 1]
            sa_c, sb_c = sa[box], sb[box]
            imgs = {name: img[box] for name, img in images.items()}
            off = y0 if axis == 0 else x0
            for cls, m in (("tissue", tissue), ("resin", resin)):
                mc = m[box]
                ma, mb = sa_c & mc, sb_c & mc
                diffs = {name: [] for name in images}
                iqrs = {name: [] for name in images}
                for s0 in range(along.min(), along.max() + 1, STRETCH):
                    band = np.s_[s0 - off:s0 - off + STRETCH] if axis == 0 else np.s_[:, s0 - off:s0 - off + STRETCH]
                    pa, pb = ma[band], mb[band]
                    if pa.sum() < MIN_PX or pb.sum() < MIN_PX:
                        continue
                    for name, img in imgs.items():
                        va, vb = img[band][pa], img[band][pb]
                        diffs[name].append(float(np.median(vb)) - float(np.median(va)))
                        iqrs[name].append((_iqr(va) + _iqr(vb)) / 2)
                if diffs[next(iter(images))]:
                    rows.append({"z": z, "seam": f"{tiles[a - 1]}|{tiles[b - 1]}", "class": cls,
                                 "stretches": len(diffs[next(iter(images))]),
                                 **{name: round(float(np.median(d)), 1) for name, d in diffs.items()},
                                 **{f"{name}_iqr": round(float(np.median(d)), 1) for name, d in iqrs.items()}})
    return rows


def artefacts(images, tissue, resin, z):
    """Per image: tissue contrast, resin noise, resin unevenness and the halo of resin next to tissue."""
    resin_in = ndimage.binary_erosion(resin, iterations=3)
    d_tissue = ndimage.distance_transform_edt(~tissue)
    near, far = resin & (d_tissue >= 2) & (d_tissue <= 10), resin & (d_tissue >= 30) & (d_tissue <= 80)
    hb, wb = (s // BLOCK for s in resin.shape)
    frac = resin[:hb * BLOCK, :wb * BLOCK].reshape(hb, BLOCK, wb, BLOCK).mean(axis=(1, 3))
    good = np.argwhere(frac >= 0.8)
    rows = []
    for name, img in images.items():
        f = img.astype(np.float32)
        t = img[tissue]
        p25, p50, p75 = np.percentile(t, (25, 50, 75))
        hp = (f - cv2.blur(f, (3, 3)))[resin_in]
        noise = 1.4826 * float(np.median(np.abs(hp - np.median(hp)))) if hp.size else np.nan
        meds = [np.median(img[i * BLOCK:(i + 1) * BLOCK, j * BLOCK:(j + 1) * BLOCK][
                    resin[i * BLOCK:(i + 1) * BLOCK, j * BLOCK:(j + 1) * BLOCK]]) for i, j in good]
        spread = float(np.percentile(meds, 90) - np.percentile(meds, 10)) if len(meds) >= 20 else np.nan
        halo = float(np.median(img[near]) - np.median(img[far])) if near.sum() > 1000 and far.sum() > 1000 else np.nan
        rows.append({"z": z, "image": name, "tissue_p25": p25, "tissue_p50": p50, "tissue_p75": p75,
                     "tissue_iqr": p75 - p25, "resin_px": int(resin.sum()), "resin_median": float(np.median(img[resin])) if resin.any() else np.nan,
                     "resin_noise": round(noise, 2), "resin_blocks": len(meds), "resin_spread": spread, "halo": halo})
    return rows


def clean(lab):
    """Labels with stray pixels (removed by a 5 x 5 opening of their own label) set to 0 (no data)."""
    out = lab.copy()
    for k in range(1, int(lab.max()) + 1):
        m = lab == k
        out[m & ~ndimage.binary_opening(m, structure=np.ones((5, 5), bool))] = 0
    return out


def main():
    argv = [a for a in sys.argv if a != "--clean"]
    tidy = len(argv) < len(sys.argv)
    cfg = load_config(argv[1], step_defaults=deep_merge(render.DEFAULTS, shading.DEFAULTS))
    old, zs, prefix = argv[2], [int(v) for v in argv[3].split(",")], argv[4]
    dirs = dict(a.split("=", 1) for a in argv[5:])
    k, n = os.environ.get("SLURM_ARRAY_TASK_ID"), int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    if k is not None:
        zs = zs[int(k)::n]
    suffix = f".{int(k):02d}.csv" if k is not None else ".csv"
    seam_rows, art_rows = [], []
    for z in zs:
        stacks = {name: tifffile.imread(Path(d) / f"z{z}.tif") for name, d in dirs.items()}
        first = stacks[next(iter(dirs))]
        before = first[0]
        images = {"before": before, **{name: (s[1] if s.ndim == 3 else s) for name, s in stacks.items()}}
        hh = min(i.shape[0] for i in images.values())
        ww = min(i.shape[1] for i in images.values())
        images = {name: i[:hh, :ww] for name, i in images.items()}
        lab, tiles = ownership(cfg, old, z)
        lab = lab[:hh, :ww]
        if tidy:
            n0 = int((lab > 0).sum())
            lab = clean(lab)
            print(z, "stray ownership px dropped:", n0 - int((lab > 0).sum()), flush=True)
        tissue, resin = classes(images["before"])
        rows = steps(lab, tiles, images, tissue, resin, z)
        seam_rows += rows
        art_rows += artefacts(images, tissue, resin, z)
        print(z, len(rows), "seam rows", flush=True)
        pd.DataFrame(seam_rows).to_csv(prefix + "_seams" + suffix, index=False)
        pd.DataFrame(art_rows).to_csv(prefix + "_artefacts" + suffix, index=False)


if __name__ == "__main__":
    main()
