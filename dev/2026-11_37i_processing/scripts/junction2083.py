"""Frame fit across the z 2082 -> 2083 layout change, with and without the pairs involving z 2083."""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import glob
import sys

import numpy as np

from pipeline import align
from pipeline.config import load_config, step_path
from pipeline.slices import load_slices

cfg = load_config(os.path.expandvars("configs/lab/${DEV_PREFIX}37i.yaml"), step_defaults=align.DEFAULTS)
a = cfg["align"]
df = load_slices(cfg)
df = df[(df.z >= 1990) & (df.z <= 2200)]
stitch = align._stitch_transforms(cfg, df)
path = [p for p in glob.glob(str(step_path(cfg, "align", "matches")) + "/*.npz") if "002000" in p.rsplit("/", 1)[-1]]
print("chunk:", path, file=sys.stderr)
f = np.load(path[0])
za, zb = f["z_a"], f["z_b"]
cross = (za <= 2082) & (zb >= 2083)
pa = align._to_montage(stitch, za[cross], f["ta"][cross], f["tiles"], f["qa"][cross])
pb = align._to_montage(stitch, zb[cross], f["tb"][cross], f["tiles"], f["qb"][cross])
ta, tb = f["tiles"][f["ta"][cross]] if f["tiles"].ndim == 1 else f["ta"][cross], f["tb"][cross]
za, zb = za[cross], zb[cross]
keys = za.astype(np.int64) * 100000 + zb
uk, pid = np.unique(keys, return_inverse=True)
gap = (uk % 100000) - (uk // 100000)
fit = dict(reject_px=float(a["reject_px"]), iterations=int(a["robust_iterations"]))
print("pairs across the change:", sorted(set(zip(za.tolist(), zb.tolist()))), file=sys.stderr)
for label, sel in (("all pairs (as now)", np.ones(len(za), bool)),
                   ("without z 2083", (za != 2083) & (zb != 2083)),
                   ("only pairs with z 2083", (za == 2083) | (zb == 2083))):
    A, used, err = align._fit_frame(pb[sel], pa[sel], pid[sel], gap, "affine", **fit)
    d = align._describe(A)
    print(f"{label}: {sel.sum()} points, {used.sum()} used, median {np.median(err[used]):.2f} px, "
          f"p90 {np.percentile(err[used], 90):.2f} px; scale x {d['scale_x']:.5f} y {d['scale_y']:.5f} "
          f"rot {d['rotation_deg']:.4f} deg", file=sys.stderr)
    if label == "without z 2083":
        A0 = A
# residuals of the z 2083 points under the fit made without them, per tile of z 2083
sel = zb == 2083
err = np.linalg.norm(np.c_[pb[sel], np.ones(sel.sum())] @ np.asarray(A0)[:2].T - pa[sel], axis=1)
tiles = f["tb"][cross][sel]
for t in np.unique(tiles):
    e = err[tiles == t]
    print(f"  z 2083 tile index {t} ({f['tiles'][t] if f['tiles'].ndim == 1 else t}): {len(e)} points, "
          f"median {np.median(e):.1f} px, p90 {np.percentile(e, 90):.1f} px", file=sys.stderr)
