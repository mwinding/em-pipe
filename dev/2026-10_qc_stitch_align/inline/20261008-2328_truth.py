# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-08T23:28:02.205Z
# Ran: local
# Heredoc line: R=${DEV_WORK}/realsize; cd ${EM_PIPE_ROOT} && ${EM_PIPE_ENV}/bin/python - <<EOF
# Assistant note before it: Preview's peak memory dropped from 12.7 GB to 1.7 GB, and the run was twice as fast. The remaining memory-mapped reads are small thumbnails. Now align, intensity, render and pyramid at real size:

import json, numpy as np, pandas as pd
t = json.load(open("$R/truth.json"))
drift = np.array(t["drift"], float)
a = pd.read_csv("$R/out/align/transforms.csv").sort_values("z")
est = a[["tx", "ty"]].to_numpy()
# montage -> aligned is +drift (up to a constant)
err = np.abs((est - est[0]) - (drift - drift[0]))
print("align max abs error vs truth (px):", err.max().round(3), " per z:", err.max(axis=1).round(2))
print(pd.read_csv("$R/out/align/residuals.csv").describe().loc[["mean","max"], ["n","rms_px","max_px"]].round(2).to_string())
lv = pd.read_csv("$R/out/intensity/levels.csv", dtype={"tile": str}); print(lv.groupby("tile")[["lo","hi"]].mean().round(0).to_string())
rj = json.load(open("$R/out/render/render.json")); print({k: rj[k] for k in rj if k in ("shape","voxel_nm","downsample","origin","canvas")})
import tensorstore as ts
from pipeline import omezarr
arr = omezarr.open_scale("$R/out/render/volume.ome.zarr", 0).read().result()
print("s0", arr.shape, arr.dtype, "nonzero frac", (arr > 0).mean().round(3), "mean", arr[arr > 0].mean().round(1))
