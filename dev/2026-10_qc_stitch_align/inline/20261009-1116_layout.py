# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-09T11:16:31.403Z
# Ran: remote, via ssh
# Heredoc line: O=${DEV_USER_AREA}/em-pipe_test-M10-D01-D02/output; ${EM_PIPE_ENV}/bin/python - <<EOF
# Assistant note before it: The two-day test is running again from stitch, now with Janelia's montage model. Here's what came out of this round: - **Your tiles aren't plain shifted copies.** They differ by small rotations and scales, which is why translation-only stitching failed. - **The model:** Janelia's affine pulled towar

import json, pandas as pd, numpy as np
L = json.load(open("$O/stitch/layout.json"))["segments"]
for k, s in L.items():
    print("segment", k, {x: s[x] for x in ("model", "mode", "max_deviation_px", "n_samples", "n_good_samples", "failed_samples", "overlap_px", "row_axis")})
    for t, v in s["tiles"].items(): print("   tile", t, {kk: (round(vv, 1) if isinstance(vv, float) else vv) for kk, vv in v.items()})
p = pd.read_csv("$O/stitch/pairs.csv", dtype={"tile_a": str, "tile_b": str})
print(p.groupby(["tile_a", "tile_b"])[["n_inliers", "residual_px", "solve_residual_px"]].median().round(2).to_string())
a = pd.read_csv("$O/align/transforms.csv").sort_values("z")
d = a[["tx", "ty"]].diff().abs()
print("align: total drift range tx %.1f..%.1f ty %.1f..%.1f px" % (a.tx.min(), a.tx.max(), a.ty.min(), a.ty.max()))
print("align: per-slice step median %.2f px, p99 %.2f, max %.2f at z %d" % (np.hypot(d.tx, d.ty).median(), np.nanpercentile(np.hypot(d.tx, d.ty), 99), np.hypot(d.tx, d.ty).max(), a.z.iloc[np.nanargmax(np.hypot(d.tx, d.ty).values)]))
print("align: step at pause z 8594:", a.set_index("z").loc[[8593, 8594, 8595], ["tx", "ty"]].round(1).values.tolist())
r = pd.read_csv("$O/align/residuals.csv")
print("align residuals: rms median %.2f p95 %.2f, rejected pairs %d of %d" % (r.rms_px.median(), r.rms_px.quantile(.95), r.rejected.sum(), len(r)))
rj = json.load(open("$O/render/volume/render.json")); print("render:", rj["shape"], "voxel", [round(v, 2) for v in rj["voxel_nm"]], "origin", rj["origin_xy"])
