# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-09T11:16:57.243Z
# Ran: local
# Heredoc line: O=${DEV_USER_AREA}/em-pipe_test-M10-D01-D02/output; ${EM_PIPE_ENV}/bin/python - <<EOF
# Assistant note before it: The two-day test is running again from stitch, now with Janelia's montage model. Here's what came out of this round: - **Your tiles aren't plain shifted copies.** They differ by small rotations and scales, which is why translation-only stitching failed. - **The model:** Janelia's affine pulled towar

import pandas as pd, numpy as np, glob
r = pd.read_csv("$O/align/residuals.csv")
near = r[(r.z_a >= 8590) & (r.z_b <= 8598)].sort_values(["z_a", "z_b"])
print(near.to_string(index=False))
# raw pairwise measurements around the pause: mean displacement pb - pa in montage coords per pair
m = []
for f in sorted(glob.glob("$O/align/matches/chunk_*.npz")):
    d = np.load(f)
    sel = (d["z_a"] >= 8590) & (d["z_b"] <= 8598)
    if sel.any():
        for za, zb in sorted(set(zip(d["z_a"][sel], d["z_b"][sel]))):
            k = sel & (d["z_a"] == za) & (d["z_b"] == zb)
            dd = np.median(d["pb"][k] - d["pa"][k], axis=0)
            m.append((int(za), int(zb), int(k.sum()), dd.round(2).tolist()))
for row in m: print("match", row)
