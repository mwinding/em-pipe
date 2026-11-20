# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-17T20:45:24.079Z
# Ran: remote, via ssh
# Heredoc line: Q=${DEV_OUT}/qc/35h_shading/v6; mkdir -p $Q && R=${CLUSTER_LOGIN}:${DEV_35H} && scp -q -o LogLevel=QUIET
# Assistant note before it: Everything finished. I'll pull the results and look before reporting:

import pandas as pd
d = pd.read_csv("qc/35h_shading/v6/seam_steps_v6.csv")
print("rows", len(d), "slices", d.z.nunique())
d = d[d.stretches >= 4]
a = d.seam.str.split("|", expand=True)
d = d[(a[0].str[0] == a[1].str[0]) | (a[0].str[2] == a[1].str[2])]
seg = pd.cut(d.z, [0, 828, 3279, 12529, 13771, 15378, 1e9], right=False, labels=range(6)).astype(int)
for cls, g in d.groupby("class"):
    print(cls, "seams", len(g))
    for c in ("before", "v5", "patchgrid"):
        print(f"   {c:9s} median {g[c].abs().median():5.1f}  >10: {int((g[c].abs() > 10).sum()):3d}  >30: {int((g[c].abs() > 30).sum()):2d}   per segment {g[c].abs().groupby(seg[g.index]).median().round(1).tolist()}")
