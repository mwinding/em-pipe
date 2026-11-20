# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-17T15:15:56.881Z
# Ran: local
# Heredoc line: cat ${DEV_WORK} | tail -16; cd ${EM_PIPE_ROOT} && ${EM_PIPE_ENV}/bin
# Assistant note before it: It's about 60 % through: the renders are done and the seam measurement has covered 32 of the 55 slices. Here's the picture so far (seams with enough tissue on both sides to measure): | | before | thumbnail balance | fixed overlap balance | |---|---|---|---| | tissue: median step | 12 | 11 | **4.5** 

import pandas as pd, numpy as np
d = pd.read_csv("qc/35h_shading/seam_steps_v5.csv")
d = d[d.stretches >= 4]
a = d.seam.str.split("|", expand=True)
d = d[(a[0].str[0] == a[1].str[0]) | (a[0].str[2] == a[1].str[2])]      # no corner contacts
seg = pd.cut(d.z, [0, 828, 3279, 12529, 13771, 15378, 10**9], right=False, labels=range(6)).astype(int)
cols = ["before", "thumbs", "overlaps_fixed"]
print("median |step| by segment (tissue):")
print(d[d["class"] == "tissue"].groupby(seg[d["class"] == "tissue"])[cols].agg(lambda s: round(s.abs().median(), 1)).to_string())
print("\nall, median |step| / >10 / >30:")
for cls, g in d.groupby("class"):
    print(cls, {c: (round(g[c].abs().median(), 1), int((g[c].abs() > 10).sum()), int((g[c].abs() > 30).sum())) for c in cols}, "of", len(g))
