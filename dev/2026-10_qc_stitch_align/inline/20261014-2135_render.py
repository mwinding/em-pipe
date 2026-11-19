# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-14T21:35:50.038Z
# Ran: remote, via ssh
# Heredoc line: ssh -o BatchMode=yes -o ConnectTimeout=20 -o LogLevel=QUIET ${CLUSTER_LOGIN} 'source ${EM_PIPE_ROOT}/slurm/env.sh >/dev/null 2>&1; cd ${DEV_LAB_AREA}
# Assistant note before it: Plane 206–207 of the 32 nm volume covers raw slices ~3370–3377. Each 32 nm plane averages 4 raw slices, so "double vision" there means one or more of those slices is misplaced relative to the others. That's in the 2×2 stretch where the tiles were moving relative to each other. I'll check the alignme

import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import pandas as pd, json
rj = json.load(open(os.path.expandvars("render/${DEV_PROJECT}_M09-D22-D23/render.json")))
for k in (205, 206, 207, 208): print("plane", k, rj["planes"][k])
t = pd.read_csv("align/transforms.csv"); r = pd.read_csv("align/residuals.csv")
print(t[(t.z >= 3362) & (t.z <= 3386)][["z", "timestamp", "tx", "ty"]].round(2).to_string(index=False))
w = r[(r.z_a >= 3362) & (r.z_a <= 3386)]
print(w[(w.rms_px > 3) | w.rejected].to_string(index=False))
print("pair rms median", w.rms_px.median().round(2), "max", w.rms_px.max().round(2))
s = pd.read_csv("stitch/tiles.csv"); s = s[(s.z >= 3350) & (s.z <= 3385)]
p = s.pivot_table(index="z", columns="tile", values=["tx", "ty"])
print(pd.DataFrame({f"{c}_{ax}": p[(ax, c)] - p[(ax, "0-0")] for c in ["0-1", "1-0", "1-1"] for ax in ["tx", "ty"]}).iloc[::3].round(1).to_string())
