# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-25T23:19:55.319Z
# Ran: remote, via ssh
# Heredoc line: ssh -o BatchMode=yes -o ConnectTimeout=30 -o LogLevel=QUIET ${CLUSTER_LOGIN} 'O=${DEV_37I}; cd ${DEV_LAB_AREA}
# Assistant note before it: That's worth checking: the first 1,914 planes are the single-tile approach, which is roughly the first 60% of the stack. There are two possibilities: - **Expected:** the single tile images a smaller field than the 2×2 grid, so in parts of the 2×2 area the early planes have no data. - **A real proble

import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import pandas as pd, json
p = pd.read_csv(os.path.expandvars("$O/work/render/${DEV_PREFIX}37i_8nm/tiles.csv"), dtype={"tile": str})
for name, sel in (("single tile z<1914", p[p.z < 1914]), ("2x2 z>=1914", p[p.z >= 1914])):
    print(name, len(sel), "lo", sel.lo.describe()[["min","50%","max"]].round(0).tolist(), "hi", sel.hi.describe()[["min","50%","max"]].round(0).tolist(), "nan", int(sel.lo.isna().sum()))
m = json.load(open(os.path.expandvars("$O/work/render/${DEV_PREFIX}37i_8nm/render.json")))
print("patchgrid" in m, m.get("shading") is not None)
s = json.load(open("$O/work/patchgrid/shapes.json"))
print("R", s.get("R"))
st = pd.read_csv("$O/work/preview/stats.csv", dtype={"tile": str})
for name, sel in (("single", st[st.z < 1914]), ("2x2", st[st.z >= 1914])):
    print(name, "raw p0.5/p50/p99.5 medians", sel[["p0_5","p50","p99_5"]].median().round(0).tolist())
