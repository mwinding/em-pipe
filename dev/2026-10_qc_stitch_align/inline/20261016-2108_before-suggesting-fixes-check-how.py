# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-16T21:08:13.412Z
# Ran: remote, via ssh
# Heredoc line: ssh -o BatchMode=yes -o ConnectTimeout=30 -o LogLevel=QUIET ${CLUSTER_LOGIN} '${EM_PIPE_ENV}/bin/python -I - <<EOF
# Assistant note before it: Before suggesting fixes, I'll check how the intensity step balances tiles, because all three cases look like tile balancing going wrong.

import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import pandas as pd
O = os.path.expandvars("${DEV_35H}/work")
lv = pd.read_csv(f"{O}/intensity/levels.csv", dtype={"tile": str})
st = pd.read_csv(f"{O}/preview/stats.csv", dtype={"tile": str})
cols = [c for c in ["z","tile","lo","hi","g","o","jump"] if c in lv.columns]
tcols = [c for c in ["z","tile","p50","t_p1","t_p50","t_p99","tissue_frac","t_frac"] if c in st.columns]
print("levels cols:", list(lv.columns)); print("stats cols:", [c for c in st.columns][:40])
for z in (0, 2044, 3231, 17688):
    m = lv[lv.z == z][cols].merge(st[st.z == z][tcols], on=["z","tile"])
    print(m.round(3).to_string(index=False)); print()
