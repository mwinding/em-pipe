# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-20T09:27:41.737Z
# Ran: remote, via ssh
# Heredoc line: ssh -o BatchMode=yes -o ConnectTimeout=30 -o LogLevel=QUIET ${CLUSTER_LOGIN} 'O=${DEV_37I}; head -1 $O/work/preview/stats.
# Assistant note before it: Both volumes go into one new folder on the lab share, next to 35h's: `${DEV_37I}/` (on the Mac: `${DEV_37I}/`) When finished, it will contain: - `EM05

import pandas as pd
st = pd.read_csv("$O/work/preview/stats.csv", dtype={"tile": str})
sl = pd.read_csv("$O/work/check/slices.csv", dtype={"tile": str})[["z", "tile", "segment"]] if "segment" in pd.read_csv("$O/work/check/slices.csv", nrows=1).columns else None
cols = [c for c in st.columns if c.startswith("p") or c in ("mean", "max", "min", "sat", "frac_max", "n_max")]
print(cols)
if sl is not None:
    st = st.merge(sl, on=["z", "tile"])
    print(st.groupby("segment")[cols].median().round(0).to_string())
else:
    st["seg"] = pd.cut(st["z"], [-1, 185, 1913, 1927, 99999], labels=[0, 1, 2, 3])
    print(st.groupby("seg")[cols].median().round(0).to_string())
