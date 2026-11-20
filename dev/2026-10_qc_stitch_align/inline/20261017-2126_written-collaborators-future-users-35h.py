# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-17T21:26:17.629Z
# Ran: remote, via ssh
# Heredoc line: ssh -o BatchMode=yes -o ConnectTimeout=30 -o LogLevel=QUIET ${CLUSTER_LOGIN} 'O=${DEV_35H}; ls -la $O; cd ${DEV_CLUSTER_PATH}
# Assistant note before it: Written for: collaborators and future users of the 35h data (the segmentation collaborator, the lab, a methods section). They know EM but not this pipeline. To get every number and setting exactly right, I'll pull the facts from the actual run files rather than from memory.

import sys, json
sys.path.insert(0, ".")
import pandas as pd
O = "$O"
sl = pd.read_csv(f"{O}/work/check/slices.csv", dtype={"tile": str}, parse_dates=["timestamp"], keep_default_na=False)
print("rows", len(sl), "excluded", sl.excluded.astype(str).str.lower().isin(["true", "1"]).sum(), "files", sl.file.nunique())
ex = sl[sl.excluded.astype(str).str.lower().isin(["true", "1"])]
print(ex.groupby(["file", "exclude_reason"]).size().to_string() if len(ex) else "no excluded rows")
s = sl[~sl.excluded.astype(str).str.lower().isin(["true", "1"])]
for seg, g in s.groupby("segment"):
    print("segment", seg, "z", g.z.min(), "-", g.z.max(), "tiles", sorted(g.tile.unique()), "w x h", sorted(set(zip(g.width, g.height))), g.timestamp.min(), "->", g.timestamp.max())
print("seams (acquisition restarts):", s.drop_duplicates("z").seam.astype(str).str.lower().isin(["true","1"]).sum())
