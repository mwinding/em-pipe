# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-17T09:59:55.260Z
# Ran: local
# Heredoc line: ${EM_PIPE_ENV}/bin/python - <<'EOF'
# Assistant note before it: Will do. I'll report the results as soon as the seam measurement finishes.

import pandas as pd, numpy as np
d = pd.read_csv("qc/35h_shading/seam_steps_sample.csv")
bounds = [0, 828, 3279, 12529, 13771, 15378, 10**9]
d["seg"] = pd.cut(d.z, bounds, right=False, labels=range(6)).astype(int)
a, b = d.seam.str.split("|", expand=True)[0], d.seam.str.split("|", expand=True)[1]
ra, ca = a.str[0].astype(int), a.str[2].astype(int); rb, cb = b.str[0].astype(int), b.str[2].astype(int)
d["kind"] = np.where((ra == rb), "vertical seam", np.where(ca == cb, "horizontal seam", "corner"))
d = d[(d.kind != "corner") & (d.stretches >= 4)]
print("seams with >= 4 stretches, no corners:", len(d))
g = d.groupby(["class", "seg"])
print(g.apply(lambda x: pd.Series({"n": len(x), "before": x.before.abs().median(), "now": x.now.abs().median(),
      ">30 before": (x.before.abs() > 30).sum(), ">30 now": (x.now.abs() > 30).sum()})).to_string())
print()
print(d.groupby(["class", "kind"]).apply(lambda x: pd.Series({"n": len(x), "before": x.before.abs().median(), "now": x.now.abs().median()})).to_string())
print("\nper seam & segment (tissue): median signed step over the sampled z, and its spread")
t = d[d["class"] == "tissue"].groupby(["seg", "seam"]).agg(n=("z", "size"), before=("before", "median"), now=("now", "median"),
      now_iqr=("now", lambda s: s.quantile(.75) - s.quantile(.25)))
print(t[t.n >= 2].round(1).to_string())
print("\nlargest |now| tissue steps:")
print(d[d["class"] == "tissue"].reindex(d[d["class"] == "tissue"].now.abs().sort_values(ascending=False).index).head(15).to_string(index=False))
