# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-17T10:00:26.467Z
# Ran: local
# Heredoc line: mkdir -p ${DEV_OUT}/qc/35h_shading/tables && cd ${DEV_OUT}/qc/35h_shading/tables && scp -q -o LogLevel=QUIET ${CLUSTER_LOGIN}:${DEV_SHARED_AREA}/
# Assistant note before it: Results are in, and they change the picture. I'll do one more quick check before we discuss.

import pandas as pd, numpy as np
st = pd.read_csv("qc/35h_shading/tables/stats.csv", dtype={"tile": str})
lv = pd.read_csv("qc/35h_shading/tables/levels.csv", dtype={"tile": str}).set_index(["z", "tile"])
d = pd.read_csv("qc/35h_shading/seam_steps_sample.csv")
d = d[(d.stretches >= 4)]
bounds = [0, 828, 3279, 12529, 13771, 15378, 10**9]
st["seg"] = pd.cut(st.z, bounds, right=False, labels=range(6)).astype(int)
st = st.sort_values(["tile", "z"])
# the per-slice 'jump' as intensity computes it: tissue median minus its 51-slice running median, per tile and segment
st["jump"] = st["t_p50"] - st.groupby(["tile", "seg"])["t_p50"].transform(lambda s: s.rolling(51, center=True, min_periods=1).median())
st = st.set_index(["z", "tile"])
rows = []
for r in d.itertuples():
    a, b = r.seam.split("|")
    try:
        wa = lv.loc[(r.z, a), "hi"] - lv.loc[(r.z, a), "lo"]; wb = lv.loc[(r.z, b), "hi"] - lv.loc[(r.z, b), "lo"]
        ja, jb = st.loc[(r.z, a), "jump"], st.loc[(r.z, b), "jump"]
    except KeyError:
        continue
    pred = -(jb * 255 / wb - ja * 255 / wa)     # step the per-slice term alone puts into the display
    rows.append({"class": r._3, "z": r.z, "seam": r.seam, "now": r.now, "before": r.before, "jump_step": pred})
x = pd.DataFrame(rows)
for cls, g in x.groupby("class"):
    print(cls, "n", len(g), "corr(now, per-slice term) %.2f" % np.corrcoef(g.now, g.jump_step)[0, 1],
          "| median |per-slice term| %.1f" % g.jump_step.abs().median(),
          "| median |now - per-slice term| %.1f vs |now| %.1f" % ((g.now - g.jump_step).abs().median(), g.now.abs().median()))
print(x[x["class"] == "tissue"].reindex(x[x["class"] == "tissue"].now.abs().sort_values(ascending=False).index).head(12).round(1).to_string(index=False))
