# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-11-13T22:14:25.339Z
# Ran: remote, via ssh
# Heredoc line: (whole command)
# Assistant note before it: The tile matching is fixed now, so what's left could be a real difference between the two halves. I'll check whether the left/right difference is also there in slices before the tissue split, where the tile matching is reliable.

ssh -o LogLevel=ERROR ${CLUSTER_LOGIN} 'cd ${DEV_37I}/work; ${EM_PIPE_ENV}/bin/python -c "
import pandas as pd, numpy as np
lv = pd.read_csv(\"intensity/levels.csv\", dtype={\"tile\": str})
st = pd.read_csv(\"preview/stats.csv\", dtype={\"tile\": str})
d = st.merge(lv, on=[\"z\", \"tile\"])
for q in (\"t_p25\", \"t_p50\", \"t_p75\"):
    d[q + \"_disp\"] = 255 * (d[q] - d.lo) / (d.hi - d.lo)
d[\"col\"] = np.where(d.tile.str[-1] == \"0\", \"A raw-left (display right)\", \"B raw-right (display left)\")
d = d[d.tile.str.contains(\"-\")]
print(\"tissue median and quartiles in display grey values (0-255), mean of the column tiles, plus tissue fraction\")
for z in (5000, 9600, 13000, 14500, 15100, 15400, 15800, 16200, 16400, 16530, 17000, 17500, 17900):
    g = d[d.z == z].groupby(\"col\")[[\"t_p25_disp\", \"t_p50_disp\", \"t_p75_disp\", \"tissue_fraction\"]].mean()
    if len(g) < 2: continue
    a, b = g.iloc[0], g.iloc[1]
    print(f\"z {z}: A p25/50/75 {a.t_p25_disp:.0f}/{a.t_p50_disp:.0f}/{a.t_p75_disp:.0f} (tissue {a.tissue_fraction:.2f})   B {b.t_p25_disp:.0f}/{b.t_p50_disp:.0f}/{b.t_p75_disp:.0f} (tissue {b.tissue_fraction:.2f})   B-A median {b.t_p50_disp - a.t_p50_disp:+.0f}\")
"'
