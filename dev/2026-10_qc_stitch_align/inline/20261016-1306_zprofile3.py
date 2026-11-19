# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-16T13:06:17.872Z
# Ran: remote, via ssh
# Heredoc line: cd ${DEV_WORK} && cat > zprofile3.py <<'EOF'
# Assistant note before it: Checking whether option 2 got queued and where all the jobs are:

"""Brightness through z of the 35h 32 nm overviews: per plane of a coarse level, the tissue median and
p5-p95 (no data 0 and resin > 240 left out); steps at the tile layout changes printed."""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.expandvars("${EM_PIPE_ROOT}"))
from pipeline import omezarr  # noqa: E402

MAIN = Path(os.path.expandvars("${DEV_35H}"))
VOLS = {"4 median-anchored": MAIN / os.path.expandvars("${DEV_PREFIX}35h_32nm.ome.zarr"),
        "2 tissue 1-99%": Path(os.path.expandvars("${DEV_USER_AREA}/em-pipe_${DEV_PROJECT}_35h_norm-tissue1-99/${DEV_PREFIX}35h_32nm_tissue1-99.ome.zarr")),
        "old windows (test B)": Path(os.path.expandvars("${DEV_USER_AREA}/em-pipe_${DEV_PROJECT}_35h_masked-norm-test/${DEV_PREFIX}35h_32nm_masked-norm.ome.zarr"))}
LEVEL = 4
sl = pd.read_csv(MAIN / "work/check/slices.csv", usecols=["z", "timestamp", "segment"], parse_dates=["timestamp"]).drop_duplicates("z")
t_of_z = sl.set_index("z")["timestamp"]
changes = sl.groupby("segment")["z"].min().sort_values().to_numpy()[1:]
fig, ax = plt.subplots(figsize=(15, 6))
for (name, root), color in zip(VOLS.items(), ("#2a78d6", "#eb6834", "#898781")):
    v = omezarr.open_scale(root, LEVEL).read().result()
    f = 4 * 2 ** LEVEL
    rows = []
    for k, plane in enumerate(v):
        tissue = plane[(plane > 0) & (plane < 240)]
        if tissue.size >= 100:
            z = int(min(t_of_z.index[-1], k * f + f // 2))
            rows.append((z, np.median(tissue), np.percentile(tissue, 5), np.percentile(tissue, 95)))
    r = pd.DataFrame(rows, columns=["z", "med", "p5", "p95"])
    t = t_of_z.reindex(r["z"]).to_numpy()
    ax.plot(t, r["med"], color=color, lw=1.5, label=f"{name}: tissue median")
    ax.fill_between(t, r["p5"], r["p95"], color=color, alpha=0.10)
    steps = []
    for z in changes:
        b = r.loc[(r.z < z) & (r.z >= z - 400), "med"].median()
        a = r.loc[(r.z >= z) & (r.z < z + 400), "med"].median()
        steps.append(a - b)
    print(f"{name}: median grey {r['med'].median():.0f} (range {r['med'].min():.0f}-{r['med'].max():.0f}, sd {r['med'].std():.1f}); "
          f"steps at layout changes {', '.join(f'{s:+.0f}' for s in steps)}; tissue p5-p95 span {np.median(r.p95 - r.p5):.0f}")
for z in changes:
    ax.axvline(t_of_z.loc[z], color="#b5b4ae", lw=0.8, ls="--")
ax.set_ylabel("rendered grey (0-255), tissue")
ax.set_title("35h brightness through z, 32 nm overviews (band: tissue p5-p95; dashed: tile layout changes)", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=9)
ax.grid(color="#e6e5e0", lw=0.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=100)
