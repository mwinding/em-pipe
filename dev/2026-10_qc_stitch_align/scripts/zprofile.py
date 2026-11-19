"""Brightness through z of the 35h overviews A (current windows) and B (tissue-only windows).

1. rendered: per plane of a coarse pyramid level, tissue median and p5 / p95 grey (no-data 0 and
   bright resin > 240 left out);
2. expected from the windows: where each slice's tissue median sits in its window, (p50 - lo) / (hi - lo),
   median over tiles, for A (all pixels) and B (tissue only);
3. raw: per slice, median over tiles of the raw p0.5 / p50 / p99.5.
"""
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
TEST = Path(os.path.expandvars("${DEV_USER_AREA}/em-pipe_${DEV_PROJECT}_35h_masked-norm-test"))
vols = {"A current": MAIN / os.path.expandvars("${DEV_PREFIX}35h_32nm.ome.zarr"),
        "B tissue-only": TEST / os.path.expandvars("${DEV_PREFIX}35h_32nm_masked-norm.ome.zarr")}
LEVEL = 4   # 32 nm x 2^4 = 512 nm, also in z

statsA = pd.read_csv(MAIN / "work/preview/stats.csv", dtype={"tile": str}, parse_dates=["timestamp"])
statsB = pd.read_csv(TEST / "work/preview/stats.csv", dtype={"tile": str})
lvA = pd.read_csv(MAIN / "work/intensity/levels.csv", dtype={"tile": str})
lvB = pd.read_csv(TEST / "work/intensity/levels.csv", dtype={"tile": str})
t_of_z = statsA.groupby("z")["timestamp"].first()
zs = t_of_z.index.to_numpy()

fig, ax = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
rendered = {}
for (name, root), color in zip(vols.items(), ("#2a78d6", "#eb6834")):
    arr = omezarr.open_scale(root, LEVEL)
    v = arr.read().result()
    f = 4 * 2 ** LEVEL                  # raw slices per plane
    med, p5, p95, t = [], [], [], []
    for k, plane in enumerate(v):
        tissue = plane[(plane > 0) & (plane < 240)]
        if tissue.size < 100:
            continue
        z_mid = zs[min(len(zs) - 1, k * f + f // 2)]
        t.append(t_of_z.loc[z_mid])
        med.append(np.median(tissue)); p5.append(np.percentile(tissue, 5)); p95.append(np.percentile(tissue, 95))
    rendered[name] = pd.DataFrame({"t": t, "med": med, "p5": p5, "p95": p95})
    ax[0].plot(t, med, color=color, lw=1.5, label=f"{name}: tissue median")
    ax[0].fill_between(t, p5, p95, color=color, alpha=0.12, label=f"{name}: tissue p5-p95")
ax[0].set_ylabel("rendered grey (0-255)")
ax[0].legend(frameon=False, fontsize=8, ncol=2)

for (name, st, lv, col50), color in zip((("A current", statsA, lvA, "p50"), ("B tissue-only", statsB, lvB, "p50")),
                                        ("#2a78d6", "#eb6834")):
    m = lv.merge(st[["z", "tile", col50]], on=["z", "tile"])
    m["pos"] = (m[col50] - m["lo"]) / (m["hi"] - m["lo"])
    per_z = m.groupby("z")["pos"].median().rolling(25, center=True, min_periods=1).median()
    ax[1].plot(t_of_z.loc[per_z.index], per_z.to_numpy(), color=color, lw=1.2, label=f"{name} windows")
ax[1].set_ylabel("tissue median within window\n(0 = black point, 1 = white)")
ax[1].legend(frameon=False, fontsize=8)

raw = statsA.groupby("z")[["p0_5", "p50", "p99_5"]].median().rolling(25, center=True, min_periods=1).median()
for col, color in (("p0_5", "#52514e"), ("p50", "#2a78d6"), ("p99_5", "#898781")):
    ax[2].plot(t_of_z.loc[raw.index], raw[col], color=color, lw=1.2, label=f"raw {col}")
ax[2].set_ylabel("raw grey (median over tiles)")
ax[2].legend(frameon=False, fontsize=8)
seg_changes = statsA.merge(pd.read_csv(MAIN / "work/check/slices.csv", usecols=["z", "segment"]).drop_duplicates("z"), on="z")
for z in seg_changes.groupby("segment")["z"].min().to_numpy()[1:]:
    for a in ax:
        a.axvline(t_of_z.loc[z], color="#b5b4ae", lw=0.8, ls="--")
for a in ax:
    a.grid(color="#e6e5e0", lw=0.5)
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("35h brightness through z (dashed: tile layout changes)", x=0.01, ha="left")
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=100)
for name, r in rendered.items():
    n = len(r)
    thirds = [r["med"].iloc[: n // 5].median(), r["med"].iloc[2 * n // 5: 3 * n // 5].median(), r["med"].iloc[-n // 5:].median()]
    print(f"{name}: tissue median grey first fifth {thirds[0]:.0f}, middle fifth {thirds[1]:.0f}, last fifth {thirds[2]:.0f}; "
          f"range over the volume {r['med'].min():.0f}-{r['med'].max():.0f}")
