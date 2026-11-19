"""Tile jumps and alignment problems over a whole run: qc/tile_jumps.png plus a text summary."""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

R = Path(sys.argv[1])
W = R / "work"
t = pd.read_csv(W / "stitch/tiles.csv", dtype={"tile": str}, parse_dates=["timestamp"])
lay = json.loads((W / "stitch/layout.json").read_text())["segments"]
res = pd.read_csv(W / "align/residuals.csv")
ch = pd.read_csv(W / "align/segment_changes.csv")
ts = t.groupby("z")["timestamp"].first()
sampled = {int(p.stem[1:]) for p in (W / "stitch/samples").glob("z*.json")}

# Slice-to-slice tile movement (relative to the reference tile), within each segment
t = t.sort_values(["segment", "tile", "z"])
t["dx"] = t.groupby(["segment", "tile"])["tx"].diff()
t["dy"] = t.groupby(["segment", "tile"])["ty"].diff()
t["d"] = np.hypot(t["dx"], t["dy"])
step = t.groupby("z").agg(d=("d", "max"), seg=("segment", "first")).dropna()
step["time"] = ts.loc[step.index].to_numpy()
jumps = step[step["d"] > 5]
t["ox"] = t["tx"] - t.groupby(["segment", "tile"])["tx"].transform("median")
t["oy"] = t["ty"] - t.groupby(["segment", "tile"])["ty"].transform("median")

print("SEGMENTS")
for s, L in lay.items():
    g = step[step.seg == int(s)]
    n_z = t[t.segment == int(s)]["z"].nunique()
    n_s = len([z for z in sampled if L["z_first"] <= z <= L["z_last"]])
    a, b = ts.loc[L["z_first"]], ts.loc[L["z_last"]]
    print(f"  {s}: z {L['z_first']}-{L['z_last']} ({n_z} slices, {a:%d %b %H:%M} - {b:%d %b %H:%M}), "
          f"{L['grid_shape'][0]}x{L['grid_shape'][1]} of {L['tile_shape'][0]}x{L['tile_shape'][1]}, "
          f"{n_s} slices stitched, jumps >5 px: {(g.d > 5).sum()}, >15 px: {(g.d > 15).sum()}, "
          f"largest {g.d.max():.0f} px, failed samples {len(L['failed_samples'])}")
print("LAYOUT CHANGES")
for _, c in ch.iterrows():
    print(f"  z {c.z_before}->{c.z_after} ({ts.loc[c.z_after]:%d %b %H:%M}): median {c.median_px:.1f} px, "
          f"p90 {c.p90_px:.1f} px, scale x {c.scale_x:.5f} y {c.scale_y:.5f}")
rej = res[res.rejected]
print(f"ALIGN: {len(res)} slice pairs, {len(rej)} rejected ({100 * len(rej) / len(res):.2f}%), "
      f"median pair rms {res.rms_px.median():.2f} px")
clusters = []
for z in np.unique(np.r_[rej.z_a, rej.z_b]):
    if clusters and z - clusters[-1][-1] <= 5:
        clusters[-1].append(z)
    else:
        clusters.append([z])
cl = []
for c in clusters:
    sub = rej[(rej.z_a >= c[0]) & (rej.z_b <= c[-1])]
    cl.append((c[0], c[-1], len(sub), sub.rms_px.max() if len(sub) else 0))
cl.sort(key=lambda x: -x[2])
print(f"  {len(cl)} stretches with rejected pairs; largest:")
for a, b, n, m in cl[:12]:
    print(f"    z {a}-{b} ({ts.loc[a]:%d %b %H:%M}): {n} pairs, max rms {m:.0f} px")
day = jumps.groupby(jumps.time.dt.date).size()
print("JUMPS (>5 px) PER DAY:", " ".join(f"{d:%d/%m}:{n}" for d, n in day.items()))

fig, ax = plt.subplots(4, 1, figsize=(16, 12), sharex=True, height_ratios=[2, 2, 1.4, 1.4])
tiles = sorted(t.tile.unique(), key=lambda s: tuple(map(int, s.split("-"))))
cmap = plt.get_cmap("tab10")
for i, tile in enumerate(tiles):
    first = t[t.tile == tile].segment.min()
    for s, g in t[t.tile == tile].groupby("segment"):
        g = g.sort_values("z")
        ax[0].plot(g.timestamp, g.ox.clip(-80, 80), lw=0.8, color=cmap(i % 10), label=tile if s == first else None)
        ax[1].plot(g.timestamp, g.oy.clip(-80, 80), lw=0.8, color=cmap(i % 10))
ax[0].set_ylabel("tile x offset (px)\nvs segment median")
ax[1].set_ylabel("tile y offset (px)\nvs segment median")
ax[0].legend(ncol=9, fontsize=8, frameon=False, loc="lower left",
             title="tile (row-col); 0-0 is the reference", title_fontsize=8)
ax[2].vlines(jumps.time, 0, jumps.d.clip(upper=100), lw=0.8,
             color=np.where(jumps.d > 15, "#d03b3b", "#eb6834"))
ax[2].set_ylabel("tile jump between\nconsecutive slices (px)")
ax[2].set_ylim(0, 100)
ok = res[~res.rejected]
ax[3].plot(ts.reindex(ok.z_a).to_numpy(), ok.rms_px, ".", ms=1, color="#2a78d6", alpha=0.4, label="slice pair rms")
ax[3].plot(ts.reindex(rej.z_a).to_numpy(), rej.rms_px.clip(upper=100), "x", ms=4, color="#d03b3b",
           label=f"rejected pair ({len(rej)})")
ax[3].set_yscale("log")
ax[3].set_ylabel("alignment residual (px)")
ax[3].legend(frameon=False, fontsize=8, loc="upper left")
for s, L in lay.items():
    x = ts.loc[L["z_first"]]
    for a in ax:
        a.axvline(x, color="#898781", lw=0.8, ls="--")
    ax[0].text(x, 82, f" {L['grid_shape'][0]}x{L['grid_shape'][1]}, {L['tile_shape'][0]}x{L['tile_shape'][1]}",
               fontsize=8, va="bottom", color="#52514e")
for a in ax:
    a.grid(color="#e6e5e0", lw=0.5)
    a.spines[["top", "right"]].set_visible(False)
ax[3].xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
fig.suptitle(f"35h: tile jumps and alignment problems ({len(jumps)} slice steps with a tile jump > 5 px; "
             "dashed: tile layout changes)", x=0.01, ha="left")
fig.tight_layout()
out = R / "qc/tile_jumps.png"
fig.savefig(out, dpi=110)
print("plot:", out)
