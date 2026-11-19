"""Merge the z-step profile tasks: decorrelation length per base z, pause junctions, and a plot."""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import glob
import json

import numpy as np
import pandas as pd

S = os.path.expandvars("${DEV_37I}")
OUT_ROOT = __import__('os').environ.get('DEV_OUT', 'dev_out')  # outputs: a directory you choose, never the dataset folder
OUT = f"{OUT_ROOT}/qc/zstep_profile"
KS = np.array([1, 2, 3, 4, 6, 8], float)
THR = (0.15, 0.1)

rows, pauses = [], []
for p in sorted(glob.glob(f"{OUT}/task_*.jsonl")):
    for line in open(p):
        r = json.loads(line)
        (pauses if "pause" in r else rows).append(r)


def length(curve, ks, thr):
    """Slices apart at which the similarity falls to thr (linear in k between samples)."""
    v = np.array([np.nan if curve[str(int(k))][0] is None else curve[str(int(k))][0] for k in ks])
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return np.nan
    v, k = v[ok], ks[ok]
    if v[0] < thr:
        return float(k[0] * 0.5)
    below = np.where(v < thr)[0]
    if not len(below):
        return np.inf
    i = below[0]
    return float(np.interp(thr, [v[i], v[i - 1]], [k[i], k[i - 1]]))


tab = pd.DataFrame([{"z": r["z"], **{f"k{int(k)}": r["curve"][str(int(k))][0] for k in KS},
                     "win": r["curve"]["1"][1], "detail": r["detail"],
                     **{f"L{t}": length(r["curve"], KS, t) for t in THR}} for r in rows]).sort_values("z")
tab.to_csv(f"{OUT}/profile.csv", index=False)
d = pd.read_csv(f"{S}/work/check/slices.csv")
t = pd.to_datetime(d[~d.excluded].groupby("z").timestamp.min())
tab["date"] = t.reindex(tab.z).dt.strftime("%m-%d").values
pd.set_option("display.width", 200)
print(f"{len(tab)} bases, {tab['L0.1'].isna().sum()} without enough tissue")
# Rolling summary in 512-slice bins
tab["bin"] = tab.z // 512 * 512
summ = tab.groupby("bin").agg(date=("date", "first"), k1=("k1", "median"), k2=("k2", "median"),
                              k4=("k4", "median"), L15=("L0.15", "median"), L10=("L0.1", "median"),
                              detail=("detail", "median"), n=("z", "size"))
print(summ.round(3).to_string())
print("\nfirst 400 slices after each pause (every base):")
for zp in (2083, 5559, 5782, 11014, 17333):
    sel = tab[(tab.z >= zp - 200) & (tab.z < zp + 400)]
    print(f"  pause z {zp}: " + ", ".join(f"{int(z)}:{l:.1f}" for z, l in zip(sel.z, sel["L0.1"])))
print("\npause junctions (similarity across the pause, and the distance it equals on each side's curve):")
ks8 = np.arange(1, 9, dtype=float)
for r in sorted(pauses, key=lambda r: r["pause"]):
    zp = r["pause"]
    acr = r["across"][f"{zp - 1}->{zp}"][0]
    out = []
    for side in ("before", "after"):
        v = np.array([np.nan if r[side][str(k)][0] is None else r[side][str(k)][0] for k in range(1, 9)])
        if acr is None or not np.isfinite(v).any():
            out.append(f"{side}: n/a")
            continue
        if acr >= v[0]:
            out.append(f"{side}: <= 1 slice")
        elif acr < np.nanmin(v):
            out.append(f"{side}: > 8 slices")
        else:
            out.append(f"{side}: like {np.interp(-acr, -v, ks8):.1f} slices")
    cur = {s: " ".join(f"{r[s][str(k)][0]:.2f}" if r[s][str(k)][0] is not None else "-" for k in range(1, 9))
           for s in ("before", "after")}
    print(f"  z {zp - 1}->{zp}: {acr if acr is None else round(acr, 3)}; " + "; ".join(out)
          + f"  | detail {r['detail_before']:.3f} -> {r['detail_after']:.3f}")
    print(f"      before k1-8: {cur['before']}\n      after  k1-8: {cur['after']}")
    print("      across: " + ", ".join(f"{k} {v[0]:.3f}" for k, v in r["across"].items() if v[0] is not None))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
for a, col, lab in zip(ax, ["L0.1", "k1", "detail"],
                       ["slices to similarity 0.1\n(more = thinner slices)", "similarity of neighbours (k=1)",
                        "fine detail (blur check)"]):
    y = tab[col].replace(np.inf, np.nan)
    a.plot(tab.z, y, ".", ms=3, color="#888")
    a.plot(tab.z, y.rolling(9, center=True, min_periods=3).median(), color="#1f5fa8", lw=1.5)
    a.set_ylabel(lab)
    for zp in (2083, 5559, 11014, 17333):
        a.axvline(zp, color="#c0392b", lw=0.8, ls="--")
    a.axvline(5782, color="#999", lw=0.8, ls=":")
ax[-1].set_xlabel("z (slice); red dashed = acquisition pauses, grey dotted = clock change")
fig.tight_layout()
fig.savefig(f"{OUT}/profile.png", dpi=110)
print(f"\nwrote {OUT}/profile.csv and profile.png")
