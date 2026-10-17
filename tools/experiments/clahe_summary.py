"""Summary of clahe_seam_steps.py output (experiment, Oct 2026).

    python tools/experiments/clahe_summary.py SEAMS_CSV [ARTEFACTS_CSV] [--ref v5]

Seams with >= 4 stretches, corner contacts (tiles differing in both row and column) left out. Per
class and column: median |step|, seams with |step| > 10 and > 30 greys, number of seams; per segment
the tissue median |step|. Contrast-normalised: per seam |step| / IQR (the local inter-quartile range
next to the seam, <name>_iqr): its median, and the seams whose |step| / IQR exceeds 10 / M and 30 / M,
M = the reference column's median IQR (so the reference keeps about its raw counts).
"""
import argparse

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("seams")
p.add_argument("artefacts", nargs="?")
p.add_argument("--ref", default="v5")
args = p.parse_args()

d = pd.read_csv(args.seams)
d = d[d.stretches >= 4]
a = d.seam.str.split("|", expand=True)
d = d[(a[0].str[0] == a[1].str[0]) | (a[0].str[2] == a[1].str[2])]
cols = [c for c in d.columns if c not in ("z", "seam", "class", "stretches") and not c.endswith("_iqr")]
d["segment"] = pd.cut(d.z, [0, 828, 3279, 12529, 13771, 15378, 1e9], right=False, labels=range(6))
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
for cls in ("tissue", "resin"):
    g = d[d["class"] == cls]
    m_ref = g[f"{args.ref}_iqr"].median()
    rows = []
    for c in cols:
        x, iqr = g[c].abs(), g[f"{c}_iqr"]
        r = x / iqr
        rows.append({"column": c, "med|step|": round(x.median(), 1), ">10": int((x > 10).sum()), ">30": int((x > 30).sum()),
                     "seams": len(x), "med IQR": round(iqr.median(), 1),
                     "med|step|/medIQR": round(x.median() / iqr.median(), 3), "med(|step|/IQR)": round(r.median(), 3),
                     f">10/{m_ref:.0f}": int((r > 10 / m_ref).sum()), f">30/{m_ref:.0f}": int((r > 30 / m_ref).sum())})
    print(f"\n== {cls} ==")
    print(pd.DataFrame(rows).to_string(index=False))
    seg = g.groupby("segment", observed=True)[cols].agg(lambda s: round(float(s.abs().median()), 1))
    segn = g.groupby("segment", observed=True).apply(
        lambda s: pd.Series({c: round(float((s[c].abs() / s[f"{c}_iqr"]).median()), 3) for c in cols}))
    print(f"\n{cls} median |step| by segment:")
    print(seg.T.to_string())
    print(f"\n{cls} median |step|/IQR by segment:")
    print(segn.T.to_string())

if args.artefacts:
    t = pd.read_csv(args.artefacts)
    t["noise/iqr"] = t.resin_noise / t.tissue_iqr
    t["spread/iqr"] = t.resin_spread / t.tissue_iqr
    t["halo/iqr"] = t.halo / t.tissue_iqr
    agg = t.groupby("image", sort=False)[["tissue_p25", "tissue_p50", "tissue_p75", "tissue_iqr", "resin_median",
                                          "resin_noise", "resin_spread", "halo", "noise/iqr", "spread/iqr",
                                          "halo/iqr"]].median().round(3)
    print("\nartefacts (median over slices):")
    print(agg.to_string())
