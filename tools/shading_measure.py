"""Shading within tiles: mean relative brightness by position in the tile, per tile layout segment,
for tissue AND plain resin (the pipeline's own qc/shading.png shows tissue only, which the fit uses).

    python tools/shading_measure.py CONFIG OUT_DIR

For every 20th slice and every tile: the preview thumbnail (1/16 scale) in raw units, divided by its
own tissue median (tissue = not background.mask) or its own bright-resin median (resin = background
and bright). Averaged per pixel position over the segment's slices. Writes OUT_DIR/shading_seg<k>.png
and profiles.csv (column profiles: mean over rows). Needs the preview step's thumbnails.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import background  # noqa: E402
from pipeline.config import load_config  # noqa: E402
from pipeline.preview import thumb_values  # noqa: E402

W = Path(load_config(sys.argv[1])["output_dir"]) / "work"
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
EVERY = 20

idx = pd.read_csv(W / "preview/thumbs/index.csv", dtype={"tile": str})
st = pd.read_csv(W / "preview/stats.csv", dtype={"tile": str})[["z", "tile", "p0_5", "p99_5"]]
sl = pd.read_csv(W / "check/slices.csv", dtype={"tile": str})
seg = sl.drop_duplicates("z").set_index("z")["segment"]
df = idx.merge(st, on=["z", "tile"])
df["segment"] = df["z"].map(seg)
df = df[df["z"] % EVERY == 0]
rows = []
for k, g in df.groupby("segment"):
    tiles = sorted(g["tile"].unique())
    acc = {}
    for r in g.itertuples():
        t8 = np.asarray(np.load(W / "preview/thumbs" / r.npy, mmap_mode="r")[int(r.i)])
        v = thumb_values(t8, r.p0_5, r.p99_5)
        bg = background.mask(t8)
        a = acc.setdefault(r.tile, {n: [np.zeros(t8.shape), np.zeros(t8.shape)] for n in ("tissue", "resin")})
        for name, m in (("tissue", ~bg), ("resin", bg & (t8 > 128))):
            if m.sum() > 2000:
                a[name][0][m] += v[m] / np.median(v[m])
                a[name][1][m] += 1
    nr = 1 + max(int(t.split("-")[0]) for t in tiles)
    nc = 1 + max(int(t.split("-")[1]) for t in tiles)
    fig, axes = plt.subplots(nr, 2 * nc, figsize=(4 * 2 * nc, 3.6 * nr + 0.8), squeeze=False)
    for t, a in acc.items():
        i, j = (int(x) for x in t.split("-"))
        for c, name in enumerate(("tissue", "resin")):
            s, n = a[name]
            m = np.where(n >= 5, s / np.maximum(n, 1), np.nan)
            ax = axes[i, 2 * j + c]
            im = ax.imshow(m, cmap="RdBu_r", vmin=0.9, vmax=1.1)
            ax.set_title(f"tile {t}: {name}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            prof = np.nanmean(m, axis=0)
            w = len(prof)
            row = {"segment": k, "tile": t, "kind": name}
            for p in (0, 10, 25, 50, 75, 90, 100):
                c0 = min(max(int(p / 100 * (w - 1)) - 5, 0), w - 10)
                row[f"x{p}"] = float(np.nanmean(prof[c0:c0 + 10]))
            rows.append(row)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="brightness relative to the thumbnail median")
    fig.suptitle(f"segment {k} ({len(tiles)} tiles, z {g.z.min()}-{g.z.max()}, every {EVERY}th slice): mean "
                 "relative brightness by position within each tile (left of each pair: tissue; right: plain resin)")
    fig.savefig(OUT / f"shading_seg{k}.png", dpi=70)
    plt.close(fig)
    print("segment", k, "done", flush=True)
pd.DataFrame(rows).round(3).to_csv(OUT / "profiles.csv", index=False)
