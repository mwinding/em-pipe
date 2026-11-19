"""Tile jumps and alignment problems over a whole run: a plot and a text summary.

    python tools/tile_jumps.py --config C [--jump-px 5] [--big-px 15] [--out PNG]

Reads the stitch and align outputs of a finished run (work/stitch/tiles.csv, layout.json, samples/;
work/align/residuals.csv, segment_changes.csv; the slice list for times) and shows, against acquisition
time:
  1-2. each tile's x and y offset from its segment median (tile 0-0 is the reference), clipped at 80 px
  3.   the largest tile movement between consecutive slices, where it exceeds --jump-px (red over --big-px)
  4.   the alignment residual of every slice pair, rejected pairs marked
with tile-layout changes as dashed lines. The summary lists each segment (slices, layout, stitched
samples, jumps, failed samples), each layout change's fit, rejected align pairs grouped into stretches
(the largest first) and jumps per day. Stretches with many rejected pairs or large jumps are candidates
for align.local_windows. Writes <output_dir>/qc/tile_jumps.png (or --out). Read-only otherwise.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import load_config, step_defaults, step_path  # noqa: E402
from pipeline.slices import load_slices  # noqa: E402

COLORS = {"line": "#2a78d6", "jump": "#eb6834", "big": "#d03b3b", "grid": "#e6e5e0", "change": "#898781", "text": "#52514e"}


def load(cfg):
    t = pd.read_csv(step_path(cfg, "stitch", "tiles.csv"), dtype={"tile": str})
    layout = json.loads(step_path(cfg, "stitch", "layout.json").read_text())["segments"]
    res = pd.read_csv(step_path(cfg, "align", "residuals.csv"))
    res["rejected"] = res["rejected"].astype(str).str.lower().isin(["true", "1"])
    ch_path = step_path(cfg, "align", "segment_changes.csv")
    changes = pd.read_csv(ch_path) if ch_path.exists() and ch_path.stat().st_size else pd.DataFrame()
    sampled = {int(p.stem[1:]) for p in step_path(cfg, "stitch", "samples").glob("z*.json")}
    sl = load_slices(cfg)
    times = sl.groupby("z")["timestamp"].first()
    return t, layout, res, changes, sampled, times


def jumps_of(t, times):
    """Per z, the largest tile movement since the previous slice of the same segment."""
    t = t.sort_values(["segment", "tile", "z"]).copy()
    t["d"] = np.hypot(t.groupby(["segment", "tile"])["tx"].diff(), t.groupby(["segment", "tile"])["ty"].diff())
    step = t.groupby("z").agg(d=("d", "max"), seg=("segment", "first")).dropna()
    step["time"] = times.reindex(step.index).to_numpy()
    t["ox"] = t["tx"] - t.groupby(["segment", "tile"])["tx"].transform("median")
    t["oy"] = t["ty"] - t.groupby(["segment", "tile"])["ty"].transform("median")
    return t, step


def stretches(rej, gap=5):
    """Rejected pairs grouped into stretches of z no more than ``gap`` apart: (z first, z last, pairs, max rms)."""
    out, cur = [], []
    for z in np.unique(np.r_[rej["z_a"], rej["z_b"]]):
        if cur and z - cur[-1] > gap:
            out.append(cur)
            cur = []
        cur.append(int(z))
    if cur:
        out.append(cur)
    res = []
    for c in out:
        sub = rej[(rej["z_a"] >= c[0]) & (rej["z_b"] <= c[-1])]
        res.append((c[0], c[-1], len(sub), float(sub["rms_px"].max()) if len(sub) else 0.0))
    return sorted(res, key=lambda x: -x[2])


def summary(cfg, t, layout, res, changes, sampled, times, step, jump_px, big_px):
    lines = ["SEGMENTS"]
    for s, L in layout.items():
        g = step[step["seg"] == int(s)]
        n_z = t[t["segment"] == int(s)]["z"].nunique()
        n_s = len([z for z in sampled if L["z_first"] <= z <= L["z_last"]])
        a, b = times.get(L["z_first"]), times.get(L["z_last"])
        lines.append(f"  {s}: z {L['z_first']}-{L['z_last']} ({n_z} slices, {a:%d %b %H:%M} - {b:%d %b %H:%M}), "
                     f"{L['grid_shape'][0]}x{L['grid_shape'][1]} of {L['tile_shape'][0]}x{L['tile_shape'][1]}, "
                     f"{n_s} slices stitched, jumps >{jump_px:g} px: {(g['d'] > jump_px).sum()}, "
                     f">{big_px:g} px: {(g['d'] > big_px).sum()}, largest {g['d'].max() if len(g) else 0:.0f} px, "
                     f"failed samples {len(L.get('failed_samples', []))}")
    if len(changes):
        lines.append("LAYOUT CHANGES")
        for _, c in changes.iterrows():
            lines.append(f"  z {c['z_before']}->{c['z_after']}: median {c['median_px']:.1f} px, p90 {c['p90_px']:.1f} px, "
                         f"scale x {c['scale_x']:.5f} y {c['scale_y']:.5f}")
    rej = res[res["rejected"]]
    lines.append(f"ALIGN: {len(res)} slice pairs, {len(rej)} rejected ({100 * len(rej) / max(len(res), 1):.2f}%), "
                 f"median pair rms {res.loc[~res['rejected'], 'rms_px'].median():.2f} px")
    st = stretches(rej)
    lines.append(f"  {len(st)} stretches with rejected pairs; largest:")
    lines += [f"    z {a}-{b}: {n} pairs, max rms {m:.0f} px" for a, b, n, m in st[:12]]
    jumps = step[step["d"] > jump_px]
    day = jumps.groupby(pd.to_datetime(jumps["time"]).dt.date).size()
    lines.append(f"JUMPS (>{jump_px:g} px) PER DAY: " + " ".join(f"{d:%d/%m}:{n}" for d, n in day.items()))
    return "\n".join(lines)


def plot(path, name, t, layout, res, times, step, jump_px, big_px):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    jumps = step[step["d"] > jump_px]
    fig, ax = plt.subplots(4, 1, figsize=(16, 12), sharex=True, height_ratios=[2, 2, 1.4, 1.4])
    tiles = sorted(t["tile"].unique(), key=lambda s: tuple(map(int, s.split("-"))))
    cmap = plt.get_cmap("tab10")
    t = t.assign(time=times.reindex(t["z"]).to_numpy())
    for i, tile in enumerate(tiles):
        first = t[t["tile"] == tile]["segment"].min()
        for s, g in t[t["tile"] == tile].groupby("segment"):
            g = g.sort_values("z")
            ax[0].plot(g["time"], g["ox"].clip(-80, 80), lw=0.8, color=cmap(i % 10), label=tile if s == first else None)
            ax[1].plot(g["time"], g["oy"].clip(-80, 80), lw=0.8, color=cmap(i % 10))
    ax[0].set_ylabel("tile x offset (px)\nvs segment median")
    ax[1].set_ylabel("tile y offset (px)\nvs segment median")
    ax[0].legend(ncol=9, fontsize=8, frameon=False, loc="lower left", title="tile (row-col); 0-0 is the reference",
                 title_fontsize=8)
    ax[2].vlines(jumps["time"], 0, jumps["d"].clip(upper=100), lw=0.8,
                 color=np.where(jumps["d"] > big_px, COLORS["big"], COLORS["jump"]))
    ax[2].set_ylabel("tile jump between\nconsecutive slices (px)")
    ax[2].set_ylim(0, 100)
    ok, rej = res[~res["rejected"]], res[res["rejected"]]
    ax[3].plot(times.reindex(ok["z_a"]).to_numpy(), ok["rms_px"], ".", ms=1, color=COLORS["line"], alpha=0.4,
               label="slice pair rms")
    ax[3].plot(times.reindex(rej["z_a"]).to_numpy(), rej["rms_px"].clip(upper=100), "x", ms=4, color=COLORS["big"],
               label=f"rejected pair ({len(rej)})")
    ax[3].set_yscale("log")
    ax[3].set_ylabel("alignment residual (px)")
    ax[3].legend(frameon=False, fontsize=8, loc="upper left")
    for L in layout.values():
        x = times.get(L["z_first"])
        for a in ax:
            a.axvline(x, color=COLORS["change"], lw=0.8, ls="--")
        ax[0].text(x, 82, f" {L['grid_shape'][0]}x{L['grid_shape'][1]}, {L['tile_shape'][0]}x{L['tile_shape'][1]}",
                   fontsize=8, va="bottom", color=COLORS["text"])
    for a in ax:
        a.grid(color=COLORS["grid"], lw=0.5)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{name}: tile jumps and alignment problems ({len(jumps)} slice steps with a tile jump > {jump_px:g} px; "
                 "dashed: tile layout changes)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", help="override output_dir from the config")
    ap.add_argument("--jump-px", type=float, default=5.0, help="tile movement counted as a jump (default 5 px)")
    ap.add_argument("--big-px", type=float, default=15.0, help="jumps drawn red above this (default 15 px)")
    ap.add_argument("--out", help="plot path (default <output_dir>/qc/tile_jumps.png)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, None, a.output_dir, step_defaults())
    t, layout, res, changes, sampled, times = load(cfg)
    t, step = jumps_of(t, times)
    print(summary(cfg, t, layout, res, changes, sampled, times, step, a.jump_px, a.big_px))
    out = Path(a.out) if a.out else Path(cfg["output_dir"]) / "qc" / "tile_jumps.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plot(out, cfg.get("name") or "run", t, layout, res, times, step, a.jump_px, a.big_px)
    print("plot:", out)


if __name__ == "__main__":
    main()
