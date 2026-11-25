"""Flicker and tile matching predicted from the levels, over the whole run and around its breaks (segment changes,
tile layout changes, seams), for two or more versions of a run's levels before any of them is rendered.

    python dev/2026-11_a1001_flicker/scripts/breaks_flicker.py CONFIG OUT_DIR NAME=OUTPUT [NAME=OUTPUT ...]
                                                               [--half 200] [--window 9] [--step-slices 25]

OUTPUT is an ``intensity.output`` name (levels in work/intensity/<OUTPUT>/levels.csv), or ``-`` for the config's
own work/intensity/levels.csv.

Flicker, as ``tools/zflicker.py --config`` ("raw - windows"): per (z, tile) the grey the tile's tissue median
(preview t_p50) comes out at in its window, minus its running mean over --window slices; per slice the median over
tiles ("common": the slice-wide flicker the render will show) and each tile minus that ("own").

Tile matching: for every in-slice relation intensity balances the tiles from (``intensity.inslice_relations``:
with patchgrid the overlaps' patch bins on the corrected values u, else one relation per overlap), the grey
difference between the two tiles at the relation's centre m, (gain m + offset - lo_a) 255 / (hi_a - lo_a) minus
(m - lo_b) 255 / (hi_b - lo_b): the seam step the levels predict; per slice and tile pair the median over its
relations.

Breaks: every z where intensity's block changes (a new segment, and with intensity.break_at_seams a seam; the
blocks the smoothing and the joint solve restart at), each labelled segment / layout (tile grid or tile size
changes) / seam. Per break and version, in the --half slices either side: the common flicker's robust SD and its
three largest values, the own flicker's robust SD, the step across the break in the slice's tissue grey (median
over tiles; median over the --step-slices slices after minus the same before) and the predicted seam steps (median
and 95th percentile of |step|).

Writes OUT_DIR/breaks.csv, breaks_flicker.csv (a row per break and version), breaks_flicker.json (whole run per
version, and the blocks) and breaks_flicker.png (the common flicker around every break, versions overlaid).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from pipeline import intensity  # noqa: E402
from pipeline.config import load_config, step_defaults  # noqa: E402
from pipeline.slices import load_slices  # noqa: E402
from zflicker import residuals, robust_sd, summarise  # noqa: E402


def breaks_of(cfg, df):
    """DataFrame of the z where intensity's block changes, with what changes there."""
    sl = load_slices(cfg)
    per_z = sl.groupby("z").agg(rows=("tile_row", "max"), cols=("tile_col", "max"), h=("height", "median"),
                                w=("width", "median"), segment=("segment", "first"), seam=("seam", "any"))
    per_z["grid"] = (per_z["rows"] + 1).astype(str) + "x" + (per_z["cols"] + 1).astype(str)
    per_z["tile_size"] = per_z["h"].astype(int).astype(str) + "x" + per_z["w"].astype(int).astype(str)
    blocks = df.groupby("z")["block"].first().sort_index()
    rows = []
    for z in blocks.index[1:][blocks.diff().iloc[1:].to_numpy() != 0]:
        prev = per_z.index[per_z.index < z].max()
        a, b = per_z.loc[prev], per_z.loc[z]
        kinds = [k for k, on in (("segment", a["segment"] != b["segment"]),
                                 ("layout", a["grid"] != b["grid"] or a["tile_size"] != b["tile_size"]),
                                 ("seam", bool(b["seam"]))) if on]
        rows.append({"z": int(z), "kind": "+".join(kinds) or "block", "segment": f"{a['segment']}->{b['segment']}",
                     "grid": f"{a['grid']}->{b['grid']}" if a["grid"] != b["grid"] else b["grid"],
                     "tile_size": (f"{a['tile_size']}->{b['tile_size']}" if a["tile_size"] != b["tile_size"]
                                   else b["tile_size"])})
    return pd.DataFrame(rows)


def flicker(df, lv, window):
    """(per (z, tile) residuals with 'own', per-z common residual, per-z tissue grey) for levels ``lv``."""
    d = df[["z", "tile", "t_p50"]].merge(lv, on=["z", "tile"], how="inner")
    d["value"] = (d["t_p50"] - d["lo"]) * 255.0 / (d["hi"] - d["lo"])
    level = d.groupby("z")["value"].median()
    r, common = residuals(d.rename(columns={"tile": "box"})[["z", "box", "value"]], window)
    return r, common.sort_index(), level.sort_index()


def seam_steps(rel, lv):
    """Per (z, a, b): the median over its relations of the predicted grey step between tile a and tile b."""
    w = lv.set_index(["z", "tile"])[["lo", "hi"]]
    a = w.reindex(pd.MultiIndex.from_arrays([rel["z"], rel["a"]])).to_numpy()
    b = w.reindex(pd.MultiIndex.from_arrays([rel["z"], rel["b"]])).to_numpy()
    m = rel["centre"].to_numpy(float)
    ga = (rel["gain"].to_numpy(float) * m + rel["offset"].to_numpy(float) - a[:, 0]) * 255.0 / (a[:, 1] - a[:, 0])
    gb = (m - b[:, 0]) * 255.0 / (b[:, 1] - b[:, 0])
    s = pd.DataFrame({"z": rel["z"].to_numpy(), "a": rel["a"].to_numpy(), "b": rel["b"].to_numpy(), "step": ga - gb})
    return s.dropna().groupby(["z", "a", "b"], as_index=False)["step"].median()


def step_across(level, z, n):
    before = level[(level.index < z) & (level.index >= z - n)]
    after = level[(level.index >= z) & (level.index < z + n)]
    return float(after.median() - before.median()) if len(before) and len(after) else float("nan")


def window_stats(z, half, r, common, level, steps, n_step):
    lo, hi = z - half, z + half
    c = common[(common.index >= lo) & (common.index < hi)]
    own = r.loc[(r["z"] >= lo) & (r["z"] < hi), "own"]
    st = steps.loc[(steps["z"] >= lo) & (steps["z"] < hi), "step"].abs()
    worst = c.abs().sort_values(ascending=False).head(3)
    return {"common_sd": round(robust_sd(c.to_numpy()), 3),
            "worst": "; ".join(f"z {int(k)}: {c[k]:+.2f}" for k in worst.index),
            "own_sd": round(robust_sd(own.to_numpy()), 3),
            "level_step": round(step_across(level, z, n_step), 2),
            "seam_median": round(float(st.median()), 2) if len(st) else float("nan"),
            "seam_p95": round(float(st.quantile(0.95)), 2) if len(st) else float("nan")}


def draw(brk, series, half, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(brk)
    cols = 4
    rows = max(1, -(-n // cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.4 * rows), squeeze=False, layout="constrained")
    for ax, (_, b) in zip(axes.flat, brk.iterrows()):
        z = b["z"]
        for name, (_, common, _) in series.items():
            c = common[(common.index >= z - half) & (common.index < z + half)]
            ax.plot(c.index, c.to_numpy(), lw=0.6, label=name)
        ax.axvline(z, color="0.4", lw=0.6, ls="--")
        ax.set_title(f"z {z}: {b['kind']} ({b['grid']})", fontsize=9, loc="left")
        ax.tick_params(labelsize=7)
    for ax in axes.flat[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"Predicted slice-wide flicker (greys; tissue median minus its running mean, median over tiles), "
                 f"+-{half} slices around each break", fontsize=10)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("out", type=Path)
    p.add_argument("versions", nargs="+", help="NAME=OUTPUT (intensity.output; - for work/intensity/levels.csv)")
    p.add_argument("--half", type=int, default=200)
    p.add_argument("--window", type=int, default=9)
    p.add_argument("--step-slices", type=int, default=25)
    args = p.parse_args(argv)
    cfg = load_config(args.config, step_defaults=step_defaults("intensity", "shading", "patchgrid"))
    df = intensity.load_inputs(cfg, ["t_p50"])
    brk = breaks_of(cfg, df)
    rel = intensity.inslice_relations(cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    brk.to_csv(args.out / "breaks.csv", index=False)
    print(brk.to_string(index=False), flush=True)
    blocks = df.groupby("block")["z"].agg(["min", "max", "count"])
    summary = {"config": Path(args.config).name, "half": args.half, "window": args.window,
               "blocks": [[int(r["min"]), int(r["max"])] for _, r in blocks.iterrows()], "versions": {}}
    rows, series = [], {}
    for spec in args.versions:
        name, _, output = spec.partition("=")
        path = intensity.levels_path(cfg, output="" if output in ("", "-") else output)
        lv = pd.read_csv(path, dtype={"tile": str})[["z", "tile", "lo", "hi"]]
        r, common, level = flicker(df, lv, args.window)
        steps = seam_steps(rel, lv)
        series[name] = (r, common, level)
        s = summarise(r, common)
        s["seam_median"] = round(float(steps["step"].abs().median()), 3)
        s["seam_p95"] = round(float(steps["step"].abs().quantile(0.95)), 3)
        s["levels"] = str(path)
        summary["versions"][name] = s
        print(f"{name}: {json.dumps(s)}", flush=True)
        for _, b in brk.iterrows():
            rows.append({"z": b["z"], "kind": b["kind"], "version": name,
                         **window_stats(b["z"], args.half, r, common, level, steps, args.step_slices)})
    table = pd.DataFrame(rows).sort_values(["z", "version"])
    table.to_csv(args.out / "breaks_flicker.csv", index=False)
    (args.out / "breaks_flicker.json").write_text(json.dumps(summary, indent=1))
    draw(brk, series, args.half, args.out / "breaks_flicker.png")
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(table.to_string(index=False))
    print(f"wrote {args.out}/breaks.csv, breaks_flicker.csv, .json, .png")


if __name__ == "__main__":
    main()
