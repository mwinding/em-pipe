"""Raw detector statistics per tile, by class: tissue, resin, off-sample (and neuropil vs other tissue).

    python tools/raw_stats.py run   --config C [--slices N | --z Z,Z,...] [--factor F] [--out DIR] [--task-id K --num-tasks N]
    python tools/raw_stats.py merge --config C [--out DIR]
    python tools/raw_stats.py compare --out DIR [--resin resin|resin_strict] NAME=STATS_DIR [NAME=STATS_DIR ...]

What it measures: the raw 16-bit values of the full-resolution tiles, exactly as recorded (before
flat-field, tile matching or display windows), at N slices spread evenly through the selected slices
(5 % to 95 % of the stack) or at given z, every tile. Each tile is classified on a copy downsampled by
F (area mean, default 16: ~128 nm at 8 nm voxels), with the pipeline's background mask
(pipeline/background.py, on the copy scaled to its own 0.5-99.5 percentiles), and the classes are
scaled back to full resolution:

  tissue        not background
  resin         background minus dark areas touching the tile border (the "current" definition)
  resin_strict  the subset of resin in large areas (>= --min-resin-px at the downsampled scale) that
                reach the tile border (resin outside the sample, not enclosed lumens), with a rim of
                --rim px and a tile-edge band of --edge px left out (charging and shading are worst there)
  offsample     dark areas touching the tile border
  neuropil, other_tissue
                tissue split by the reference-tissue mask (pipeline/refmask.py) on the preview thumbnails,
                when that module is available (it isn't part of every em-pipe version); otherwise absent

Resin has NOT been validated as a reference: the background mask finds smooth, low-texture areas,
which can include smooth tissue; check the classes by eye before relying on resin values.

Outputs (default <output_dir>/qc/raw_stats/): hist.npz (pooled 65536-bin histograms per class),
tiles.csv (per z and tile: fraction of the tile and p1/p5/p25/p50/p75/p95/p99 per class), pooled.png
(pooled curves per class) and tiles.png (per tile: tissue p5-p95 as a line, tissue median as a dot,
resin median as a ring). Pooled curves mix tiles that sit at different raw levels, so they are wider
than any one tile; tiles.csv and tiles.png are per tile.

run reads the raw stacks (slow over a network share: run it on the cluster, slurm/raw_stats.sbatch).
With --num-tasks N, task K takes every N-th sampled slice and writes part files; merge combines them
and draws the plots (run without tasks does both). compare draws several datasets' runs together.
Read-only on the data; writes only to --out.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import background  # noqa: E402
from pipeline.cli import task_info  # noqa: E402
from pipeline.config import load_config, step_defaults  # noqa: E402
from pipeline.imagej_tiff import ImageJStack  # noqa: E402
from pipeline.slices import load_slices  # noqa: E402

try:  # the reference-tissue mask is only in some em-pipe versions
    from pipeline import refmask  # noqa: E402
except ImportError:  # pragma: no cover - depends on the branch
    refmask = None

NBINS = 65536
PCTS = (1, 5, 25, 50, 75, 95, 99)
TISSUE, RESIN, OFFSAMPLE = 1, 2, 3          # small label map values
BASE_CLASSES = ("tissue", "resin", "resin_strict", "offsample")
NP_CLASSES = ("neuropil", "other_tissue")
# Categorical colours in fixed order (validated default palette) and text inks
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
STYLES = ("-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (8, 2)))
INK, MUTED, GRID = "#1a1a19", "#6b6a64", "#e4e3dd"


# --- classification ---------------------------------------------------------------------------------

def downsample(img, factor):
    """Area mean over factor x factor blocks (cropped to a whole number of blocks)."""
    h, w = (img.shape[0] // factor) * factor, (img.shape[1] // factor) * factor
    small = cv2.resize(np.ascontiguousarray(img[:h, :w]), (w // factor, h // factor), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32), (h, w)


def to8(small):
    lo, hi = np.percentile(small, [0.5, 99.5])
    return np.clip((small - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)


def strict_resin(resin, min_px=1500, rim=8, edge=16):
    """Resin areas reaching the tile border, at least min_px, eroded by rim, edge band removed."""
    from scipy import ndimage
    lab, n = ndimage.label(resin, structure=np.ones((3, 3)))
    if not n:
        return np.zeros_like(resin)
    border = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    border = border[border > 0]
    if not len(border):
        return np.zeros_like(resin)
    sizes = np.asarray(ndimage.sum(resin, lab, border))
    keep = np.isin(lab, border[sizes >= min_px])
    if rim:
        keep = ndimage.binary_erosion(keep, iterations=rim)
    if edge:
        keep[:edge] = keep[-edge:] = False
        keep[:, :edge] = keep[:, -edge:] = False
    return keep


def classify(small, min_resin_px=1500, rim=8, edge=16):
    """(label map, strict-resin mask) of a downsampled tile: TISSUE / RESIN / OFFSAMPLE per pixel."""
    img8 = to8(small)
    bg = background.mask(img8)
    dark = background._dark_at_border(img8.astype(np.float32), background.DARK_EDGE, background.MIN_DARK_PX)
    lab = np.full(small.shape, TISSUE, np.uint8)
    lab[bg & ~dark] = RESIN
    lab[bg & dark] = OFFSAMPLE
    return lab, strict_resin(lab == RESIN, min_resin_px, rim, edge)


def upscale(mask_small, factor, shape):
    m = np.repeat(np.repeat(mask_small, factor, 0), factor, 1)
    return m[:shape[0], :shape[1]]


def percentiles(hist, ps=PCTS):
    """Percentiles (values) of a histogram over 0..len-1; NaN when empty."""
    n = int(hist.sum())
    if not n:
        return [float("nan")] * len(ps)
    c = np.cumsum(hist)
    return [float(np.searchsorted(c, p / 100 * n)) for p in ps]


def tile_histograms(img, factor=16, neuropil_small=None, **kw):
    """{class: 65536-bin histogram} and {class: fraction of the tile} for one raw tile."""
    small, (h, w) = downsample(img, factor)
    lab, strict = classify(small, **kw)
    crop = img[:h, :w]
    masks = {"tissue": lab == TISSUE, "resin": lab == RESIN, "resin_strict": strict, "offsample": lab == OFFSAMPLE}
    if neuropil_small is not None:
        npl = cv2.resize(neuropil_small.astype(np.uint8), (lab.shape[1], lab.shape[0]),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
        masks["neuropil"] = (lab == TISSUE) & npl
        masks["other_tissue"] = (lab == TISSUE) & ~npl
    hists, fracs = {}, {}
    for k, m in masks.items():
        full = upscale(m, factor, crop.shape)
        hists[k] = np.bincount(crop[full].ravel(), minlength=NBINS)[:NBINS].astype(np.int64)
        fracs[k] = float(m.mean())
    return hists, fracs


# --- sampling and the neuropil mask --------------------------------------------------------------------

def sample_z(slices, n=None, zs=None):
    all_z = np.unique(slices["z"].to_numpy())
    if zs:
        return [z for z in zs if z in set(all_z.tolist())]
    k = np.linspace(0.05 * (len(all_z) - 1), 0.95 * (len(all_z) - 1), n).round().astype(int)
    return sorted(set(all_z[k].tolist()))


def neuropil_masks(cfg, z):
    """{tile: bool mask at thumbnail resolution} from the reference-tissue mask, or None."""
    if refmask is None:
        return None
    out_dir = Path(cfg["output_dir"])
    tdir = out_dir / "work" / "preview" / "thumbs"
    try:
        idx = pd.read_csv(tdir / "index.csv", dtype={"tile": str})
        st = pd.read_csv(out_dir / "work" / "preview" / "stats.csv", dtype={"tile": str},
                         usecols=["z", "tile", "p0_5", "p99_5"])
        rows = idx.merge(st, on=["z", "tile"])
        rows = rows[rows["z"] == z]
        tiles = {r.tile: (np.asarray(np.load(tdir / r.npy, mmap_mode="r")[int(r.i)]), r.p0_5, r.p99_5)
                 for r in rows.itertuples()}
        if not tiles:
            return None
        masks = refmask.slice_masks(tiles)[0]
        return {t: refmask.to_pixels(m, tiles[t][0].shape) for t, m in masks.items()}
    except Exception as e:  # the mask is optional: never fail the statistics because of it
        print(f"raw_stats: no neuropil mask at z {z}: {e}", file=sys.stderr)
        return None


# --- run / merge ---------------------------------------------------------------------------------------

def default_out(cfg):
    return Path(cfg["output_dir"]) / "qc" / "raw_stats"


def run(cfg, args):
    out = Path(args.out) if args.out else default_out(cfg)
    out.mkdir(parents=True, exist_ok=True)
    sl = load_slices(cfg)
    zs = sample_z(sl, args.slices, [int(v) for v in args.z.split(",")] if args.z else None)
    task, n_tasks = task_info(args)
    mine = zs[task::n_tasks]
    classes = BASE_CLASSES + (NP_CLASSES if refmask is not None else ())
    pooled = {k: np.zeros(NBINS, np.int64) for k in classes}
    rows = []
    kw = {"min_resin_px": args.min_resin_px, "rim": args.rim, "edge": args.edge}
    for z in mine:
        npm = neuropil_masks(cfg, z) if refmask is not None else None
        for r in sl[sl["z"] == z].itertuples():
            img = ImageJStack(Path(cfg["raw_dir"]) / r.file).read(int(r.index))
            hists, fracs = tile_histograms(img, args.factor, (npm or {}).get(r.tile), **kw)
            rec = {"z": int(z), "tile": r.tile, "segment": int(r.segment)}
            for k, h in hists.items():
                pooled.setdefault(k, np.zeros(NBINS, np.int64))
                pooled[k] += h
                rec[f"{k}_frac"] = fracs[k]
                rec.update({f"{k}_p{p}": v for p, v in zip(PCTS, percentiles(h))})
            rows.append(rec)
            print(f"raw_stats: z {z} tile {r.tile}: tissue {fracs['tissue']:.2f} resin {fracs['resin']:.2f}", flush=True)
    meta = {"z": zs, "factor": args.factor, **kw, "neuropil": any("neuropil_frac" in r for r in rows)}
    if n_tasks > 1:
        stem = out / f"part{task}of{n_tasks}"
        np.savez_compressed(f"{stem}.npz", **pooled)
        pd.DataFrame(rows).to_csv(f"{stem}.csv", index=False)
        (out / f"part{task}of{n_tasks}.json").write_text(json.dumps(meta))
        return
    write_results(out, pooled, pd.DataFrame(rows), meta)


def write_results(out, pooled, tiles, meta):
    np.savez_compressed(out / "hist.npz", **pooled)
    tiles.sort_values(["z", "tile"]).to_csv(out / "tiles.csv", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    plot_pooled(out / "pooled.png", pooled)
    plot_tiles(out / "tiles.png", {"": tiles})
    print(f"raw_stats: wrote {out}/hist.npz, tiles.csv, pooled.png, tiles.png ({len(tiles)} tile slices)")


def merge(cfg, args):
    out = Path(args.out) if args.out else default_out(cfg)
    parts = sorted(out.glob("part*of*.npz"))
    if not parts:
        sys.exit(f"raw_stats merge: no part files in {out}")
    n = {int(p.stem.split("of")[1]) for p in parts}
    if len(n) != 1 or len(parts) != n.pop():
        sys.exit(f"raw_stats merge: {len(parts)} part files in {out} don't make one complete run")
    pooled, tables, meta = {}, [], None
    for p in parts:
        with np.load(p) as d:
            for k in d.files:
                pooled[k] = pooled.get(k, np.zeros(NBINS, np.int64)) + d[k]
        tables.append(pd.read_csv(p.with_suffix(".csv"), dtype={"tile": str}))
        meta = json.loads(p.with_suffix(".json").read_text())
    write_results(out, pooled, pd.concat(tables, ignore_index=True), meta)


# --- plots ----------------------------------------------------------------------------------------------

def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK})
    return plt


def _finish(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _curve(ax, h, label, i, bin_w=64):
    h = np.asarray(h, float)
    if not h.sum():
        return
    hb = h[: len(h) // bin_w * bin_w].reshape(-1, bin_w).sum(1)
    x = np.arange(len(hb)) * bin_w + bin_w / 2
    ax.plot(x, hb / hb.sum() * 100 * (100 / bin_w), color=SERIES[i % 8], ls=STYLES[i % 8], lw=2, label=label)


def _xlim(ax, hists):
    lo, hi = [], []
    for h in hists:
        if np.asarray(h).sum():
            p = percentiles(np.asarray(h), (0.1, 99.9))
            lo.append(p[0]); hi.append(p[1])
    if lo:
        pad = 0.05 * (max(hi) - min(lo))
        ax.set_xlim(min(lo) - pad, max(hi) + pad)


def plot_pooled(path, pooled):
    plt = _style()
    fig, ax = plt.subplots(figsize=(9, 5))
    shown = [k for k in ("tissue", "neuropil", "other_tissue", "resin", "resin_strict", "offsample") if k in pooled]
    for i, k in enumerate(shown):
        _curve(ax, pooled[k], f"{k.replace('_', ' ')} (median {percentiles(pooled[k], (50,))[0]:,.0f})", i)
    _xlim(ax, [pooled[k] for k in shown])
    ax.set_xlabel("raw detector value (16-bit)")
    ax.set_ylabel("% of pixels per 100 counts")
    ax.set_title("Raw values by class, all sampled tiles pooled (nothing corrected)", loc="left")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=10)
    _finish(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_tiles(path, tables, resin="resin"):
    """Per tile: tissue p5-p95 line, tissue median dot, resin median ring (tiles with > 5 % resin)."""
    plt = _style()
    names = list(tables)
    fig, ax = plt.subplots(figsize=(max(6, 3.2 * len(names)), 6))
    ys = []
    for i, name in enumerate(names):
        t = tables[name].sort_values(["z", "tile"]).reset_index(drop=True)
        if t.empty:
            continue
        x = i + np.linspace(-0.38, 0.38, len(t))
        col = SERIES[i % 8]
        ax.vlines(x, t["tissue_p5"], t["tissue_p95"], color=col, lw=1.3, alpha=0.75)
        ax.scatter(x, t["tissue_p50"], s=16, color=col, zorder=3)
        r = t[f"{resin}_frac"] > 0.05
        ax.scatter(x[r.to_numpy()], t.loc[r, f"{resin}_p50"], s=34, facecolor="white", edgecolor=col, lw=1.4, zorder=4)
        ys += [t["tissue_p5"].min(), t["tissue_p95"].max(), t.loc[r, f"{resin}_p50"].max() if r.any() else np.nan]
        span = int((t["tissue_p95"] - t["tissue_p5"]).median())
        gap = (t.loc[r, f"{resin}_p50"] - t.loc[r, "tissue_p50"]).median() if r.any() else float("nan")
        ax.text(i, 0.02, f"within a tile:\ntissue p5-p95 {span:,}\nresin above tissue "
                + ("n/a" if np.isnan(gap) else f"{int(gap):,}"), transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(range(len(names)), names)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ys = [v for v in ys if np.isfinite(v)]
    if ys:
        pad = 0.25 * (max(ys) - min(ys) + 1)
        ax.set_ylim(min(ys) - pad, max(ys) + 0.1 * pad)
    ax.set_ylabel("raw detector value (16-bit)")
    ax.set_title(f"Each tile separately (resin: {resin.replace('_', ' ')})", loc="left")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color=MUTED, lw=1.3, label="tissue p5-p95"),
                       Line2D([0], [0], marker="o", color=MUTED, lw=0, ms=5, label="tissue median"),
                       Line2D([0], [0], marker="o", mfc="white", mec=MUTED, lw=0, ms=7, label="resin median")],
              frameon=False, fontsize=9.5, loc="upper right")
    _finish(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def compare(args):
    """Several datasets' runs together: pooled tissue and resin curves, the per-tile panel and, where
    every run has it, neuropil / other tissue / resin per dataset."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for item in args.runs:
        name, d = item.split("=", 1)
        with np.load(Path(d) / "hist.npz") as h:
            runs[name] = ({k: h[k] for k in h.files}, pd.read_csv(Path(d) / "tiles.csv", dtype={"tile": str}))
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=False)
    for ax, key, title in ((axes[0], "tissue", "Tissue, all tiles pooled"), (axes[1], args.resin, "Resin, all tiles pooled")):
        for i, (name, (h, _)) in enumerate(runs.items()):
            _curve(ax, h[key], f"{name} (median {percentiles(h[key], (50,))[0]:,.0f})", i)
        _xlim(ax, [h[key] for h, _ in runs.values()])
        ax.set_xlabel("raw detector value (16-bit)")
        ax.set_title(title, loc="left")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=10)
        _finish(ax)
    axes[0].set_ylabel("% of pixels per 100 counts")
    fig.tight_layout()
    fig.savefig(out / "compare_pooled.png", dpi=110)
    plt.close(fig)
    plot_tiles(out / "compare_tiles.png", {n: t for n, (_, t) in runs.items()}, resin=args.resin)
    if all("neuropil" in h for h, _ in runs.values()):
        fig, axes = plt.subplots(1, len(runs), figsize=(6.3 * len(runs), 5.5), sharey=True, squeeze=False)
        for ax, (name, (h, _)) in zip(axes[0], runs.items()):
            for i, (k, lab) in enumerate((("neuropil", "neuropil"), ("other_tissue", "other tissue"), (args.resin, "resin"))):
                _curve(ax, h[k], lab, i)
            _xlim(ax, [h["tissue"], h[args.resin]])
            ax.set_title(name, loc="left")
            ax.set_xlabel("raw detector value (16-bit)")
            _finish(ax)
        axes[0][0].set_ylabel("% of pixels per 100 counts")
        if axes[0][-1].get_legend_handles_labels()[0]:
            axes[0][-1].legend(frameon=False, fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "compare_by_tissue.png", dpi=110)
        plt.close(fig)
    print(f"raw_stats compare: wrote {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("run", "merge"):
        p = sub.add_parser(cmd)
        p.add_argument("--config", required=True)
        p.add_argument("--raw-dir")
        p.add_argument("--output-dir")
        p.add_argument("--out", help="output folder (default <output_dir>/qc/raw_stats)")
        if cmd == "run":
            p.add_argument("--slices", type=int, default=8, help="number of slices sampled (default 8)")
            p.add_argument("--z", help="comma-separated z instead of --slices")
            p.add_argument("--factor", type=int, default=16, help="downsampling for the classification (default 16)")
            p.add_argument("--min-resin-px", type=int, default=1500, help="strict resin: smallest area (downsampled px)")
            p.add_argument("--rim", type=int, default=8, help="strict resin: rim left out (downsampled px)")
            p.add_argument("--edge", type=int, default=16, help="strict resin: tile-edge band left out (downsampled px)")
            p.add_argument("--task-id", type=int)
            p.add_argument("--num-tasks", type=int)
    p = sub.add_parser("compare")
    p.add_argument("--out", required=True)
    p.add_argument("--resin", choices=("resin", "resin_strict"), default="resin")
    p.add_argument("runs", nargs="+", help="NAME=STATS_DIR (a run's output folder)")
    args = ap.parse_args(argv)
    if args.cmd == "compare":
        return compare(args)
    cfg = load_config(args.config, args.raw_dir, args.output_dir, step_defaults())
    return run(cfg, args) if args.cmd == "run" else merge(cfg, args)


if __name__ == "__main__":
    main()
