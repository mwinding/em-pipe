"""Choose a reference contrast standard: several volumes side by side at several p5-p95 widths.

intensity.window: reference puts every volume's neuropil at the same median grey and p5-p95 width. To
choose the width, this tool shows crops of several volumes (columns) under the reference levels at each
candidate width (rows), and reports how much is clipped to black or white at each:

  - per crop: the fraction of its pixels at 0 or 255;
  - per volume: the fraction of the tissue at 0 or 255 in the median sampled slice, from refstats' tissue
    quantiles (all sampled slices, the whole slice, every tissue type), so not just what a crop happens
    to show;
  - per crop, what a smaller width costs in the middle of the range: the pixel noise in greys (noise_sd)
    against the error of rounding to whole greys (1 / sqrt(12) = 0.29 grey), as the share of the noise
    variance that rounding adds. While the noise is several greys, rounding to fewer grey levels loses
    next to nothing; clipping loses what it clips.

Each volume's config must be a variant with intensity.window: reference and its own levels
(intensity.output), already computed at one width (its reference.contrast). Another width w is the same
levels stretched around the median grey m: g' = m + (g - m) w / contrast. That is exact for
brightness: tissue while the window's width changes slowly along z (the shift then scales with the
width), so nothing has to be recomputed.

Spec (JSON): {"widths": [130, 135, 140], "crop_um": 10, "seed": 0, "crop_kinds": ["inside", "border"],
              "items": [{"name": "35h", "config": "...yaml", "volume": "..._8nm.ome.zarr", "z": 8848,
                         "region": [x0, y0, x1, y1] or null}, ...]}

  python tools/standard_compare.py SPEC.json OUT.png [--json OUT.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference_preview import F, Dataset, LevelsMap, pick_crops, region_mask  # noqa: E402
from pipeline import intensity, omezarr, patchgrid, preview, refmask, refstats, shading  # noqa: E402
from pipeline.config import deep_merge, load_config, qc_path  # noqa: E402


def unclipped(lmap, g, z):
    """Greys of rendered values g under the variant's levels (LevelsMap's map, before clipping to 0-255)."""
    Lo, Ho = lmap.old.loc[z, ["Lc", "Hc"]]
    Ln, Hn = lmap.new.loc[z, ["Lc", "Hc"]]
    u = Lo + np.asarray(g, np.float64) * (Ho - Lo) / 255.0
    return 255.0 * (u - Ln) / (Hn - Ln)


ROUNDING_SD = 1 / np.sqrt(12)       # SD of the error from rounding to whole greys (uniform over one step)


def noise_sd(img, ok=None):
    """Pixel noise SD of an image in its own units (Immerkaer 1996): the image filtered with
    [[1,-2,1],[-2,4,-2],[1,-2,1]], which cancels smooth structure and scales white noise by 6, its
    median absolute value as a robust SD (so membranes and other edges, a minority, don't inflate it)
    divided by 6. ``ok``: pixels to use (their 3x3 neighbourhood whole); None: all. Noise correlated
    between neighbours (e.g. from interpolation) reads low, so for comparing with the rounding error
    the estimate errs on the cautious side."""
    from scipy.ndimage import binary_erosion
    from scipy.signal import convolve2d
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], float)
    r = convolve2d(np.asarray(img, np.float64), k, mode="valid")
    if ok is not None:
        r = r[binary_erosion(ok, np.ones((3, 3), bool))[1:-1, 1:-1]]
    return float(1.4826 * np.median(np.abs(r)) / 6.0) if r.size else float("nan")


def other_width(g, med, width, width_to):
    """Greys under reference levels made for p5-p95 width ``width``, as levels made for ``width_to`` would
    give them: a stretch around the median grey (unclipped in, clipped to 0-255 out)."""
    return np.clip(med + (np.asarray(g, np.float64) - med) * float(width_to) / float(width), 0, 255)


def volume_clipping(samples, med, width, widths):
    """Per candidate width: the fraction of tissue below 0 and above 255 in the median sampled slice
    (and the 95th percentile over slices of the total), from the reference QC's per-sample table
    (intensity_reference[_<output>]_samples.csv: the neuropil's median, rq_5 / rq_95, its grey and
    p5-p95 width under the levels, and the tissue quantiles tq_<percent>)."""
    qcols = sorted((c for c in samples.columns if c.startswith("tq_")), key=lambda c: float(c[3:]))
    ps = np.array([float(c[3:]) for c in qcols]) / 100
    s = samples.dropna(subset=["grey_new", "contrast_new", *qcols])
    span = 255.0 * (s["rq_95"] - s["rq_5"]).to_numpy(float) / s["contrast_new"].to_numpy(float)   # raw per 255 greys
    centre = s["median"].to_numpy(float) - (s["grey_new"].to_numpy(float) - med) * span / 255.0  # raw value at grey med
    tq = s[qcols].to_numpy(float)
    out = {}
    for w in widths:
        span_w = span * float(width) / float(w)
        lo, hi = centre - med * span_w / 255.0, centre + (255.0 - med) * span_w / 255.0
        black = np.array([np.interp(a, q, ps, left=0.0, right=1.0) for a, q in zip(lo, tq)])
        white = np.array([1.0 - np.interp(b, q, ps, left=0.0, right=1.0) for b, q in zip(hi, tq)])
        out[w] = {"black_median": float(np.median(black)), "white_median": float(np.median(white)),
                  "total_p95": float(np.percentile(black + white, 95)), "samples": int(len(s))}
    return out


def main(argv=None):
    import pandas as pd
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("spec")
    p.add_argument("out_png")
    p.add_argument("--json")
    a = p.parse_args(argv)
    spec = json.loads(Path(a.spec).read_text())
    widths = [float(w) for w in spec["widths"]]
    crop = int(round(float(spec.get("crop_um", 10)) / 0.008))
    rng = np.random.default_rng(int(spec.get("seed", 0)))
    kinds = spec.get("crop_kinds", ["inside", "border"])
    defaults = deep_merge(deep_merge(refmask.DEFAULTS, preview.DEFAULTS), {"refmask": {"method": "classifier"}})
    for d in (intensity.DEFAULTS, shading.DEFAULTS, patchgrid.DEFAULTS, refstats.DEFAULTS):
        defaults = deep_merge(defaults, d)
    cols = []
    for it in spec["items"]:
        z = int(it["z"])
        cfg = load_config(it["config"], step_defaults=defaults)
        r = intensity.reference_settings(cfg)
        med, width = float(r["median"]), float(r["contrast"])
        samples = pd.read_csv(qc_path(cfg, intensity._qc_name(cfg, "intensity_reference") + "_samples.csv"))
        vol = volume_clipping(samples, med, width, widths)
        lmap = LevelsMap(cfg)
        ds = Dataset(cfg, it["volume"])
        mask, bg, cov, _ = ds.volume_mask(z, F)
        reg = region_mask(mask.shape, it.get("region"), F)
        picks = dict(zip(("inside", "border"), pick_crops(mask & reg, cov & ~bg & reg, crop // F, rng)))
        arr = omezarr.open_scale(ds.root, 0)
        crops = {}
        for kind in kinds:
            if kind not in picks:
                continue
            y, x = picks[kind][0] * F, picks[kind][1] * F
            g = np.asarray(arr[ds.planes[z], y:y + crop, x:x + crop].read().result())
            new = unclipped(lmap, g, z)
            data = g > 0
            # noise at the levels' own width, from pixels the rendered volume didn't clip
            sd = noise_sd(new, (g > 0) & (g < 255))
            crops[kind] = {"y": y, "x": x, "noise_sd": sd, "rows": {}}
            for w in widths:
                gw = other_width(new, med, width, w)
                sd_w = sd * w / width
                crops[kind]["rows"][w] = {"img": gw[::2, ::2].astype(np.uint8),
                                          "black": float((gw[data] <= 0).mean()), "white": float((gw[data] >= 255).mean()),
                                          "noise_sd": sd_w, "rounding_share": float(ROUNDING_SD ** 2 / sd_w ** 2)}
        cols.append({"name": it["name"], "z": z, "median": med, "width": width, "volume": vol, "crops": crops})
        print(it["name"], z, {w: {k: round(100 * v, 2) for k, v in c.items() if k != "samples"} for w, c in vol.items()},
              flush=True)
        for kind, cr in crops.items():
            print(f"  {kind}: pixel noise SD (greys) " + ", ".join(
                f"width {w:g}: {rw['noise_sd']:.1f} (rounding adds {100 * rw['rounding_share']:.2f} % to the noise variance)"
                for w, rw in cr["rows"].items()), flush=True)
    plot(cols, widths, kinds, spec, a.out_png)
    if a.json:
        Path(a.json).write_text(json.dumps({"widths": widths, "items": [
            {"name": c["name"], "z": c["z"], "median": c["median"], "levels_width": c["width"],
             "volume_clipping": {str(w): v for w, v in c["volume"].items()},
             "crops": {k: {"y": v["y"], "x": v["x"], "rounding_sd": round(float(ROUNDING_SD), 4),
                           "per_width": {str(w): {"black": round(rw["black"], 5), "white": round(rw["white"], 5),
                                                  "noise_sd": round(rw["noise_sd"], 3),
                                                  "rounding_share": round(rw["rounding_share"], 6)}
                                         for w, rw in v["rows"].items()}}
                       for k, v in c["crops"].items()}} for c in cols]}, indent=1))
    return 0


def plot(cols, widths, kinds, spec, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = {"inside": "inside the neuropil", "border": "neuropil / cell-body border"}
    rows = [(k, w) for k in kinds for w in widths]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(4.3 * len(cols), 4.75 * len(rows)), squeeze=False,
                             layout="constrained")
    for i, (kind, w) in enumerate(rows):
        for j, c in enumerate(cols):
            ax = axes[i, j]
            ax.set_xticks([]), ax.set_yticks([])
            cr = c["crops"].get(kind)
            if cr is None:
                ax.set_axis_off()
                continue
            rw, vol = cr["rows"][w], c["volume"][w]
            ax.imshow(rw["img"], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_title(f"{c['name']} z {c['z']}, {names[kind]}\nwidth {w:g}: crop black {100 * rw['black']:.2f} %, "
                         f"white {100 * rw['white']:.2f} %\nwhole volume (median slice): black "
                         f"{100 * vol['black_median']:.2f} %, white {100 * vol['white_median']:.2f} %\n"
                         f"pixel noise {rw['noise_sd']:.1f} greys; rounding adds {100 * rw['rounding_share']:.2f} %",
                         fontsize=9, loc="left")
            if j == 0:
                ax.set_ylabel(f"p5-p95 width {w:g}", fontsize=11)
    med = cols[0]["median"] if cols else 150
    fig.suptitle(f"Neuropil median {med:g}, p5-p95 width {' / '.join(f'{w:g}' for w in widths)} (rows), "
                 f"volumes side by side ({spec.get('crop_um', 10)} um crops, shown at 16 nm)", fontsize=12)
    fig.savefig(out_png, dpi=90)


if __name__ == "__main__":
    sys.exit(main())
