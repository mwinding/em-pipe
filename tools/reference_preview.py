"""Preview what a reference standard (neuropil median and p5-p95 width) would do to rendered volumes.

A rendered volume is a linear map of the corrected data, and so is intensity.window: reference, so
the reference window can be previewed on the rendered greys without re-rendering. For each listed
(volume, z): the neuropil mask (pipeline/refmask, as tools/refmask_check.py draws it) is placed in the
volume's plane, 8 nm pixels are pooled from windows wholly inside it (optionally within a region),
and the neuropil's median and p5-p95 width measured. The preview maps every grey g to

    g' = median_std + (g - median) * width_std / width          (clipped to 0-255)

which puts that slice's neuropil at the standard (the slice's own numbers, no smoothing over z).
Two crops per slice are shown before and after: one inside the neuropil, one across the neuropil /
cell-body border. Pixels already at 0 or 255 in the rendered volume stay at the ends.

Spec (JSON): {"median": 150, "width": 150, "crop_um": 10, "windows": 24, "seed": 0,
              "items": [{"name": "35h", "config": "...yaml", "volume": "..._8nm.ome.zarr", "z": 8848,
                         "region": [x0, y0, x1, y1] or null}, ...]}

  python tools/reference_preview.py SPEC.json OUT.png [--json OUT.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from refmask_check import Dataset, place  # noqa: E402
from pipeline import omezarr, preview, refmask  # noqa: E402
from pipeline.config import deep_merge, load_config  # noqa: E402

F = 16          # mask grid: 8 nm px per cell of the plane grid (128 nm)
WIN = 512       # measuring windows, 8 nm px (4 um)
STRIDE = 2      # 8 nm px stride of the pooled values


def region_mask(shape, region, f):
    """Bool mask of the plane grid (downsample f) inside region [x0, y0, x1, y1] (None sides open)."""
    ok = np.ones(shape, bool)
    if region:
        x0, y0, x1, y1 = region
        if x0 is not None:
            ok[:, :max(int(np.ceil(x0 / f)), 0)] = False
        if x1 is not None:
            ok[:, int(x1 // f):] = False
        if y0 is not None:
            ok[:max(int(np.ceil(y0 / f)), 0), :] = False
        if y1 is not None:
            ok[int(y1 // f):, :] = False
    return ok


def window_fraction(a, size):
    """Mean of ``a`` over every size x size window (top-left indexed)."""
    S = np.pad(np.cumsum(np.cumsum(a.astype(np.float64), 0), 1), ((1, 0), (1, 0)))
    return (S[size:, size:] - S[:-size, size:] - S[size:, :-size] + S[:-size, :-size]) / (size * size)


def neuropil_numbers(arr, plane, ok, n, rng):
    """Pooled 8 nm greys of up to n windows inside ok: (median, p5, p95, windows used)."""
    s = WIN // F
    vals = []
    for y, x in place(ok, n, s, rng):
        w = np.asarray(arr[plane, y * F:y * F + WIN, x * F:x * F + WIN].read().result())[::STRIDE, ::STRIDE]
        vals.append(w[w > 0].ravel())      # 0: no data
    if not vals:
        return None
    v = np.concatenate(vals)
    p5, med, p95 = np.percentile(v, [5, 50, 95])
    return float(med), float(p5), float(p95), len(vals)


def pick_crops(neuropil, tissue, size, rng):
    """(y, x) top-left on the mask grid of a crop inside the neuropil and one across its border."""
    fn, ft = window_fraction(neuropil, size), window_fraction(tissue, size)
    out = []
    for cond in ((fn > 0.97), (fn > 0.35) & (fn < 0.65) & (ft > 0.97)):
        ys, xs = np.nonzero(cond)
        if len(ys):
            i = rng.integers(len(ys))
            out.append((int(ys[i]), int(xs[i])))
    return out


def standardise(g, med, width, med_std, width_std):
    return np.clip(med_std + (g.astype(np.float64) - med) * width_std / width, 0, 255)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("spec")
    p.add_argument("out_png")
    p.add_argument("--json")
    a = p.parse_args(argv)
    spec = json.loads(Path(a.spec).read_text())
    med_std, width_std = float(spec.get("median", 150)), float(spec.get("width", 150))
    crop = int(round(float(spec.get("crop_um", 10)) / 0.008))
    rng = np.random.default_rng(int(spec.get("seed", 0)))
    rows, results = [], []
    for it in spec["items"]:
        defaults = deep_merge(deep_merge(refmask.DEFAULTS, preview.DEFAULTS), {"refmask": {"method": "classifier"}})
        cfg = load_config(it["config"], step_defaults=defaults)
        ds = Dataset(cfg, it["volume"])
        z = int(it["z"])
        mask, bg, cov, _ = ds.volume_mask(z, F)
        ok = mask & region_mask(mask.shape, it.get("region"), F)
        tissue = cov & ~bg & region_mask(mask.shape, it.get("region"), F)
        arr = omezarr.open_scale(ds.root, 0)
        plane = ds.planes[z]
        num = neuropil_numbers(arr, plane, ok, int(spec.get("windows", 24)), rng)
        if num is None:
            print(f"{it['name']} z {z}: no neuropil windows", flush=True)
            continue
        med, p5, p95, nwin = num
        width = p95 - p5
        res = {"name": it["name"], "z": z, "windows": nwin, "median": round(med, 1), "p5": round(p5, 1),
               "p95": round(p95, 1), "width": round(width, 1), "scale": round(width_std / width, 3),
               "shift_at_median": round(med_std - med, 1), "crops": []}
        for kind, (y, x) in zip(("inside the neuropil", "neuropil / cell-body border"),
                                pick_crops(mask, tissue, crop // F, rng)):
            Y, X = y * F, x * F
            g = np.asarray(arr[plane, Y:Y + crop, X:X + crop].read().result())
            new = standardise(g, med, width, med_std, width_std)
            data = g > 0
            clip = {"dark_now": float((g[data] == 0).mean()) if data.any() else 0.0,
                    "dark_std": float((new[data] <= 0).mean()), "bright_now": float((g[data] >= 255).mean()),
                    "bright_std": float((new[data] >= 255).mean())}
            res["crops"].append({"kind": kind, "y": Y, "x": X, **{k: round(v, 4) for k, v in clip.items()}})
            rows.append((it["name"], z, kind, g[::2, ::2], new[::2, ::2].astype(np.uint8), res, clip))
        results.append(res)
        print(json.dumps(res), flush=True)
    plot(rows, med_std, width_std, spec, a.out_png)
    if a.json:
        Path(a.json).write_text(json.dumps({"standard": {"median": med_std, "width": width_std},
                                            "items": results}, indent=1))
    return 0


def plot(rows, med_std, width_std, spec, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(10, 5.2 * n), squeeze=False)
    for i, (name, z, kind, now, std, res, clip) in enumerate(rows):
        for j, (img, title) in enumerate(((now, f"now: neuropil median {res['median']:.0f}, p5-p95 width {res['width']:.0f}"),
                                          (std, f"standard {med_std:.0f} / {width_std:.0f}: "
                                                f"contrast x{res['scale']:.2f}"))):
            ax = axes[i, j]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            k = "now" if j == 0 else "std"
            ax.set_title(f"{name} z {z}, {kind}\n{title}\nblack {100 * clip['dark_' + k]:.2f} %, "
                         f"white {100 * clip['bright_' + k]:.2f} % of the crop", fontsize=9.5, loc="left")
    fig.suptitle(f"Neuropil standard: median {med_std:.0f}, p5-p95 width {width_std:.0f} "
                 f"({spec.get('crop_um', 10)} um crops, shown at 16 nm)", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_png, dpi=90)


if __name__ == "__main__":
    sys.exit(main())
