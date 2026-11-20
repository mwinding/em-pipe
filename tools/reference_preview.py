"""Preview what a reference standard (neuropil median and p5-p95 width) would do to rendered volumes.

A rendered volume is a linear map of the corrected data, and so is intensity.window: reference, so
the reference window can be previewed on the rendered greys without re-rendering. For each listed
(volume, z): the neuropil mask (pipeline/refmask, as tools/refmask_check.py draws it) is placed in the
volume's plane, and the neuropil's median and p5-p95 width are measured from 8 nm pixels pooled over
all of the slice's neuropil, eroded by one mask cell (optionally within a region): the same definition
refstats and intensity.window: reference use (spec "measure": "windows" pools windows wholly inside
the neuropil instead, which gives a narrower width). The preview maps every grey g to

    g' = median_std + (g - median) * width_std / width          (clipped to 0-255)

which puts that slice's neuropil at the standard (the slice's own numbers, no smoothing over z).
Two crops per slice are shown before and after: one inside the neuropil, one across the neuropil /
cell-body border. Pixels already at 0 or 255 in the rendered volume stay at the ends.

With spec "mode": "levels" the crops are instead mapped exactly from each volume's levels to the levels a
variant config wrote (intensity.output, e.g. intensity.window: reference): what a re-render would give.
"z" may then be a list, and "crop_kinds" (["inside", "border"]) chooses the crops.

Spec (JSON): {"median": 150, "width": 150, "crop_um": 10, "windows": 24, "seed": 0, "measure": "slice",
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


def neuropil_numbers_slice(arr, plane, ok, erode_cells=1, cell=16, stride=4):
    """As refstats measures (the definition intensity.window: reference uses): 8 nm greys pooled over all
    the slice's neuropil, eroded by ``erode_cells`` mask cells (``cell`` px of the F grid), every
    ``stride``-th px: (median, p5, p95, pixels used)."""
    from scipy import ndimage
    if erode_cells:
        ok = ndimage.distance_transform_edt(ok) > erode_cells * cell
    img = np.asarray(arr[plane].read().result())[::stride, ::stride]
    iy = np.minimum(np.arange(img.shape[0]) * stride // F, ok.shape[0] - 1)
    ix = np.minimum(np.arange(img.shape[1]) * stride // F, ok.shape[1] - 1)
    sel = ok[np.ix_(iy, ix)] & (img > 0)
    v = img[sel]
    if not v.size:
        return None
    p5, med, p95 = np.percentile(v, [5, 50, 95])
    return float(med), float(p5), float(p95), int(v.size)


def neuropil_numbers(arr, plane, ok, n, rng):
    """Pooled 8 nm greys of up to n windows inside ok: (median, p5, p95, windows used). Narrower than the
    slice-wide measure (windows deep inside the neuropil leave out its edges and between-area spread)."""
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


def _segments(cfg):
    """z -> segment (work/check/slices.csv)."""
    import pandas as pd
    from pipeline.config import step_path
    sl = pd.read_csv(step_path(cfg, "check", "slices.csv"), usecols=["z", "segment"])
    return sl.drop_duplicates("z").set_index("z")["segment"]


class LevelsMap:
    """The exact change from a volume's levels to a variant's levels (e.g. intensity.window: reference with
    intensity.output): per z, both common-scale windows, so a rendered grey g maps to the grey render would
    write with the new levels: 255 (Lo + g (Ho - Lo) / 255 - Ln) / (Hn - Ln)."""

    def __init__(self, cfg):
        import pandas as pd
        from pipeline import intensity
        bal = intensity.balanced_scale(cfg)[["z", "tile", "g", "o"]]
        def window(path):
            lv = pd.read_csv(path, dtype={"tile": str})[["z", "tile", "lo", "hi"]].merge(bal, on=["z", "tile"])
            return intensity.common_window(lv)
        self.old_path, self.new_path = intensity.levels_path(cfg, output=""), intensity.levels_path(cfg)
        if self.old_path == self.new_path:
            raise ValueError("the config has no intensity.output: nothing to compare its levels with")
        self.old, self.new = window(self.old_path), window(self.new_path)
        self.segment = _segments(cfg)
        self.ref = (cfg.get("intensity") or {}).get("reference")

    def __call__(self, g, z):
        Lo, Ho = self.old.loc[z, ["Lc", "Hc"]]
        Ln, Hn = self.new.loc[z, ["Lc", "Hc"]]
        u = Lo + g.astype(np.float64) * (Ho - Lo) / 255.0
        return np.clip(255.0 * (u - Ln) / (Hn - Ln), 0, 255), float((Ho - Lo) / (Hn - Ln))


def standardise(g, med, width, med_std, width_std):
    return np.clip(med_std + (g.astype(np.float64) - med) * width_std / width, 0, 255)


_maps = {}


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
    mode = spec.get("mode", "measure")      # measure: the slice's own numbers | levels: a variant's levels
    kinds = spec.get("crop_kinds", ["inside", "border"])
    for it, z in ((it, int(z)) for it in spec["items"] for z in (it["z"] if isinstance(it["z"], list) else [it["z"]])):
        defaults = deep_merge(deep_merge(refmask.DEFAULTS, preview.DEFAULTS), {"refmask": {"method": "classifier"}})
        if mode == "levels":
            from pipeline import intensity, patchgrid, refstats, shading
            for d in (intensity.DEFAULTS, shading.DEFAULTS, patchgrid.DEFAULTS, refstats.DEFAULTS):
                defaults = deep_merge(defaults, d)
        cfg = load_config(it["config"], step_defaults=defaults)
        ds = Dataset(cfg, it["volume"])
        lmap = None
        if mode == "levels":
            if it["config"] not in _maps:
                _maps[it["config"]] = LevelsMap(cfg)
            lmap = _maps[it["config"]]
        mask, bg, cov, _ = ds.volume_mask(z, F)
        ok = mask & region_mask(mask.shape, it.get("region"), F)
        tissue = cov & ~bg & region_mask(mask.shape, it.get("region"), F)
        arr = omezarr.open_scale(ds.root, 0)
        plane = ds.planes[z]
        if mode == "levels":
            seg = int(lmap.segment.get(z, -1))
            res = {"name": it["name"], "z": z, "mode": "levels", "segment": seg, "crops": []}
        else:
            if spec.get("measure", "slice") == "slice":
                num = neuropil_numbers_slice(arr, plane, ok)
            else:
                num = neuropil_numbers(arr, plane, ok, int(spec.get("windows", 24)), rng)
            if num is None:
                print(f"{it['name']} z {z}: no neuropil windows", flush=True)
                continue
            med, p5, p95, nwin = num
            width = p95 - p5
            res = {"name": it["name"], "z": z, "measure": spec.get("measure", "slice"), "n": nwin, "median": round(med, 1),
                   "p5": round(p5, 1), "p95": round(p95, 1), "width": round(width, 1), "scale": round(width_std / width, 3),
                   "shift_at_median": round(med_std - med, 1), "crops": []}
        picks = dict(zip(("inside", "border"), pick_crops(mask, tissue, crop // F, rng)))
        names = {"inside": "inside the neuropil", "border": "neuropil / cell-body border"}
        for kind, (y, x) in ((names[k], picks[k]) for k in kinds if k in picks):
            Y, X = y * F, x * F
            g = np.asarray(arr[plane, Y:Y + crop, X:X + crop].read().result())
            if mode == "levels":
                new, factor = lmap(g, z)
                res["scale"] = round(factor, 3)
            else:
                new = standardise(g, med, width, med_std, width_std)
            data = g > 0
            clip = {"dark_now": float((g[data] == 0).mean()) if data.any() else 0.0,
                    "dark_std": float((new[data] <= 0).mean()), "bright_now": float((g[data] >= 255).mean()),
                    "bright_std": float((new[data] >= 255).mean())}
            res["crops"].append({"kind": kind, "y": Y, "x": X, **{k: round(v, 4) for k, v in clip.items()}})
            rows.append((it["name"], z, kind, g[::2, ::2], new[::2, ::2].astype(np.uint8), res, clip))
        results.append(res)
        print(json.dumps(res), flush=True)
    per = None
    if mode == "levels" and _maps:        # the standard the variants' levels were made with, for the title
        ref = (next(iter(_maps.values())).ref or {})
        med_std, width_std = float(ref.get("median", med_std)), float(ref.get("contrast", width_std))
        per = ref.get("per", "slice")
    plot(rows, med_std, width_std, spec, a.out_png, per)
    if a.json:
        Path(a.json).write_text(json.dumps({"standard": {"median": med_std, "width": width_std},
                                            "items": results}, indent=1))
    return 0


def plot(rows, med_std, width_std, spec, out_png, per=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(10, 5.2 * n), squeeze=False, layout="constrained")
    for i, (name, z, kind, now, std, res, clip) in enumerate(rows):
        if res.get("mode") == "levels":
            seg = f", segment {res['segment']}" if res.get("segment", -1) >= 0 else ""
            titles = (f"now (current levels){seg}", f"new levels{seg}: contrast x{res['scale']:.2f}")
        else:
            titles = (f"now: neuropil median {res['median']:.0f}, p5-p95 width {res['width']:.0f}",
                      f"standard {med_std:.0f} / {width_std:.0f}: contrast x{res['scale']:.2f}")
        for j, (img, title) in enumerate(((now, titles[0]), (std, titles[1]))):
            ax = axes[i, j]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            k = "now" if j == 0 else "std"
            ax.set_title(f"{name} z {z}, {kind}\n{title}\nblack {100 * clip['dark_' + k]:.2f} %, "
                         f"white {100 * clip['bright_' + k]:.2f} % of the crop", fontsize=9.5, loc="left")
    if spec.get("mode") == "levels":
        how = ("one contrast and brightness shift per segment" if per == "segment"
               else "contrast and brightness shift following the neuropil slowly")
        head = (f"Before and after: reference contrast, neuropil median {med_std:.0f}, p5-p95 width {width_std:.0f} "
                f"({spec.get('crop_um', 10)} um crops, shown at 16 nm)\n{how}, brightness following every slice's tissue")
    else:
        head = (f"Neuropil standard: median {med_std:.0f}, p5-p95 width {width_std:.0f} "
                f"({spec.get('crop_um', 10)} um crops, shown at 16 nm)")
    fig.suptitle(head, fontsize=11, x=0.01, ha="left")
    fig.savefig(out_png, dpi=90)


if __name__ == "__main__":
    sys.exit(main())
