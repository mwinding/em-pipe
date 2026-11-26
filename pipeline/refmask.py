"""Reference tissue mask: where one kind of tissue (by default neuropil) is, as a coarse yes/no per
cell of ``cell_px`` thumbnail pixels (16 px of a 128 nm preview thumbnail: ~2 um squares).

It only chooses where intensity statistics are measured, so that volumes stained or imaged
differently can be put on one reference (e.g. "neuropil median grey 150"); it is never applied to the
image. It runs on the preview thumbnails (raw values, before any contrast step), so every feature is
a ratio of texture energies, unchanged by the slice's brightness and contrast.

Neuropil (with fibre tracts and commissures) at ~128 nm is a fine, even texture: neurites and their
membranes, well under a micrometre. The cell body rind is coarser: nuclei and cell bodies of a few
micrometres, smooth inside with outlines between them, as are other non-neuropil tissues (gut, glia
sheaths, trachea). Per thumbnail pixel, three band energies of the raw values: fine (the image minus
its Gaussian blur at 1 px), mid (blur 1 px minus blur 4 px) and coarse (blur 4 px minus blur 16 px);
per cell their rms, and the score

    score = (log2(fine / coarse) + log2(mid / coarse)) / 2

averaged over neighbouring tissue cells (Gaussian, ``smooth_cells``). Neuropil scores high (~1 on the
three volumes tried), rind low (~0). Cells mostly background (pipeline/background.py: resin,
off-sample) have no score. A slice's candidate cells score above one threshold for all its tiles:
Otsu's threshold of the slice's tissue scores, kept within ``threshold_range`` (a slice with little
neuropil has no two clear classes). Of these, only connected areas of at least ``min_cells`` cells
holding a seed are kept: a cell whose score averaged over a wider neighbourhood (``seed_smooth_cells``,
~4 um) is over ``seed_score``. Patches of rind with neuropil-like texture are mostly small and
middling, so they rarely hold a seed. Holes of up to ``max_hole_cells`` are then filled (dark granule
clusters and large neurites inside neuropil).

Every filter is isotropic (Gaussian blurs, square cells), so the score doesn't depend on how
structures run in the section plane (volumes cut in different orientations, tracts cut across or
along). Known weakness: densely packed small cell bodies (~2-3 um, little nucleus contrast) have a
neuropil-like spectrum at this resolution and can be included.
"""

import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

from . import background, boost

log = logging.getLogger(__name__)

DEFAULTS = {
    "refmask": {
        "enabled": True,               # run_pipeline.sh: the refmask step after preview (for intensity.window: reference)
        "tissue": "neuropil",          # the reference tissue (only neuropil so far)
        "method": "classifier",        # classifier: trained trees on texture features | score: the texture score alone
        "model": None,                 # classifier model JSON (None: the one shipped, pipeline/refmask_<tissue>.json)
        "prob_threshold": 0.5,         # classifier: cells above this probability are candidates
        "seed_prob": 0.8,              # classifier: a kept area holds a cell whose wider-average probability is over this
        "cell_px": 16,                 # cell size in thumbnail px (16 at 128 nm: ~2 um)
        "smooth_cells": 1.0,           # Gaussian sigma (cells) of the score averaging
        "threshold_range": [0.4, 0.7],  # the per-slice Otsu threshold is kept within this range
        "seed_smooth_cells": 2.0,      # Gaussian sigma (cells) of the wider average for seeds
        "seed_score": 0.9,             # a kept area must hold a cell whose wider average is over this
        "min_cells": 12,               # smallest neuropil area kept (cells)
        "max_hole_cells": 4,           # holes up to this size (cells) are filled
        "max_background": 0.5,         # cells with more background than this have no score
    },
}
TISSUES = ("neuropil",)
SIGMAS = (1.0, 4.0, 16.0)   # thumbnail px: fine | mid | coarse band edges
EPS = 1e-6


def _cells(a, c):
    """Block means of ``a`` over c x c cells (edge cells padded by reflection)."""
    ph, pw = -a.shape[0] % c, -a.shape[1] % c
    if ph or pw:
        a = np.pad(a, ((0, ph), (0, pw)), mode="reflect")
    h, w = a.shape
    return a.reshape(h // c, c, w // c, c).mean(axis=(1, 3))


def _smooth(v, sigma):
    """Gaussian average of the finite values of ``v`` (NaN stays NaN)."""
    ok = np.isfinite(v)
    w = ndimage.gaussian_filter(ok.astype(np.float64), sigma)
    a = ndimage.gaussian_filter(np.where(ok, v, 0.0), sigma) / np.maximum(w, EPS)
    return np.where(ok, a, np.nan)


def tile_scores(values, bg, cell_px=16, smooth_cells=1.0, max_background=0.5):
    """Per-cell score of one tile slice (NaN where background) and its tissue cells.

    values: the tile's raw values at preview resolution (e.g. preview.thumb_values of its thumbnail);
    bg: its background mask (pipeline.background.mask of the thumbnail)."""
    x = np.asarray(values, np.float64)
    g = [ndimage.gaussian_filter(x, s) for s in SIGMAS]
    fine = np.sqrt(_cells((x - g[0]) ** 2, cell_px))
    mid = np.sqrt(_cells((g[0] - g[1]) ** 2, cell_px))
    coarse = np.sqrt(_cells((g[1] - g[2]) ** 2, cell_px)) + EPS
    raw = 0.5 * (np.log2((fine + EPS) / coarse) + np.log2((mid + EPS) / coarse))
    tissue = _cells(np.asarray(bg, np.float64), cell_px) <= max_background
    score = np.where(tissue, raw, np.nan)
    return (_smooth(score, smooth_cells) if smooth_cells else score), tissue


BANDS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)   # Gaussian sigmas (thumbnail px): band k = blur[k-1] - blur[k]
CONTEXT = (2.0, 4.0)                       # sigmas (cells) of the neighbourhood averages: ~4 and ~8 um


def _hessian_eigs(x, sigma):
    hxx = ndimage.gaussian_filter(x, sigma, order=(0, 2))
    hyy = ndimage.gaussian_filter(x, sigma, order=(2, 0))
    hxy = ndimage.gaussian_filter(x, sigma, order=(1, 1))
    tr, d = hxx + hyy, np.sqrt((hxx - hyy) ** 2 + 4 * hxy ** 2)
    return 0.5 * (tr + d), 0.5 * (tr - d)


def cell_features(values, bg, cell_px=16, max_background=0.5):
    """Per-cell features of one tile slice for the classifier: ({name: (h, w) map}, tissue cells).

    All are unchanged by a linear change of the values (gain and offset) and by rotation: relative
    band energies (log2 rms of 6 difference-of-Gaussian bands minus their mean), the texture score,
    texture and brightness relative to the tile's tissue (its IQR and median), dark/light pixel
    fractions (below the tissue p5 / above p95), structure-tensor coherence, Hessian roundness at
    ~0.4 and ~1 um (round blobs 1, ridges 0) and the sign of blobs at ~1 um; each also averaged over
    ~4 and ~8 um of tissue around the cell, plus the spread of the score and the tissue fraction around it."""
    x = np.asarray(values, np.float64)
    bg = np.asarray(bg, bool)
    v = x[~bg] if (~bg).sum() >= 1000 else x.ravel()
    q5, q25, q50, q75, q95 = np.percentile(v, [5, 25, 50, 75, 95])
    iqr = max(q75 - q25, EPS)
    x = (x - q50) / iqr   # derivative kernels don't sum exactly to 0: keep a large offset out of them
    q5, q50, q95 = (q5 - q50) / iqr, 0.0, (q95 - q50) / iqr
    iqr = 1.0
    c = lambda a: _cells(a, cell_px)   # noqa: E731
    g = [x] + [ndimage.gaussian_filter(x, s) for s in BANDS]
    e = np.stack([np.log2(np.sqrt(c((g[k] - g[k + 1]) ** 2)) + EPS) for k in range(len(BANDS))])
    f = {f"band{k}": e[k] - e.mean(0) for k in range(len(BANDS))}
    b1, b4, b16 = (g[1 + BANDS.index(s)] for s in (1.0, 4.0, 16.0))
    fine, mid, coarse = (np.sqrt(c(a ** 2)) + EPS for a in (x - b1, b1 - b4, b4 - b16))
    f["score"] = 0.5 * (np.log2(fine / coarse) + np.log2(mid / coarse))
    f["texture"] = np.log2(np.sqrt(c((x - g[1 + BANDS.index(2.0)]) ** 2)) / iqr + EPS)
    f["bright"] = (c(x) - q50) / iqr
    f["dark"] = c((x < q5).astype(np.float64))
    f["light"] = c((x > q95).astype(np.float64))
    gy, gx = np.gradient(b1)
    jxx, jyy, jxy = (ndimage.gaussian_filter(a, 2.0) for a in (gx * gx, gy * gy, gx * gy))
    f["coherence"] = c(np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)) / (c(jxx + jyy) + EPS)
    for name, sigma in (("round_0.4um", 3.0), ("round_1um", 8.0)):
        l1, l2 = _hessian_eigs(x, sigma)
        f[name] = c(2 * np.abs(l1 * l2)) / (c(l1 ** 2 + l2 ** 2) + EPS)
    log = 64.0 * (ndimage.gaussian_filter(x, 8.0, order=(0, 2)) + ndimage.gaussian_filter(x, 8.0, order=(2, 0)))
    f["blob_sign_1um"] = c(log) / (np.sqrt(c(log ** 2)) + EPS)
    tissue = c(bg.astype(np.float64)) <= max_background
    base = list(f)
    for sigma in CONTEXT:
        for k in base:
            f[f"{k}_ctx{sigma:g}"] = _smooth(np.where(tissue, f[k], np.nan), sigma)
    sm = _smooth(np.where(tissue, f["score"], np.nan), CONTEXT[0])
    f["score_spread"] = np.sqrt(np.maximum(_smooth(np.where(tissue, (f["score"] - sm) ** 2, np.nan), CONTEXT[0]), 0))
    f["tissue_frac"] = ndimage.gaussian_filter(tissue.astype(np.float64), CONTEXT[1])
    return f, tissue


def load_model(settings):
    path = settings.get("model") or Path(__file__).with_name(f"refmask_{settings['tissue']}.json")
    return json.loads(Path(path).read_text())


def tile_probability(model, values, bg, cell_px=16, max_background=0.5):
    """Per-cell classifier probability of one tile slice (NaN where background) and its tissue cells."""
    f, tissue = cell_features(values, bg, cell_px, max_background)
    X = np.stack([f[n][tissue] for n in model["names"]], axis=1)
    p = np.full(tissue.shape, np.nan)
    if len(X):
        p[tissue] = boost.predict(model, np.nan_to_num(X))
    return p, tissue


def otsu(v, bins=256):
    """Otsu's threshold of the values ``v``."""
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    if v.size < 2 or v.min() == v.max():
        return float(v[0]) if v.size else float("nan")
    h, e = np.histogram(v, bins)
    c, p = 0.5 * (e[1:] + e[:-1]), h / h.sum()
    w0, m = np.cumsum(p), np.cumsum(p * c)
    between = (m[-1] * w0 - m) ** 2 / np.maximum(w0 * (1 - w0), 1e-12)
    return float(c[np.argmax(between)])


def clean(mask, seeds=None, tissue=None, min_cells=12, max_hole_cells=4):
    """Keep components of at least min_cells holding a seed (any cell when seeds is None); fill holes
    of tissue cells up to max_hole_cells (4-connected)."""
    mask = np.asarray(mask, bool)
    tissue = np.ones_like(mask) if tissue is None else np.asarray(tissue, bool)
    lab, n = ndimage.label(mask)
    if n:
        keep = np.bincount(lab.ravel(), minlength=n + 1) >= min_cells
        if seeds is not None:
            seeded = np.zeros(n + 1, bool)
            seeded[lab[np.asarray(seeds, bool) & mask]] = True
            keep &= seeded
        keep[0] = False
        mask = keep[lab]
    if max_hole_cells:
        lab, n = ndimage.label(~mask)
        if n:
            sizes = np.bincount(lab.ravel())
            edge = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
            small = sizes <= max_hole_cells
            small[0] = False
            small[edge] = False   # open to the tile border: not a hole
            mask = mask | (small[lab] & tissue)
    return mask


def slice_masks(tiles, settings=None):
    """Reference tissue masks of one slice.

    tiles: {tile: (thumb uint8, lo, hi)} with lo/hi the values the thumbnail was scaled from
    (preview stats p0_5 / p99_5). Returns ({tile: bool cell mask}, {tile: cell probabilities (classifier) or
    scores}, threshold)."""
    s = {**DEFAULTS["refmask"], **(settings or {})}
    if s["tissue"] not in TISSUES:
        raise ValueError(f"refmask.tissue must be one of {TISSUES}, got {s['tissue']!r}")
    from .preview import thumb_values   # preview imports matplotlib; keep refmask light
    scores, tissue = {}, {}
    if s["method"] == "classifier":
        model = load_model(s)
        for tile, (thumb, lo, hi) in tiles.items():
            scores[tile], tissue[tile] = tile_probability(model, thumb_values(thumb, lo, hi), background.mask(thumb),
                                                          s["cell_px"], s["max_background"])
        thr = float(s["prob_threshold"])
        masks = {t: clean(np.nan_to_num(v, nan=0.0) > thr,
                          np.nan_to_num(_smooth(v, s["seed_smooth_cells"]), nan=0.0) > s["seed_prob"],
                          tissue[t], s["min_cells"], s["max_hole_cells"]) for t, v in scores.items()}
        return masks, scores, thr
    if s["method"] != "score":
        raise ValueError(f"refmask.method must be classifier or score, got {s['method']!r}")
    for tile, (thumb, lo, hi) in tiles.items():
        scores[tile], tissue[tile] = tile_scores(thumb_values(thumb, lo, hi), background.mask(thumb), s["cell_px"],
                                      s["smooth_cells"], s["max_background"])
    allv = np.concatenate([v[np.isfinite(v)] for v in scores.values()]) if scores else np.zeros(0)
    lo_t, hi_t = s["threshold_range"]
    thr = float(np.clip(otsu(allv), lo_t, hi_t)) if allv.size else float(hi_t)
    masks = {t: clean(np.nan_to_num(v, nan=-np.inf) > thr,
                      np.nan_to_num(_smooth(v, s["seed_smooth_cells"]), nan=-np.inf) > s["seed_score"],
                      tissue[t], s["min_cells"], s["max_hole_cells"]) for t, v in scores.items()}
    return masks, scores, thr


def to_pixels(cell_mask, shape, cell_px=16):
    """A cell mask at thumbnail resolution (``shape``)."""
    m = np.repeat(np.repeat(np.asarray(cell_mask, bool), cell_px, 0), cell_px, 1)
    return m[:shape[0], :shape[1]]


# ----- pipeline step -------------------------------------------------------------------------------
# refmask run (array over preview's z-chunks): every selected (z, tile) classified from its preview
# thumbnail; work/refmask/masks/z<chunk>_tile<t>.npy, uint8 [n, cells y, cells x]: CELL_BACKGROUND
# (no score), CELL_TISSUE, CELL_REFERENCE; work/refmask/chunks/chunk_<chunk>.csv (z, tile, npy, i,
# threshold, tissue_cells, reference_cells, digest) marks a chunk done. refmask merge: index.csv and
# qc/refmask.png (reference tissue's share of the tissue per slice).

CELL_BACKGROUND, CELL_TISSUE, CELL_REFERENCE = 0, 1, 2


def settings_of(cfg):
    return {**DEFAULTS["refmask"], **(cfg.get("refmask") or {})}


def digest(s):
    """Identifies what the masks were made with: the settings and the model file's bytes."""
    h = hashlib.sha256(json.dumps({k: v for k, v in s.items() if k != "enabled"}, sort_keys=True).encode())
    if s["method"] == "classifier":
        path = s.get("model") or Path(__file__).with_name(f"refmask_{s['tissue']}.json")
        h.update(Path(path).read_bytes())
    return h.hexdigest()[:16]


def cell_codes(mask, score):
    out = np.full(mask.shape, CELL_TISSUE, np.uint8)
    out[~np.isfinite(score)] = CELL_BACKGROUND
    out[mask] = CELL_REFERENCE
    return out


def _thumbs(cfg):
    import pandas as pd
    from .config import step_path
    idx = pd.read_csv(step_path(cfg, "preview", "thumbs", "index.csv"), dtype={"tile": str})
    st = pd.read_csv(step_path(cfg, "preview", "stats.csv"), dtype={"tile": str}, usecols=["z", "tile", "p0_5", "p99_5"])
    return idx.merge(st, on=["z", "tile"])


def run(cfg, args):
    import pandas as pd
    from .cli import atomic_write, my_chunks, task_info
    from .config import step_dir, step_path
    from .preview import chunk_name, z_chunks
    from .slices import load_slices
    s = settings_of(cfg)
    dg = digest(s)
    if int(cfg["preview"]["factor"]) != 16:
        log.warning("preview.factor is %s: the shipped model was trained on 16x thumbnails (128 nm at 8 nm)",
                    cfg["preview"]["factor"])
    slices = load_slices(cfg)
    thumbs = _thumbs(cfg)
    thumbs_dir = step_path(cfg, "preview", "thumbs")
    mine = my_chunks(z_chunks(slices, cfg["preview"]["chunk_slices"]), *task_info(args))
    log.info("%d chunks for this task", len(mine))
    for zs in mine:
        name = chunk_name(zs)
        out = step_dir(cfg, "refmask", "chunks", f"chunk_{name}.csv")
        want = slices[slices["z"].isin(zs)][["z", "tile"]]
        if out.exists() and not args.overwrite:
            done = pd.read_csv(out, dtype={"tile": str})
            if (done["digest"] == dg).all() and pd.MultiIndex.from_frame(want).isin(
                    pd.MultiIndex.from_frame(done[["z", "tile"]])).all():
                log.info("chunk %s exists, skipping", name)
                continue
        out.unlink(missing_ok=True)
        rows, stacks = [], {}
        for z in zs:
            tz = thumbs[thumbs["z"] == z]
            tiles = {r.tile: (np.asarray(np.load(thumbs_dir / r.npy, mmap_mode="r")[int(r.i)]), r.p0_5, r.p99_5)
                     for r in tz.itertuples()}
            if not tiles:
                continue
            masks, scores, thr = slice_masks(tiles, s)
            for t in sorted(tiles):
                code = cell_codes(masks[t], scores[t])
                stacks.setdefault(t, []).append(code)
                rows.append({"z": z, "tile": t, "npy": f"z{name}_tile{t}.npy", "i": len(stacks[t]) - 1,
                             "threshold": round(float(thr), 4), "tissue_cells": int((code > 0).sum()),
                             "reference_cells": int((code == CELL_REFERENCE).sum()), "digest": dg})
        for t, st in stacks.items():
            data = np.stack(st)
            atomic_write(step_dir(cfg, "refmask", "masks", f"z{name}_tile{t}.npy"), lambda f, d=data: np.save(f, d))
        atomic_write(out, lambda f: pd.DataFrame(rows).to_csv(f, index=False))
        log.info("chunk %s: %d tile slices", name, len(rows))
    return 0


def merge(cfg, args):
    import pandas as pd
    from .cli import atomic_write
    from .config import qc_path, step_dir
    from .preview import chunk_name, save, z_chunks
    from .slices import load_slices
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = settings_of(cfg)
    dg = digest(s)
    slices = load_slices(cfg)
    parts, missing = [], []
    for zs in z_chunks(slices, cfg["preview"]["chunk_slices"]):
        path = step_dir(cfg, "refmask", "chunks", f"chunk_{chunk_name(zs)}.csv")
        if not path.exists():
            missing.append(path.name)
            continue
        d = pd.read_csv(path, dtype={"tile": str})
        if not (d["digest"] == dg).all():
            missing.append(f"{path.name} (other settings or model)")
            continue
        parts.append(d[d["z"].isin(zs)])
    if missing:
        raise RuntimeError(f"{len(missing)} refmask chunks missing or out of date, e.g. {missing[:3]}: run 'refmask run'")
    idx = pd.concat(parts, ignore_index=True)
    atomic_write(step_dir(cfg, "refmask", "index.csv"), lambda f: idx.to_csv(f, index=False))
    per_z = idx.groupby("z")[["tissue_cells", "reference_cells"]].sum()
    fig, ax = plt.subplots(figsize=(12, 3.5), layout="constrained")
    ax.plot(per_z.index, 100 * per_z["reference_cells"] / per_z["tissue_cells"].clip(lower=1), lw=0.8, color="#2a78d6")
    ax.set_xlabel("z")
    ax.set_ylabel(f"{s['tissue']} (% of tissue cells)")
    ax.set_title(f"Reference tissue mask ({s['method']}): share of the tissue per slice")
    save(fig, qc_path(cfg, "refmask.png"))
    log.info("refmask index: %d (z, tile) over %d slices", len(idx), len(per_z))
    return 0


class Cells:
    """Cell codes of (z, tile) from work/refmask (index.csv and the mask stacks)."""

    def __init__(self, cfg):
        import pandas as pd
        from .config import step_path
        path = step_path(cfg, "refmask", "index.csv")
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run the refmask step (run, merge)")
        self.dir = step_path(cfg, "refmask", "masks")
        self.index = pd.read_csv(path, dtype={"tile": str}).set_index(["z", "tile"])
        self.cell_px = int(settings_of(cfg)["cell_px"])
        self._open = {}

    def get(self, z, tile):
        if (z, tile) not in self.index.index:
            return None
        r = self.index.loc[(z, tile)]
        if r["npy"] not in self._open:
            self._open[r["npy"]] = np.load(self.dir / r["npy"], mmap_mode="r")
        return np.asarray(self._open[r["npy"]][int(r["i"])])


def main(argv=None):
    from .cli import base_parser, setup
    from .config import deep_merge
    from .preview import DEFAULTS as PREVIEW
    p = base_parser("Reference tissue mask of every selected (z, tile) from the preview thumbnails")
    p.add_argument("command", choices=["run", "merge"])
    args = p.parse_args(argv)
    cfg = setup(args, deep_merge(DEFAULTS, PREVIEW))
    return run(cfg, args) if args.command == "run" else merge(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
