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

import numpy as np
from scipy import ndimage

from . import background

DEFAULTS = {
    "refmask": {
        "tissue": "neuropil",          # the reference tissue (only neuropil so far)
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
    (preview stats p0_5 / p99_5). Returns ({tile: bool cell mask}, {tile: cell scores}, threshold)."""
    s = {**DEFAULTS["refmask"], **(settings or {})}
    if s["tissue"] not in TISSUES:
        raise ValueError(f"refmask.tissue must be one of {TISSUES}, got {s['tissue']!r}")
    from .preview import thumb_values   # preview imports matplotlib; keep refmask light
    scores, tissue = {}, {}
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
