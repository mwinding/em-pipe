"""Tests for pipeline.refmask on a synthetic thumbnail: fine-textured 'neuropil', a 'rind' of smooth
round cell bodies with outlines, and flat resin."""

import numpy as np
from scipy import ndimage

from pipeline import features, refmask

C = 16


def synthetic(seed=0, h=320, w=480):
    """Raw-like values: neuropil left, rind right, resin in the top-right corner; and the truth."""
    rng = np.random.default_rng(seed)
    neuropil = ndimage.gaussian_filter(rng.normal(0, 1, (h, w)), 0.8) * 900
    dots = ndimage.gaussian_filter((rng.random((h, w)) < 0.004).astype(float), 1.2) * -4e4
    # rind: Voronoi cells ~ 30 px across (nuclei at 128 nm), smooth inside, soft dark outlines between
    pts = rng.random((int(h * w / 900), 2)) * (h, w)
    yy, xx = np.mgrid[:h, :w]
    d = np.stack([np.hypot(yy - py, xx - px) for py, px in pts])
    d2 = np.sort(d, axis=0)
    outline = np.exp(-((d2[1] - d2[0]) / 4.0) ** 2)
    shade = ndimage.gaussian_filter(rng.normal(0, 1, len(pts))[np.argmin(d, axis=0)], 3)
    rind = 1500 + 800 * shade - 2500 * outline + rng.normal(0, 30, (h, w))
    truth = xx < w // 2
    img = np.where(truth, 30000 + neuropil + dots, 30000 + rind)
    resin = (yy < h // 4) & (xx > 3 * w // 4)
    img[resin] = 34000 + 3 * xx[resin] + rng.normal(0, 20, resin.sum())
    return img, truth, resin


def masks_for(img):
    lo, hi = np.percentile(img, [0.5, 99.5])
    thumb = features.to_uint8(img, lo, hi)
    masks, scores, thr = refmask.slice_masks({"0-0": (thumb, lo, hi)})
    return masks["0-0"], scores["0-0"], thr


def cells(a):
    return refmask._cells(a.astype(float), C) > 0.5


def test_neuropil_found_rind_and_resin_left_out():
    img, truth, resin = synthetic()
    mask, scores, _ = masks_for(img)
    t, r = cells(truth), cells(resin)
    assert mask[t & ~r].mean() > 0.85
    assert mask[~t].mean() < 0.1
    assert not mask[r].any()
    assert np.isnan(scores[r]).mean() > 0.7   # background: no score (but its rim next to tissue)


def test_mask_ignores_brightness_and_contrast():
    img, _, _ = synthetic(seed=1)
    a, _, _ = masks_for(img)
    b, _, _ = masks_for(0.4 * img + 9000)
    assert (a == b).mean() > 0.97


def test_rind_only_slice_gives_little_mask():
    img, truth, _ = synthetic(seed=2)
    rind = np.where(truth, np.fliplr(img), img)   # the rind half mirrored over the neuropil half
    mask, _, _ = masks_for(rind)
    assert mask.mean() < 0.05
