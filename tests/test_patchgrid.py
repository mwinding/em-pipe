"""patchgrid: within-tile shapes from the overlaps' patch bins, the tile balance on them, and render applying them."""

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import synth
from pipeline import intensity, omezarr, overlaps, patchgrid, render, shading
from pipeline.config import load_config, step_defaults
from test_preview import run_preview
from test_render import SMALL, _aligned, _corr, _setup

R = 36000.0

# ----- solve: four 1000 px tiles (2 x 2, 100 px overlaps) with bins computed from known maps -----

W = H = 1000
ORIGIN = {"0-0": (0, 0), "0-1": (900, 0), "1-0": (0, 900), "1-1": (900, 900)}
FLAT = {"0-0": (1.0, -100.0), "0-1": (1.03, -200.0), "1-0": (0.98, 150.0), "1-1": (1.0, 40.0)}


def true_map(tile, x, y, flat=False):
    """True correction (g, c) of a tile at its pixels: scene = g (v - R) + R + c. Not flat: within-tile
    gradients the flat-field missed, so the seams' relations change along each seam and the loop of
    four seams doesn't close with one level per tile."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if flat:
        return np.full_like(x, FLAT[tile][0]), np.full_like(x, FLAT[tile][1]) + 0 * y
    if tile == "0-0":
        return np.ones_like(x), 300 * x / W - 100
    if tile == "0-1":
        return np.full_like(x, 1.03), np.full_like(x, -200.0) + 0 * y
    if tile == "1-0":
        return np.full_like(x, 0.98), 150 + 250 * y / H
    return np.ones_like(x), -120 * x / W + 0 * y


def make_bins(grid, flat=False, noise=0.0, cnr=20.0, seed=0, truth=None):
    """Bin relations of every overlapping pair, as overlaps measures them; noise: raw error of each
    relation's value at its centre; truth(tile, x, y) -> (g, c): the true maps (default true_map)."""
    truth = truth or (lambda tile, x, y: true_map(tile, x, y, flat))
    rng = np.random.default_rng(seed)
    rows = []
    tiles = sorted(ORIGIN)
    for i, a in enumerate(tiles):
        for b in tiles[i + 1:]:
            (ax, ay), (bx, by) = ORIGIN[a], ORIGIN[b]
            gx = np.arange(max(ax, bx), min(ax, bx) + W, 4) + 0.5
            gy = np.arange(max(ay, by), min(ay, by) + H, 4) + 0.5
            if not len(gx) or not len(gy):
                continue
            X, Y = (v.ravel() for v in np.meshgrid(gx, gy))
            xa, ya, xb, yb = X - ax, Y - ay, X - bx, Y - by
            pa = overlaps.patch_index(xa, ya, W, H, grid)
            pb = overlaps.patch_index(xb, yb, W, H, grid)
            for key in np.unique(pa * 1000 + pb):
                sel = pa * 1000 + pb == key
                if sel.sum() < 20:
                    continue
                m = [v[sel].mean() for v in (xa, ya, xb, yb)]
                g_a, c_a = (float(v) for v in truth(a, m[0], m[1]))
                g_b, c_b = (float(v) for v in truth(b, m[2], m[3]))
                gain = g_b / g_a
                offset = R + (c_b - c_a) / g_a - gain * R + noise * rng.normal()
                rows.append({"z": 0, "segment": 0, "a": a, "b": b, "pa": key // 1000, "pb": key % 1000, "gain": gain,
                             "offset": offset, "n": 16 * int(sel.sum()), "cnr": cnr, "centre": 36500.0,
                             "xa": m[0], "ya": m[1], "xb": m[2], "yb": m[3], "ha": H, "wa": W, "hb": H, "wb": W})
    return pd.DataFrame(rows)


def predicted_steps(bins, maps, span=5000.0):
    """Per bin: the step (a minus b, greys of a ``span``-wide window) at its centre after the maps,
    each taken (bilinear) at the bin's mean position, as render applies them."""
    out = []
    for r in bins.itertuples():
        ga, ca = (float(patchgrid.at_points(v, r.xa, r.ya, H, W)) for v in maps[r.a])
        gb, cb = (float(patchgrid.at_points(v, r.xb, r.yb, H, W)) for v in maps[r.b])
        va = r.gain * r.centre + r.offset
        out.append(((ga * (va - R) + ca) - (gb * (r.centre - R) + cb)) / span * 255)
    return np.array(out)


def within_range(maps, v=36500.0):
    """Largest within-tile range of the maps at raw value v, in greys of a 5000-raw window."""
    xs, ys = np.meshgrid(np.linspace(0, W - 1, 21), np.linspace(0, H - 1, 21))
    return max(float(np.ptp(patchgrid.at_points(g, xs, ys, H, W) * (v - R) + patchgrid.at_points(c, xs, ys, H, W)))
               for g, c in maps.values()) / 5000 * 255


def gain_ratio(maps):
    return max(g.max() / g.min() for g, _ in maps.values())


def test_patch_solve_removes_mismatch_one_level_per_tile_cannot():
    one = make_bins(1)
    tile_level = predicted_steps(one, patchgrid.solve_slice(one, 1, R))
    bins = make_bins(8)
    maps = patchgrid.solve_slice(bins, 8, R, smooth=0.3)
    patch = predicted_steps(bins, maps)
    assert np.abs(tile_level).max() > 4.0              # greys: one level per tile leaves visible seams
    assert np.abs(patch).max() < 0.25 * np.abs(tile_level).max() and np.median(np.abs(patch)) < 0.5
    g = np.concatenate([v[0].ravel() for v in maps.values()])
    c = np.concatenate([v[1].ravel() for v in maps.values()])
    assert abs(g.mean() - 1) < 1e-3 and abs(c.mean()) < 1e-3 * R
    # the chosen 35h smoothness (1.0) is stiffer, yet still far better than one level per tile
    stiff = predicted_steps(bins, patchgrid.solve_slice(bins, 8, R, smooth=1.0))
    assert np.abs(stiff).max() < 0.5 * np.abs(tile_level).max()


def test_consistent_tiles_give_flat_maps():
    """Without within-tile gradients one level per tile closes every seam: the patch solve must not
    invent gradients, also with noisy relations."""
    bins = make_bins(8, flat=True)
    maps = patchgrid.solve_slice(bins, 8, R)
    assert np.abs(predicted_steps(bins, maps)).max() < 0.2 and within_range(maps) < 0.5
    noisy = make_bins(8, flat=True, noise=40.0)        # ~2 greys of noise per bin relation
    assert within_range(patchgrid.solve_slice(noisy, 8, R)) < 4.0


def test_resin_bins_give_no_gain_equation():
    """Bins with little contrast (resin only) pin a seam's level but not its gain: theirs is noise
    (35h: 0.4-1.9), which bent the maps in the 3x3 bottom corners. Below gain_cnr_min they only
    enter the c solve."""
    bins = make_bins(8, flat=True)
    resin = bins["b"].eq("1-1") & bins["a"].isin(["0-1", "1-0"])   # a corner tile with resin on both its seams
    wild = np.random.default_rng(3).uniform(0.4, 1.9, int(resin.sum()))
    m = bins.loc[resin, "centre"]
    # the same value at the centre (the level is right), a meaningless slope
    bins.loc[resin, "offset"] += (bins.loc[resin, "gain"] - wild) * m
    bins.loc[resin, "gain"] = wild
    bins.loc[resin, "cnr"] = 4.5
    kept = patchgrid.solve_slice(bins, 8, R, gain_cnr_min=0.0)
    cut = patchgrid.solve_slice(bins, 8, R, gain_cnr_min=5.0)
    assert gain_ratio(kept) > 1.03
    assert gain_ratio(cut) < 1.002
    assert np.abs(predicted_steps(bins[~resin], cut)).max() < 0.2


def set_gain(bins, sel, gain):
    """Give the bins ``sel`` another gain with the same value at their centre (the level stays right)."""
    bins.loc[sel, "offset"] += (bins.loc[sel, "gain"] - gain) * bins.loc[sel, "centre"]
    bins.loc[sel, "gain"] = gain


def test_35h_cut_off_drops_a_mostly_resin_corner_bin():
    """35h, segment 0: where 2-0 and 2-1 meet at the bottom the seam is resin, and one bin there
    (2-0 patch 63 | 2-1 patch 56) has enough tissue for cnr 6.2 but a meaningless gain of 1.53, in
    every slice, so the median over slices keeps what it does. With nothing else pinning the gains
    there, at gain_cnr_min 5 it bent both tiles' maps (2-0: 12 %); the 35h settings must not."""
    bins = make_bins(8, flat=True)
    seam = bins["a"].eq("1-0") & bins["b"].eq("1-1")                 # the bottom tiles' seam
    resin = seam & (bins["pa"] // 8 >= 6)                             # its bottom two patch rows
    set_gain(bins, resin, np.random.default_rng(3).uniform(0.4, 1.9, int(resin.sum())))
    bins.loc[resin, "cnr"] = 3.0
    corner = seam & bins["pa"].eq(63) & bins["pb"].eq(56)
    set_gain(bins, corner, 1.53)
    bins.loc[corner, "cnr"] = 6.2
    c = load_config(Path(__file__).resolve().parents[1] / "configs" / "P667_EM05024_35h.yaml",
                    step_defaults=patchgrid.DEFAULTS)["patchgrid"]
    kw = {k: c[k] for k in ("smooth", "prior", "gain_power")}
    bent = patchgrid.solve_slice(bins, 8, R, gain_cnr_min=5.0, **kw)
    assert gain_ratio(bent) > 1.05
    maps = patchgrid.solve_slice(bins, 8, R, gain_cnr_min=c["gain_cnr_min"], **kw)
    assert gain_ratio(maps) < 1.005
    assert np.abs(predicted_steps(bins[bins["cnr"] > 10], maps)).max() < 0.1


def test_prior_pulls_a_weakly_pinned_tile_towards_gain_1():
    """A tile whose only gain equation is one weak bin (cnr 6) with a gain 1.5x too high: without the
    prior its gain follows that bin; the prior pulls it part of the way back. (A tile with no gain
    equation at all can't show the prior: with the mean gain fixed at 1, the least-squares solve's
    minimum-norm answer for an undetermined level is the same pull towards 1.)"""
    bins = make_bins(8, flat=True)
    touch = bins["a"].eq("1-1") | bins["b"].eq("1-1")
    bins.loc[touch, "cnr"] = 3.0
    one = bins.index[touch & bins["a"].eq("1-0") & bins["b"].eq("1-1") & bins["pa"].eq(39) & bins["pb"].eq(32)]
    assert len(one) == 1
    set_gain(bins, bins.index.isin(one), 1.5 * bins.loc[one[0], "gain"])
    bins.loc[one, "cnr"] = 6.0
    true = FLAT["1-1"][0] / FLAT["1-0"][0]

    def ratio(prior):
        maps = patchgrid.solve_slice(bins, 8, R, prior=prior)
        return maps["1-1"][0].mean() / maps["1-0"][0].mean() / true

    assert ratio(0.0) == pytest.approx(1.5, abs=0.01)
    assert 1.3 < ratio(0.02) < 1.47 and ratio(0.2) < 1.1


def test_poorly_measured_gains_give_smoother_maps():
    """The smoothness scales with the bins' sqrt(n), not with their gain weights, so where every gain
    is poorly measured (low contrast) the maps get smoother instead of freer."""
    noisy = make_bins(8, noise=40.0, seed=1)
    noisy["gain"] *= np.random.default_rng(2).normal(1, 0.02, len(noisy))
    sharp = patchgrid.solve_slice(noisy, 8, R)
    weak = patchgrid.solve_slice(noisy.assign(cnr=6.0), 8, R)
    assert gain_ratio(weak) < 1 + 0.5 * (gain_ratio(sharp) - 1)


def test_bin_weights():
    bins = pd.DataFrame({"n": [100, 400, 400, 400], "cnr": [20.0, 10.0, 6.0, 4.0]})
    w, gw = patchgrid.bin_weights(bins, 2.0, 5.0)
    np.testing.assert_allclose(w, [10, 20, 20, 20])
    np.testing.assert_allclose(gw, [1.0, 1.0, 0.36, 0.0])


# ----- shapes and their algebra -----

def smooth_shape(seed, grid=8):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:grid, 0:grid] / (grid - 1)
    sg = 1 + 0.02 * rng.normal() * xx + 0.015 * rng.normal() * yy ** 2
    sc = 150 * rng.normal() * xx * yy - 80 * rng.normal() * yy
    return sg / sg.mean(), sc - sc.mean()


def apply_shape(shape, v, x, y, h=H, w=W):
    return patchgrid.at_points(shape[0], x, y, h, w) * (v - R) + R + patchgrid.at_points(shape[1], x, y, h, w)


def test_shape_of_is_the_map_without_its_level():
    g, c = smooth_shape(1)
    g, c = 1.04 * g, c + 250
    sg, sc = patchgrid.shape_of(g, c)
    assert sg.mean() == pytest.approx(1) and sc.mean() == pytest.approx(0, abs=1e-9)
    v = np.linspace(33000, 40000, 7)
    for p in (0, 17, 63):   # map = level (mean g, mean c) after the shape, patch by patch
        u = sg.ravel()[p] * (v - R) + R + sc.ravel()[p]
        np.testing.assert_allclose(g.mean() * (u - R) + R + c.mean(), g.ravel()[p] * (v - R) + R + c.ravel()[p])


def test_shape_corrected_relations():
    """A bin's relation carried over to shape-corrected values: u_a = gain' u_b + offset' holds
    wherever v_a = gain v_b + offset did, with each side's shape taken at the bin's position."""
    bins = make_bins(8).iloc[::5].reset_index(drop=True)
    shapes = {(0, t): smooth_shape(k) for k, t in enumerate(sorted(ORIGIN))}
    corr = patchgrid.shape_corrected(bins, shapes, R)
    for r, q in zip(bins.itertuples(), corr.itertuples()):
        vb = np.linspace(33000, 40000, 5)
        ua = apply_shape(shapes[(0, r.a)], r.gain * vb + r.offset, r.xa, r.ya)
        ub = apply_shape(shapes[(0, r.b)], vb, r.xb, r.yb)
        np.testing.assert_allclose(ua, q.gain * ub + q.offset, rtol=1e-12, atol=1e-6)
        assert q.centre == pytest.approx(float(apply_shape(shapes[(0, r.b)], r.centre, r.xb, r.yb)))
    # no shape (or the identity): unchanged
    same = patchgrid.shape_corrected(bins, {}, R)
    pd.testing.assert_frame_equal(same, bins)
    ident = patchgrid.shape_corrected(bins, {(0, t): patchgrid.identity(8) for t in ORIGIN}, R)
    np.testing.assert_allclose(ident[["gain", "offset", "centre"]], bins[["gain", "offset", "centre"]])


def test_bilinear_between_patch_centres_clamped_at_the_border():
    rng = np.random.default_rng(0)
    vals = rng.normal(size=(8, 8))
    rows, cols = np.arange(0, 1000, 37), np.arange(5, 1000, 41)
    X, Y = np.meshgrid(cols, rows)
    np.testing.assert_allclose(patchgrid.on_grid(vals, rows, cols, 1000, 1000), patchgrid.at_points(vals, X, Y, 1000, 1000))
    centres = (np.arange(8) + 0.5) * 125 - 0.5                # patch centres of a 1000 px tile, 8 patches
    np.testing.assert_allclose(patchgrid.on_grid(vals, centres, centres, 1000, 1000), vals, atol=1e-12)
    edge = patchgrid.on_grid(vals, [0, 30], [999, 980], 1000, 1000)   # before the first / after the last centre
    np.testing.assert_allclose(edge, vals[0, 7], atol=1e-12)
    assert np.all(patchgrid.on_grid(np.full((1, 1), 2.5), [0, 500], [3, 9, 999], 1000, 1000) == 2.5)


# ----- render's fields -----

TILE = (192, 224)


def test_fields_constant_shape_is_an_exact_affine_change():
    rng = np.random.default_rng(0)
    img = rng.integers(26000, 41000, (100, 150)).astype(np.uint16)
    rows, cols = slice(40, 140), slice(30, 180)
    shape = (np.full((4, 4), 1.05), np.full((4, 4), 30.0))
    got = patchgrid.apply_fields(img, patchgrid.tile_fields(shape, R, None, TILE), TILE, rows, cols, block=32)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, 1.05 * (img - R) + R + 30, rtol=2e-7, atol=0.02)
    same = patchgrid.apply_fields(img, patchgrid.tile_fields(patchgrid.identity(4), R, None, TILE), TILE, rows, cols)
    np.testing.assert_allclose(same, img, rtol=1e-7)


def test_fields_fuse_flat_field_and_shape():
    rng = np.random.default_rng(1)
    img = rng.integers(26000, 41000, TILE).astype(np.uint16)
    yy, xx = np.mgrid[0:48, 0:56]
    field = (1 + 0.1 * (xx / 55 - 0.5) - 0.04 * (yy / 47 - 0.5) ** 2).astype(np.float32)
    full = (slice(0, TILE[0]), slice(0, TILE[1]))
    # identity shape: the flat-field division render does without patchgrid
    flat = patchgrid.apply_fields(img, patchgrid.tile_fields(patchgrid.identity(4), R, field, TILE), TILE, *full)
    np.testing.assert_allclose(flat, img / shading.gain_window(field, TILE, *full), rtol=1e-5)
    # a smooth shape after the flat-field, against the exact composition
    shape = smooth_shape(4, grid=4)
    got = patchgrid.apply_fields(img, patchgrid.tile_fields(shape, R, field, TILE), TILE, *full, block=50)
    ys, xs = np.arange(TILE[0]), np.arange(TILE[1])
    want = (patchgrid.on_grid(shape[0], ys, xs, *TILE) * (img / shading.gain_window(field, TILE, *full).astype(float) - R)
            + R + patchgrid.on_grid(shape[1], ys, xs, *TILE))
    # Raw units. The fields are the shape sampled on the flat-field's grid and resampled linearly, which
    # rounds the shape's kinks at patch centres: this steep shape (~3 raw/px) is off by up to ~3 there
    # (0.15 grey of a 5000-raw window), elsewhere by ~0.
    err = np.abs(got - want)
    assert err.max() < 4.0 and np.percentile(err, 99) < 1.5 and np.median(err) < 0.1
    part = patchgrid.apply_fields(img[50:120, 60:200], patchgrid.tile_fields(shape, R, field, TILE), TILE,
                                  slice(50, 120), slice(60, 200), block=16)
    np.testing.assert_allclose(part, got[50:120, 60:200], atol=1.0)   # the fields' resampling starts elsewhere


def test_fields_with_the_window_folded_in():
    """render's patchgrid path folds the tile's window into the fields: the same as the corrected
    values through render._normalise, to far below a grey, clipped to [0, 1]."""
    rng = np.random.default_rng(2)
    img = rng.integers(26000, 41000, TILE).astype(np.uint16)
    yy, xx = np.mgrid[0:48, 0:56]
    field = (1 + 0.1 * (xx / 55 - 0.5) - 0.04 * (yy / 47 - 0.5) ** 2).astype(np.float32)
    fields = patchgrid.tile_fields(smooth_shape(5, grid=4), R, field, TILE)
    for rows, cols in ((slice(0, TILE[0]), slice(0, TILE[1])), (slice(37, 150), slice(20, 201))):
        part = img[rows, cols]
        want = render._normalise(patchgrid.apply_fields(part, fields, TILE, rows, cols), 33000.0, 38000.0)
        got = patchgrid.apply_fields(part, fields, TILE, rows, cols, (33000.0, 38000.0), block=40)
        assert got.dtype == np.float32 and got.min() == 0 and got.max() == 1
        np.testing.assert_allclose(got, want, atol=2e-6)   # of [0, 1]: 0.0005 grey


# ----- intensity: the tile balance on shape-corrected bins -----

def test_tile_balance_on_shape_corrected_bins(tmp_path):
    """intensity with patchgrid: per slice one gain/offset per tile from the patch bins, carried over
    to u by the shape of their (segment, tile) and each matched at its own centre, resin bins (wild
    gains, cnr 4.5) giving no gain equation. Tiles that are one level apart once their shapes are
    applied must close every good bin. There is no relations.csv: intensity must not read it."""
    out = tmp_path / "out"
    zseg = [(0, 0), (1, 1), (2, 1)]            # (z, segment): two segments with their own shapes
    shapes = {(s, t): smooth_shape(10 * s + k) for s in (0, 1) for k, t in enumerate(sorted(ORIGIN))}
    rng = np.random.default_rng(5)
    parts = []
    for z, seg in zseg:
        def truth(t, x, y, seg=seg):           # the tile's level (FLAT) after its shape
            sg, sc = (patchgrid.at_points(v, x, y, H, W) for v in shapes[(seg, t)])
            return FLAT[t][0] * sg, FLAT[t][0] * sc + FLAT[t][1]
        b = make_bins(8, truth=truth).assign(z=z)
        b["centre"] = rng.uniform(34000, 39000, len(b))   # the relations hold at any value
        resin = b["b"].eq("1-1") & b["a"].isin(["0-1", "1-0"])
        set_gain(b, resin, rng.uniform(0.4, 1.9, int(resin.sum())))
        b.loc[resin, "cnr"] = 4.5
        parts.append(b)
    bins = pd.concat(parts, ignore_index=True)
    bins = bins.assign(n_total=bins["n"], rms=50.0, median_a=bins["gain"] * bins["centre"] + bins["offset"])
    for d in ("check", "overlaps"):
        (out / "work" / d).mkdir(parents=True)
    bins[overlaps.PATCH_COLUMNS].to_csv(out / "work" / "overlaps" / "patches.csv", index=False, float_format="%.12g")
    (out / "work" / "overlaps" / "patches.json").write_text(json.dumps({"patch_grid": 8, "patch_min_pairs": 2000}))
    slices = pd.DataFrame([{"z": z, "timestamp": f"2026-09-2{z}T10:00:00", "tile": t, "tile_row": int(t[0]),
                            "tile_col": int(t[2]), "file": f"s{seg}_{t}.tif", "index": z, "height": H, "width": W,
                            "segment": seg, "seam": False, "excluded": False, "exclude_reason": "", "label": ""}
                           for z, seg in zseg for t in sorted(ORIGIN)])
    slices.to_csv(out / "work" / "check" / "slices.csv", index=False)
    patchgrid.write_shapes(out / "work" / "patchgrid" / "shapes.npz", shapes, R, 8)
    cfg = load_config(synth.write_config(tmp_path / "c.yaml", tmp_path, out, overlaps={"patch_grid": 8},
                                         patchgrid={"enabled": True}, intensity={"balance_source": "overlaps"}),
                      step_defaults=step_defaults())
    meas, every = intensity.overlap_measurements(cfg)
    assert every == 1 and sorted(meas["z"].unique()) == [0, 1, 2] and len(meas) == 12
    lv = meas.set_index(["z", "tile"])
    u = patchgrid.shape_corrected(bins.assign(ha=H, wa=W, hb=H, wb=W, segment=bins["z"].map(dict(zseg))), shapes, R)
    at_a, at_b = (lv.loc[list(zip(u["z"], u[side]))] for side in ("a", "b"))
    va = u["gain"].to_numpy() * u["centre"].to_numpy() + u["offset"].to_numpy()
    step = (at_a["g"].to_numpy() * va + at_a["o"].to_numpy()) - (at_b["g"].to_numpy() * u["centre"].to_numpy()
                                                                  + at_b["o"].to_numpy())
    good = u["cnr"].to_numpy() > 10
    assert np.abs(step[good]).max() / 5000 * 255 < 0.05     # greys of a 5000-raw window
    for z, _ in zseg:                                         # gains as the true levels, relative to each other
        for t in ("0-1", "1-0", "1-1"):
            assert lv.loc[(z, t), "g"] / lv.loc[(z, "0-0"), "g"] == pytest.approx(FLAT[t][0], abs=1e-3)


# ----- the chain: overlaps (patch_grid) -> patchgrid -> intensity -> render on synthetic tiles -----

N = 20   # slices per segment
# Two segments (tile layouts) with their own tile size, each with brightness changes towards one corner of two
# tiles: too local for the degree-2 flat-field. Tile names repeat across segments; their shapes must not.
SEGMENTS = [
    dict(tile_shape=TILE, tile_edge={"0-1": ("bottom-left", -0.10, 50), "1-0": ("top-right", 0.08, 50)},
         start=datetime(2026, 9, 20, 10), seed=21),
    dict(tile_shape=(208, 200), tile_edge={"0-0": ("bottom-right", 0.09, 50), "1-1": ("top-left", -0.10, 50)},
         start=datetime(2026, 9, 22, 10), seed=22)]
OFFSET = [(0, 0), (6, -4)]   # each segment's place in the aligned frame
SECTIONS = dict(
    preview={"factor": 4}, shading={"enabled": True, "every": 1},
    overlaps={"enabled": True, "factor": 2, "trim_px": 2, "smooth_px": 0.5, "min_pixels": 50, "chunk_slices": 8,
              "patch_grid": 4, "patch_min_pairs": 100},
    intensity={"window": "percentile", "per_slice": "slice", "balance_source": "overlaps", "balance_smooth_slices": 1},
    render={**SMALL, "blend": "cut"})
# 4 x 4 patches of ~50 px on these small tiles: less smoothing than 35h's 8 x 8 (smooth 1.0) to follow the corner.
PATCHGRID = {"every": 2, "smooth": 0.3}


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    """Synthetic run up to overlaps (with patch bins), then intensity and render with patchgrid off and on."""
    root = tmp_path_factory.mktemp("patchgrid")
    truths = [synth.make_dataset(root / "raw", n_slices=N, overlap=(40, 48), tile_jitter=0.03, noise=0.002, **seg)
              for seg in SEGMENTS]
    cfg, base = _setup(root / "base", list(zip(truths, OFFSET)), levels=False, **SECTIONS)
    run_preview(cfg)
    assert shading.main(["--config", str(cfg)]) == 0
    assert overlaps.main(["run", "--config", str(cfg)]) == 0
    assert overlaps.main(["merge", "--config", str(cfg)]) == 0
    runs = {}
    for name, on in (("off", False), ("on", True)):
        out = root / name
        shutil.copytree(base, out)
        c = synth.write_config(root / f"{name}.yaml", truths[0].raw_dir, out, **SECTIONS,
                               patchgrid={**PATCHGRID, "enabled": on})
        if on:
            assert patchgrid.main(["--config", str(c)]) == 0
            (out / "work" / "overlaps" / "relations.csv").unlink()   # the balance on patch bins must not need it
        assert intensity.main(["--config", str(c)]) == 0
        assert render.main(["init", "--config", str(c)]) == 0
        assert render.main(["run", "--config", str(c)]) == 0
        runs[name] = (c, out)
    return truths, root, runs


def _render_json(out):
    return json.loads((out / "work" / "render" / "synthetic" / "render.json").read_text())


def seam_steps(cfg, out):
    """Per (z, seam, 16 px stretch along it): the median of tile a minus tile b over the middle third of
    their overlap, each tile as render places it (flat-field, shape, window), in greys."""
    c = load_config(cfg, step_defaults=step_defaults())
    plan = pd.read_csv(out / "work" / "render" / "synthetic" / "tiles.csv", dtype={"tile": str, "file": str})
    rows = []
    with ThreadPoolExecutor(1) as pool:
        r = render._Renderer(c, _render_json(out), plan, pool)
        for z, tiles in r.tiles.items():
            placed = {t["tile"]: r._tile(t) for t in tiles}
            for a in placed:
                for b in placed:
                    (ra, ca), (rb, cb) = (map(int, t.split("-")) for t in (a, b))
                    if not (a < b and (ra == rb or ca == cb)):
                        continue
                    (wa, va, *_), (wb, vb, *_) = placed[a], placed[b]
                    y0, y1 = max(wa[0].start, wb[0].start), min(wa[0].stop, wb[0].stop)
                    x0, x1 = max(wa[1].start, wb[1].start), min(wa[1].stop, wb[1].stop)
                    d = 255 * (va[y0 - wa[0].start:y1 - wa[0].start, x0 - wa[1].start:x1 - wa[1].start]
                               - vb[y0 - wb[0].start:y1 - wb[0].start, x0 - wb[1].start:x1 - wb[1].start])
                    d = d if ra == rb else d.T                # rows run along the seam
                    k = d.shape[1] // 3
                    rows += [(z, r.segment[z], f"{a}|{b}", s, float(np.median(d[s:s + 16, k:-k])))
                             for s in range(0, d.shape[0] - 15, 16)]
    return pd.DataFrame(rows, columns=["z", "segment", "seam", "pos", "step"])


def test_patchgrid_removes_seams_the_flat_field_leaves(chain):
    """In both segments, each with its own tile size and corners: a segment given another segment's
    shapes (same tile names) would keep its seams."""
    truths, root, runs = chain
    steps = {name: seam_steps(*runs[name]) for name in runs}
    # systematic: per stretch of seam, the median over slices (the shapes are static; the levels follow each slice)
    sysm = {name: s.groupby(["segment", "seam", "pos"])["step"].median().abs() for name, s in steps.items()}
    for seg in range(len(SEGMENTS)):
        off, on = sysm["off"].xs(seg), sysm["on"].xs(seg)
        print("segment %d: systematic seam step (greys), median / p90: off %.1f / %.1f, on %.1f / %.1f" % (
            seg, off.median(), off.quantile(0.9), on.median(), on.quantile(0.9)))
        assert off.median() > 4
        assert on.median() < 0.45 * off.median() and on.median() < 3
        assert on.quantile(0.9) < 0.7 * off.quantile(0.9)
        each = {name: s.loc[s["segment"] == seg, "step"].abs().median() for name, s in steps.items()}
        assert each["on"] < 0.5 * each["off"]
    # the rendered volume still follows the anatomy
    vol = omezarr.open_scale(runs["on"][1] / "synthetic.ome.zarr", 0).read().result()
    origin = np.rint(_render_json(runs["on"][1])["origin_xy"]).astype(int)
    got, want = [], []
    for k, (t, zl, off) in enumerate((t, zl, off) for t, off in zip(truths, OFFSET) for zl in range(N)):
        (dx, dy), (H_, W_) = t.drift[zl] + off - origin, t.montage_shape
        got.append(vol[k, dy:dy + H_, dx:dx + W_].ravel())
        want.append(t.montage(zl).ravel())
    assert len(vol) == len(SEGMENTS) * N and _corr(np.concatenate(got), np.concatenate(want)) > 0.98


def test_outputs_and_render_plan(chain):
    truths, root, runs = chain
    on = runs["on"][1]
    info = json.loads((on / "work" / "patchgrid" / "shapes.json").read_text())
    bins = pd.read_csv(on / "work" / "overlaps" / "patches.csv", dtype={"a": str, "b": str})
    assert info["grid"] == 4 and info["R_from"] == "median bin centre"
    assert info["R"] == pytest.approx(np.median(bins["centre"]))
    assert {(s["segment"], s["tile"]) for s in info["shapes"]} == {(k, t) for k, tr in enumerate(truths)
                                                                    for t in tr.tiles}
    assert all(s["slices"] == N // 2 for s in info["shapes"])   # every 2nd measured slice of each segment
    shapes, R_ = patchgrid.load(load_config(runs["on"][0], step_defaults=patchgrid.DEFAULTS))
    assert R_ == pytest.approx(info["R"])
    sg, sc = shapes[(0, "0-1")]
    assert sg.shape == (4, 4) and sg.mean() == pytest.approx(1) and sc.mean() == pytest.approx(0, abs=1e-6)
    # the corner changes, segment 0: 0-1 darker towards its bottom-left, so its shape raises it there (sc > 0),
    # 1-0 brighter towards its top-right
    assert sc[3, 0] > sc[0, 3] + 50
    sg, sc = shapes[(0, "1-0")]
    assert sc[0, 3] < sc[3, 0] - 50
    # segment 1: 0-0 brighter towards its bottom-right, 1-1 darker towards its top-left
    sg, sc = shapes[(1, "0-0")]
    assert sc[3, 3] < sc[0, 0] - 50
    sg, sc = shapes[(1, "1-1")]
    assert sc[0, 0] > sc[3, 3] + 50
    assert (on / "qc" / "patchgrid.png").stat().st_size > 0
    meta_on, meta_off = _render_json(on), _render_json(runs["off"][1])
    assert meta_on["patchgrid"] == patchgrid.digest(load_config(runs["on"][0])) and len(meta_on["patchgrid"]) == 64
    assert "patchgrid" not in meta_off
    # re-solved shapes change the plan: init asks for --overwrite, and run (e.g. an array resumed) refuses
    path = on / "work" / "patchgrid" / "shapes.npz"
    kept = path.read_bytes()
    with np.load(path) as z:
        np.savez(path, **{**z, "R": R_ + 1})
    try:
        assert render.main(["init", "--config", str(runs["on"][0])]) == 1
        with pytest.raises(ValueError, match="patchgrid changed"):
            render.main(["run", "--config", str(runs["on"][0])])
    finally:
        path.write_bytes(kept)


def test_run_refuses_corrections_init_did_not_plan_with(chain, tmp_path):
    """render run reads the flat-fields and shapes from disk in every task. Changed since init (the
    plan's levels were solved for the old ones, and slabs already done keep their markers), or
    switched off in the config, it refuses instead of rendering the remaining slabs with them."""
    truths, root, runs = chain
    cfg, out = runs["on"]
    path = out / "work" / "shading" / "fields.npz"
    kept = path.read_bytes()
    with np.load(path) as z:
        np.savez(path, **{k: 1.01 * z[k] for k in z.files})
    try:
        with pytest.raises(ValueError, match="shading changed"):
            render.main(["run", "--config", str(cfg)])
    finally:
        path.write_bytes(kept)
    off = synth.write_config(tmp_path / "off.yaml", truths[0].raw_dir, out, **SECTIONS,
                             patchgrid={**PATCHGRID, "enabled": False})
    with pytest.raises(ValueError, match="patchgrid changed"):
        render.main(["run", "--config", str(off)])
    plan = pd.read_csv(out / "work" / "render" / "synthetic" / "tiles.csv", dtype={"tile": str, "file": str})
    with pytest.raises(ValueError, match="patchgrid.enabled is false"):
        render._Renderer(load_config(off, step_defaults=step_defaults()), _render_json(out), plan, None)
    assert render.main(["run", "--config", str(cfg)]) == 0   # as planned: every slab is done


def test_plane_meta_renders_slices_as_the_volume(chain):
    """QC tools render slices with the current levels on an older render's canvas (render.plane_meta).
    With patchgrid on, the plan must carry the shapes: then a slice comes out as in the volume rendered
    with them. Kept from the older render (no shapes) they would be dropped while the levels, windows on
    the shape-corrected values, stay."""
    truths, root, runs = chain
    c = load_config(runs["on"][0], step_defaults=step_defaults())
    on, old = _render_json(runs["on"][1]), _render_json(runs["off"][1])
    meta = render.plane_meta(c, old, on["downsample"])
    assert meta["patchgrid"] == on["patchgrid"] and meta["shading"] == on["shading"]
    assert meta["shape"] == [1, *on["shape"][1:]]
    stale = {k: v for k, v in meta.items() if k != "patchgrid"}
    plan = render._tile_plan(c)
    vol = omezarr.open_scale(runs["on"][1] / "synthetic.ome.zarr", 0)
    with ThreadPoolExecutor(2) as pool:
        r, r_stale = render._Renderer(c, meta, plan, pool), render._Renderer(c, stale, plan, pool)
        for k in (3, N + 5):   # one slice of each segment
            want = vol[k].read().result()
            np.testing.assert_array_equal(r.plane(on["planes"][k]), want)
            assert np.abs(r_stale.plane(on["planes"][k]).astype(int) - want).max() >= 4


def test_off_ignores_shapes_on_disk(chain, tmp_path):
    """patchgrid off: render is as without patchgrid, whatever work/patchgrid holds."""
    truths, root, runs = chain
    cfg, out = runs["off"]
    shutil.copytree(runs["on"][1] / "work" / "patchgrid", out / "work" / "patchgrid", dirs_exist_ok=True)
    c = synth.write_config(tmp_path / "again.yaml", truths[0].raw_dir, out, **{**SECTIONS, "render": {
        **SECTIONS["render"], "name": "again.ome.zarr"}}, patchgrid={**PATCHGRID, "enabled": False})
    assert render.main(["init", "--config", str(c)]) == 0
    assert render.main(["run", "--config", str(c)]) == 0
    a = omezarr.open_scale(out / "synthetic.ome.zarr", 0).read().result()
    b = omezarr.open_scale(out / "again.ome.zarr", 0).read().result()
    np.testing.assert_array_equal(a, b)


def test_settings_and_errors(chain, tmp_path):
    truths, root, runs = chain
    raw = truths[0].raw_dir
    cfg, out = runs["on"]
    # a fixed reference intensity is used and recorded; the shapes' seams are the same up to the solve's weights
    c = synth.write_config(tmp_path / "ref.yaml", raw, tmp_path / "ref", **SECTIONS,
                           patchgrid={**PATCHGRID, "enabled": True, "reference": 30000})
    shutil.copytree(out / "work" / "overlaps", tmp_path / "ref" / "work" / "overlaps")
    shutil.copytree(out / "work" / "check", tmp_path / "ref" / "work" / "check")
    assert patchgrid.main(["--config", str(c)]) == 0
    info = json.loads((tmp_path / "ref" / "work" / "patchgrid" / "shapes.json").read_text())
    assert info["R"] == 30000 and info["R_from"] == "config"
    # off: nothing to do
    off = synth.write_config(tmp_path / "off.yaml", raw, out, **SECTIONS)
    assert patchgrid.main(["--config", str(off)]) == 1
    assert patchgrid.load(load_config(off, step_defaults=patchgrid.DEFAULTS)) == ({}, None)
    assert patchgrid.digest(load_config(off, step_defaults=patchgrid.DEFAULTS)) is None
    # bins of another grid than overlaps.patch_grid, or none
    other = synth.write_config(tmp_path / "grid.yaml", raw, out,
                               **{**SECTIONS, "overlaps": {**SECTIONS["overlaps"], "patch_grid": 8}},
                               patchgrid={"enabled": True})
    with pytest.raises(ValueError, match="patch grid 4"):
        patchgrid.main(["--config", str(other)])
    with pytest.raises(FileNotFoundError, match="overlaps.patch_grid"):
        patchgrid.read_bins(load_config(synth.write_config(tmp_path / "e.yaml", raw, tmp_path / "empty")))
    # the thumbnail balance doesn't know the shapes
    thumbs = synth.write_config(tmp_path / "thumbs.yaml", raw, out,
                                **{**SECTIONS, "intensity": {**SECTIONS["intensity"], "balance_source": "thumbnails"}},
                                patchgrid={"enabled": True})
    with pytest.raises(ValueError, match="balance_source: overlaps"):
        intensity.main(["--config", str(thumbs)])
    # render init without the shapes fails before creating anything (on a copy without them)
    bare = tmp_path / "bare"
    shutil.copytree(out, bare, ignore=shutil.ignore_patterns("*.ome.zarr", "patchgrid"))
    missing = synth.write_config(tmp_path / "missing.yaml", raw, bare,
                                 **{**SECTIONS, "render": {**SECTIONS["render"], "name": "missing.ome.zarr"}},
                                 patchgrid={"enabled": True})
    with pytest.raises(FileNotFoundError, match="patchgrid step"):
        render.main(["init", "--config", str(missing)])
    assert not (bare / "missing.ome.zarr").exists()
