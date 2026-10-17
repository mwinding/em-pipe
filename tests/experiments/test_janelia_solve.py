"""Synthetic test of the Janelia-style patch solve (tools/experiments/janelia_solve.py).

Four tiles (2 x 2, 1000 px, 100 px overlaps) whose raw values relate to the scene through per-tile
gains plus within-tile gradients the flat-field missed. One level per tile cannot match all four
seams (the gradients make the seam relations vary along each seam and the loop of four seams not
close); the patch maps can.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "experiments"))
import janelia_solve as js  # noqa: E402

R = js.R
W = H = 1000
ORIGIN = {"0-0": (0, 0), "0-1": (900, 0), "1-0": (0, 900), "1-1": (900, 900)}


FLAT = {"0-0": (1.0, -100.0), "0-1": (1.03, -200.0), "1-0": (0.98, 150.0), "1-1": (1.0, 40.0)}


def truth(tile, x, y, flat=False):
    """True correction (g, c) of a tile at its pixels: scene = g (v - R) + R + c."""
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


def bins(grid, flat=False, noise=0.0, seed=0):
    """Bin relations of every overlapping pair (as janelia_bins.py would measure them); noise: raw
    units of error in each relation's value at its centre."""
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
            pa = np.minimum((ya // (H / grid)).astype(int), grid - 1) * grid + np.minimum((xa // (W / grid)).astype(int), grid - 1)
            pb = np.minimum((yb // (H / grid)).astype(int), grid - 1) * grid + np.minimum((xb // (W / grid)).astype(int), grid - 1)
            for key in np.unique(pa * 1000 + pb):
                sel = pa * 1000 + pb == key
                if sel.sum() < 20:
                    continue
                m = [v[sel].mean() for v in (xa, ya, xb, yb)]
                g_a, c_a = (float(v) for v in truth(a, m[0], m[1], flat))
                g_b, c_b = (float(v) for v in truth(b, m[2], m[3], flat))
                gain = g_b / g_a
                offset = R + (c_b - c_a) / g_a - gain * R + noise * rng.normal()
                rows.append({"z": 0, "a": a, "b": b, "pa": key // 1000, "pb": key % 1000, "gain": gain,
                             "offset": offset, "n": 16 * int(sel.sum()), "cnr": 20.0, "centre": 36500.0,
                             "xa": m[0], "ya": m[1], "xb": m[2], "yb": m[3]})
    return pd.DataFrame(rows)


def steps(grid, flat=False, noise=0.0, **kw):
    rels = bins(grid, flat, noise)
    maps = js.solve_patches(rels, grid=grid, **kw)
    hw = {t: (H, W) for t in ORIGIN}
    return js.predicted_steps(rels, maps, (R - 2500, R + 2500), hw), maps, rels


def test_patch_solve_removes_mismatch_tile_solve_cannot():
    tile_level, _, _ = steps(1)
    patch, maps, _ = steps(8)
    worst_tile = tile_level["after"].abs().max()
    worst_patch = patch["after"].abs().max()
    assert worst_tile > 4.0                 # greys: one level per tile leaves visible seams
    assert worst_patch < 0.25 * worst_tile
    assert patch["after"].abs().median() < 0.5
    g = np.concatenate([v[0].ravel() for v in maps.values()])
    c = np.concatenate([v[1].ravel() for v in maps.values()])
    assert abs(g.mean() - 1) < 1e-3 and abs(c.mean()) < 1e-3 * R


def interior_range(maps, v=36500.0):
    """Largest within-tile range of the maps at raw value v, in greys of a 5000-raw window."""
    xs, ys = np.meshgrid(np.linspace(0, 999, 21), np.linspace(0, 999, 21))
    out = 0.0
    for g, c in maps.values():
        val = js.interp_grid(g, xs, ys, W, H) * (v - R) + js.interp_grid(c, xs, ys, W, H)
        out = max(out, float(np.ptp(val)) / 5000 * 255)
    return out


def test_consistent_tiles_give_flat_maps():
    """Without within-tile gradients the seams close with one level per tile, and the patch solve
    must not invent gradients: flat maps, also with noisy relations."""
    st, maps, _ = steps(8, flat=True)
    assert st["after"].abs().max() < 0.2
    assert interior_range(maps) < 0.5
    st, maps, _ = steps(8, flat=True, noise=40.0)     # ~2 greys of noise per bin relation
    assert interior_range(maps) < 4.0


def test_shape_plus_level_reproduces_patch_solve():
    """Variant B's algebra: exact shapes + a grid-1 level solve give the same seams as the patch solve."""
    patch, maps, rels = steps(8)
    shapes = {t: (g / g.mean(), c - c.mean()) for t, (g, c) in maps.items()}
    lev = js.solve_patches(js.shape_corrected(rels, shapes, 8), grid=1, smooth=0.0, prior=0.1)
    composed = {t: js.compose(shapes[t], lev[t]) for t in lev}
    again = js.predicted_steps(rels, composed, (R - 2500, R + 2500), {t: (H, W) for t in ORIGIN})
    # the level solve is regularised differently from the patch solve: equal to within a fraction of a grey
    assert np.abs(again["after"].to_numpy() - patch["after"].to_numpy()).max() < 0.5
    assert again["after"].abs().max() <= patch["after"].abs().max() + 0.1


def test_map_rows_matches_interp_grid():
    rng = np.random.default_rng(0)
    vals = rng.normal(size=(8, 8))
    rows, cols = np.arange(0, 1000, 37), np.arange(5, 1000, 41)
    full = js.map_rows(vals, rows, cols, 1000, 1000)
    X, Y = np.meshgrid(cols, rows)
    assert np.allclose(full, js.interp_grid(vals, X, Y, 1000, 1000), atol=1e-5)
