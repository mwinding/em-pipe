"""Tile-level affine solve in windows of slices ("problem areas"), run at the end of align solve.

Everywhere else a tile's transform is align[z] ∘ stitch[(z, tile)]: tiles fitted to each other in
their narrow overlaps only (pulled toward rigid, which those strips pin down poorly), then one
translation per slice. Where that is not enough, e.g. at a change of tile layout whose tiles each
differ slightly in scale and shear from the old ones, every tile of every slice in a window gets
its own affine, solved jointly from

- align's point matches with the tiles of the next few slices (they cover whole tiles, so scale
  and shear are well determined and the affines are free), and
- the current seams between tiles of the same slice (points in each overlap that must stay where
  both tiles put them now).

Slices just outside a window keep their transforms, so a correction fades in and out over the
window. Windows: ``align.local_half_width`` slices either side of every tile layout change
(``align.local_at_layout_changes``) and the z ranges in ``align.local_windows``. Writes
``align/local_tiles.csv`` (tile -> aligned transform for every (z, tile) in a window, used by
render instead of align ∘ stitch) and ``align/local_windows.csv`` (mismatch before and after).
"""

import logging

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

from . import cli, transforms
from .config import step_dir

log = logging.getLogger(__name__)

SEAM_WEIGHT = 4.0    # per seam point, relative to a matched point across slices
SEAM_GRID = (25, 6)  # seam points along and across each overlap
WINDOW_COLUMNS = ["z_first", "z_last", "reason", "tile_slices", "points", "median_before_px",
                  "median_after_px", "p90_before_px", "p90_after_px", "change_median_before_px",
                  "change_median_after_px", "change_p90_before_px", "change_p90_after_px",
                  "seam_median_after_px", "seam_max_after_px", "rejected_points"]


def windows(zs, run, a):
    """[(first position, last position, reason)] in ``zs``, overlapping ones merged."""
    n, hw = len(zs), int(a["local_half_width"])
    out = []
    if a["local_at_layout_changes"]:
        for k in np.flatnonzero(np.diff(run)) + 1:
            out.append((max(0, k - hw), min(n - 1, k + hw - 1), f"layout change z {zs[k - 1]}->{zs[k]}"))
    for z0, z1 in a["local_windows"] or []:
        i0, i1 = np.searchsorted(zs, z0), np.searchsorted(zs, z1, side="right") - 1
        if i0 <= i1:
            out.append((int(i0), int(i1), f"configured z {z0}-{z1}"))
    out.sort()
    merged = []
    for w in out:
        if merged and w[0] <= merged[-1][1] + 1:
            p = merged[-1]
            merged[-1] = (p[0], max(p[1], w[1]), p[2] + "; " + w[2])
        else:
            merged.append(w)
    return merged


def seam_points(T, shapes):
    """Points in every overlap of one slice's tiles: [(tile_a, q_a, tile_b, q_b)], q in tile pixels.

    T: {tile: tile -> aligned}; shapes: {tile: (height, width)}.
    """
    out = []
    tiles = sorted(T)
    box = {t: transforms.bbox(T[t], shapes[t][1], shapes[t][0]) for t in tiles}
    for i, ta in enumerate(tiles):
        for tb in tiles[i + 1:]:
            x0, y0 = np.maximum(box[ta][:2], box[tb][:2])
            x1, y1 = np.minimum(box[ta][2:], box[tb][2:])
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            n_x, n_y = SEAM_GRID if x1 - x0 >= y1 - y0 else SEAM_GRID[::-1]
            gx, gy = np.meshgrid(np.linspace(x0 + 1, x1 - 1, n_x), np.linspace(y0 + 1, y1 - 1, n_y))
            p = np.column_stack([gx.ravel(), gy.ravel()])
            qa = transforms.apply(transforms.invert(T[ta]), p)
            qb = transforms.apply(transforms.invert(T[tb]), p)
            inside = _inside(qa, shapes[ta]) & _inside(qb, shapes[tb])
            if inside.sum() >= 6:
                out.append((ta, qa[inside], tb, qb[inside]))
    return out


def _inside(q, shape, margin=2.0):
    h, w = shape
    return (q[:, 0] >= margin) & (q[:, 0] <= w - margin) & (q[:, 1] >= margin) & (q[:, 1] <= h - margin)


def solve_window(T0, shapes, matches, inside, a):
    """Affine per (z, tile) in ``inside``, from matches and seams; the rest of T0 stays fixed.

    T0: {(z, tile): tile -> aligned}, for the window and the slices around it; shapes: {(z, tile):
    (height, width)}; matches: dict of arrays za, ta, qa, zb, tb, qb (tile pixels) with at least one
    side inside; inside: set of z. Returns ({(z, tile): A}, report dict).
    """
    keys = sorted(k for k in T0 if k[0] in inside)
    idx = {k: i for i, k in enumerate(keys)}
    centre = {k: np.array(shapes[k][::-1], float) / 2 for k in T0}
    scale = {k: max(shapes[k]) / 2 for k in T0}

    # Cross-slice matches (robustly re-weighted) and seam points (always kept) as point pairs.
    ka = list(zip(matches["za"].tolist(), matches["ta"].tolist()))
    kb = list(zip(matches["zb"].tolist(), matches["tb"].tolist()))
    ok = np.array([a_ in T0 and b_ in T0 for a_, b_ in zip(ka, kb)], bool)
    ka = [k for k, o in zip(ka, ok) if o]
    kb = [k for k, o in zip(kb, ok) if o]
    qa, qb = matches["qa"][ok].astype(float), matches["qb"][ok].astype(float)
    n_match = len(ka)
    seams = []
    for z in sorted(inside):
        tz = {k[1]: T0[k] for k in keys if k[0] == z}
        for ta, pa, tb, pb in seam_points(tz, {t: shapes[(z, t)] for t in tz}):
            seams.append(((z, ta), pa, (z, tb), pb))
    for key_a, pa, key_b, pb in seams:
        ka += [key_a] * len(pa)
        kb += [key_b] * len(pb)
    if seams:
        qa = np.vstack([qa] + [s[1] for s in seams])
        qb = np.vstack([qb] + [s[3] for s in seams])
    w = np.r_[np.ones(n_match), np.full(len(ka) - n_match, SEAM_WEIGHT)]
    is_match = np.arange(len(ka)) < n_match

    ia = np.array([idx.get(k, -1) for k in ka])
    ib = np.array([idx.get(k, -1) for k in kb])
    Ha, ra0 = _basis_and_start(ka, qa, T0, centre, scale)
    Hb, rb0 = _basis_and_start(kb, qb, T0, centre, scale)
    r0 = ra0 - rb0   # mismatch with the starting transforms

    keep = np.ones(len(ka), bool)
    n = len(keys)
    for it in range(int(a["robust_iterations"]) + 1):
        if it:   # the first solve uses every match: before it, a layout change is off by tens of px
            comp = r[is_match].ravel()
            sigma = 1.4826 * np.median(np.abs(comp - np.median(comp))) if len(comp) else 0.0
            keep = ~is_match | (np.linalg.norm(r, axis=1) <= max(float(a["reject_px"]), 3 * sigma))
        D = _solve(n, ia, ib, Ha, Hb, r0, w * keep, float(a["local_regularization"]))
        r = r0 + _apply(D, ia, Ha) - _apply(D, ib, Hb)
        log.debug("local solve iteration %d: %d of %d points kept", it, int(keep.sum()), len(keep))

    out = {}
    for k, i in idx.items():
        dx, dy = D[i]   # corrections (u, v, 1) -> px, u = (q - centre) / scale
        L = np.vstack([dx[:2], dy[:2]]) / scale[k]
        t = np.array([dx[2], dy[2]]) - L @ centre[k]
        out[k] = T0[k] + np.column_stack([L, t])

    def stats(v):
        return (float(np.median(v)), float(np.percentile(v, 90))) if len(v) else (np.nan, np.nan)
    before = np.linalg.norm(r0, axis=1)
    after = np.linalg.norm(r, axis=1)
    rep = {"tile_slices": n, "points": int(n_match), "rejected_points": int((is_match & ~keep).sum())}
    # Over every match, including those the robust fit set aside: they are mismatch too.
    rep["median_before_px"], rep["p90_before_px"] = stats(before[is_match])
    rep["median_after_px"], rep["p90_after_px"] = stats(after[is_match])
    rep["seam_median_after_px"] = float(np.median(after[~is_match])) if (~is_match).any() else np.nan
    rep["seam_max_after_px"] = float(after[~is_match].max()) if (~is_match).any() else np.nan
    return out, rep, (before[:n_match], after[:n_match], keep[:n_match], ok)


def _basis_and_start(keys, q, T0, centre, scale):
    """Per point: basis (u, v, 1) of its tile's correction, and its position under T0."""
    c = np.array([centre[k] for k in keys]).reshape(-1, 2)
    s = np.array([scale[k] for k in keys], float)
    H = np.column_stack([(q - c) / s[:, None], np.ones(len(q))]) if len(q) else np.zeros((0, 3))
    A = np.array([T0[k] for k in keys]).reshape(-1, 2, 3)
    p = np.einsum("nij,nj->ni", A[:, :, :2], q) + A[:, :, 2] if len(q) else np.zeros((0, 2))
    return H, p


def _apply(D, i, H):
    """Correction (px) of each point: H . D[i] (zero for fixed tiles, i = -1)."""
    out = np.zeros((len(i), 2))
    f = i >= 0
    out[f] = np.einsum("nk,nak->na", H[f], D[i[f]])
    return out


def _solve(n, ia, ib, Ha, Hb, r0, w, reg):
    """Least squares over corrections D (n, 2 axes, 3) of r0 + Ha D[ia] - Hb D[ib] = 0, plus a
    pull reg * (point weight of the tile) toward D = 0 (the starting transforms)."""
    k = 3
    rows, cols, vals = [], [], []
    rhs = np.zeros((n * k, 2))

    def block(i, j, G):   # G (m, k, k) summed into block (i, j)
        r = i[:, None, None] * k + np.arange(k)[None, :, None]
        c = j[:, None, None] * k + np.arange(k)[None, None, :]
        rows.append(np.broadcast_to(r, G.shape).ravel())
        cols.append(np.broadcast_to(c, G.shape).ravel())
        vals.append(G.ravel())

    wa, wb = w[:, None, None] * Ha[:, :, None], w[:, None, None] * Hb[:, :, None]
    fa, fb = ia >= 0, ib >= 0
    block(ia[fa], ia[fa], wa[fa] * Ha[fa][:, None, :])
    block(ib[fb], ib[fb], wb[fb] * Hb[fb][:, None, :])
    both = fa & fb
    block(ia[both], ib[both], -wa[both] * Hb[both][:, None, :])
    block(ib[both], ia[both], -wb[both] * Ha[both][:, None, :])
    for ax in range(2):
        for j in range(k):
            np.add.at(rhs[:, ax], ia[fa] * k + j, -w[fa] * Ha[fa, j] * r0[fa, ax])
            np.add.at(rhs[:, ax], ib[fb] * k + j, w[fb] * Hb[fb, j] * r0[fb, ax])
    weight = np.bincount(ia[fa], w[fa], minlength=n) + np.bincount(ib[fb], w[fb], minlength=n)
    diag = np.repeat(reg * np.maximum(weight, 1.0), k)
    rows.append(np.arange(n * k))
    cols.append(np.arange(n * k))
    vals.append(diag)
    M = sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n * k, n * k)).tocsc()
    U = spsolve(M, rhs).reshape(n, k, 2)
    return U.transpose(0, 2, 1)   # (n, axis, k)


def solve(cfg, df, zs, mats, run, stitch, load_tile_matches):
    """Run every window; write align/local_tiles.csv and align/local_windows.csv."""
    a = cfg["align"]
    wins = windows(zs, run, a)
    rows, reports = [], []
    shapes = {(z, t): (int(h), int(w)) for z, t, h, w in zip(df["z"], df["tile"], df["height"], df["width"])}
    nb = int(a["neighbors"])
    for i0, i1, reason in wins:
        lo, hi = max(0, i0 - nb), min(len(zs) - 1, i1 + nb)
        inside = set(zs[i0:i1 + 1].tolist())
        around = zs[lo:hi + 1]
        pos = {int(z): i for i, z in enumerate(zs[lo:hi + 1], lo)}
        T0 = {k: transforms.compose(mats[pos[k[0]]], stitch[k]) for k in shapes if k[0] in pos}
        m = load_tile_matches(int(around[0]), int(around[-1]))
        sel = np.isin(m["za"], list(inside)) | np.isin(m["zb"], list(inside))
        m = {k: v[sel] for k, v in m.items()}
        out, rep, (before, after, keep, ok) = solve_window(T0, shapes, m, inside, a)
        rep.update(z_first=int(zs[i0]), z_last=int(zs[i1]), reason=reason)
        rep["change_median_before_px"] = rep["change_median_after_px"] = np.nan
        rep["change_p90_before_px"] = rep["change_p90_after_px"] = np.nan
        if "layout change" in reason:
            za, zb = m["za"][ok], m["zb"][ok]
            seg_of = dict(zip(zs.tolist(), run.tolist()))
            across = np.array([seg_of[x] != seg_of[y] for x, y in zip(za.tolist(), zb.tolist())], bool)
            if across.any():
                rep["change_median_before_px"] = float(np.median(before[across]))
                rep["change_p90_before_px"] = float(np.percentile(before[across], 90))
                rep["change_median_after_px"] = float(np.median(after[across]))
                rep["change_p90_after_px"] = float(np.percentile(after[across], 90))
        log.info("local tile solve z %d-%d (%s): %d tile slices, %d points; mismatch median %.2f -> %.2f px, "
                 "p90 %.2f -> %.2f px; across the change median %.2f -> %.2f px; seams moved median %.2f, max %.2f px",
                 rep["z_first"], rep["z_last"], reason, rep["tile_slices"], rep["points"],
                 rep["median_before_px"], rep["median_after_px"], rep["p90_before_px"], rep["p90_after_px"],
                 rep["change_median_before_px"], rep["change_median_after_px"],
                 rep["seam_median_after_px"], rep["seam_max_after_px"])
        reports.append(rep)
        rows += [({"z": int(z), "tile": t}, A) for (z, t), A in sorted(out.items())]
    tiles = transforms.to_frame(rows) if rows else pd.DataFrame(columns=["z", "tile", *transforms.COLUMNS])
    cli.atomic_write(step_dir(cfg, "align", "local_tiles.csv"), lambda tmp: tiles.to_csv(tmp, index=False))
    rep = pd.DataFrame(reports, columns=WINDOW_COLUMNS)
    cli.atomic_write(step_dir(cfg, "align", "local_windows.csv"), lambda tmp: rep.to_csv(tmp, index=False))
    return tiles, rep
