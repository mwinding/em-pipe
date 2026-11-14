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
window. Matches across an elastic interface (pipeline/elastic.py) are left out: its field corrects
that mismatch. So a window of one slice right after such an interface (e.g. the first slice after a
pause with one tile scanned before the beam settled) fits that slice's tiles to the slices after it. Windows: ``align.local_half_width`` slices either side of every tile layout change
(``align.local_at_layout_changes``) and the z ranges in ``align.local_windows``. Writes
``align/local_tiles.csv`` (tile -> aligned transform for every (z, tile) in a window, used by
render instead of align ∘ stitch) and ``align/local_windows.csv`` (mismatch before and after).

With ``align.local_warp_px`` every tile in a window also gets a smooth displacement field on a grid
of that spacing over the tile (bilinear, as pipeline/elastic.py's fields; differences between
neighbouring grid values penalised by ``align.local_warp_smoothness`` x points per grid cell), added
after its affine: for a distortion within one tile that no affine captures, e.g. a tile scanned
before the beam settled after a pause. The fields go to ``align/local_warps.json`` (tile pixels ->
displacement in aligned px), which render applies.

Seams (``align.local_seams``): ``current`` holds each overlap where the tiles meet now (stitch and
align). ``matched`` matches features in each overlap of the window's slices instead (read from the
raw tiles, as the stitch's fine step) and holds those correspondences: right also where the stitch
could only compromise, e.g. a rigid tile placed against a stretched one. Overlaps with too few
matches (resin) keep the current seam.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

from . import cli, features, transforms
from .config import deep_merge, step_dir, step_path
from .slices import StackCache
from .elastic import _bilinear, field_at

log = logging.getLogger(__name__)

SEAM_GRID = (25, 6)  # seam points along and across each overlap
WINDOW_COLUMNS = ["z_first", "z_last", "reason", "tile_slices", "points", "median_before_px",
                  "median_after_px", "p90_before_px", "p90_after_px", "change_median_before_px",
                  "change_median_after_px", "change_p90_before_px", "change_p90_after_px",
                  "seam_median_after_px", "seam_max_after_px", "rejected_points", "warp_max_px"]


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


def matched_seams(read, T, shapes, opts):
    """Seam correspondences of one slice's tiles from their image content: [(tile_a, q_a, tile_b, q_b)].

    read(tile, rows, cols): the raw tile region; T: {tile: tile -> aligned}, the current transforms
    (they say where the overlaps are); shapes: {tile: (height, width)}; opts: the stitch's matching
    settings (fine_factor, fine_margin_px, ratio, ransac_px, min_inliers, max_features,
    max_points_per_pair) plus chunk_px. In each overlap features are matched as in the stitch's fine
    step, then kept by a translation RANSAC per chunk_px along the overlap: the matches follow a
    distortion along the seam (a stretched tile) that one rigid fit of the whole overlap would cut
    down to a band. Overlaps with fewer than min_inliers matches are left out.
    """
    from .stitch import _detect   # stitch's detector: same downsampling and keypoint convention
    out = []
    tiles = sorted(T)
    box = {t: transforms.bbox(T[t], shapes[t][1], shapes[t][0]) for t in tiles}
    ff, margin = int(opts["fine_factor"]), float(opts["fine_margin_px"])
    threshold, min_in = float(opts["ransac_px"]) * ff, int(opts["min_inliers"])
    for i, ta in enumerate(tiles):
        for tb in tiles[i + 1:]:
            x0, y0 = np.maximum(box[ta][:2], box[tb][:2])
            x1, y1 = np.minimum(box[ta][2:], box[tb][2:])
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], float)
            kp = {}
            for t in (ta, tb):
                q = transforms.apply(transforms.invert(T[t]), corners)
                h, w = shapes[t]
                r0, r1 = (int(np.clip(v, 0, h)) for v in (np.floor(q[:, 1].min() - margin), np.ceil(q[:, 1].max() + margin)))
                c0, c1 = (int(np.clip(v, 0, w)) for v in (np.floor(q[:, 0].min() - margin), np.ceil(q[:, 0].max() + margin)))
                if r1 - r0 < 8 * ff or c1 - c0 < 8 * ff:
                    break
                kp[t] = _detect(np.asarray(read(t, slice(r0, r1), slice(c0, c1))), ff, opts, (c0, r0))
            if len(kp) < 2:
                continue
            ia, ib = features.match(kp[ta][1], kp[tb][1], opts["ratio"])
            qa, qb = kp[ta][0][ia].astype(float), kp[tb][0][ib].astype(float)
            pa, pb = transforms.apply(T[ta], qa), transforms.apply(T[tb], qb)
            near = np.linalg.norm(pa - pb, axis=1) < margin   # where the current transforms expect them
            qa, qb, pa, pb = qa[near], qb[near], pa[near], pb[near]
            if len(qa) < min_in:
                continue
            along = 0 if x1 - x0 >= y1 - y0 else 1
            chunk = np.floor((pa[:, along] - pa[:, along].min()) / float(opts["chunk_px"])).astype(int)
            keep = np.zeros(len(qa), bool)
            for c in np.unique(chunk):
                sel = np.flatnonzero(chunk == c)
                A, inl = features.fit_model(pb[sel], pa[sel], "translation", threshold=threshold,
                                            min_inliers=max(4, min_in // 3))
                if A is not None:
                    keep[sel[inl]] = True
            if keep.sum() < min_in:
                continue
            qa, qb = qa[keep], qb[keep]
            pick = np.unique(np.linspace(0, len(qa) - 1, min(len(qa), int(opts["max_points_per_pair"]))).round().astype(int))
            out.append((ta, qa[pick], tb, qb[pick]))
    return out


def seam_options(cfg):
    """The stitch's matching settings (config over its defaults) plus align.local_seam_chunk_px."""
    from .stitch import DEFAULTS as STITCH
    st = deep_merge(STITCH, {"stitch": cfg.get("stitch") or {}})["stitch"]
    return {**st, "chunk_px": float(cfg["align"]["local_seam_chunk_px"])}


def _inside(q, shape, margin=2.0):
    h, w = shape
    return (q[:, 0] >= margin) & (q[:, 0] <= w - margin) & (q[:, 1] >= margin) & (q[:, 1] <= h - margin)


def solve_window(T0, shapes, matches, inside, a, matched=None):
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
        found = {(ta, tb): (pa, pb) for ta, pa, tb, pb in (matched or {}).get(z, [])}
        for ta, pa, tb, pb in seam_points(tz, {t: shapes[(z, t)] for t in tz}):
            pa, pb = found.pop((ta, tb), (pa, pb))   # matched in the image where it could be
            seams.append(((z, ta), pa, (z, tb), pb))
        seams += [((z, ta), pa, (z, tb), pb) for (ta, tb), (pa, pb) in found.items()]
    for key_a, pa, key_b, pb in seams:
        ka += [key_a] * len(pa)
        kb += [key_b] * len(pb)
    if seams:
        qa = np.vstack([qa] + [s[1] for s in seams])
        qb = np.vstack([qb] + [s[3] for s in seams])
    w = np.r_[np.ones(n_match), np.full(len(ka) - n_match, float(a["local_seam_weight"]))]
    is_match = np.arange(len(ka)) < n_match

    ia = np.array([idx.get(k, -1) for k in ka])
    ib = np.array([idx.get(k, -1) for k in kb])
    Ha, ra0 = _basis_and_start(ka, qa, T0, centre, scale)
    Hb, rb0 = _basis_and_start(kb, qb, T0, centre, scale)
    r0 = ra0 - rb0   # mismatch with the starting transforms

    keep = np.ones(len(ka), bool)
    n = len(keys)
    spacing = a.get("local_warp_px")
    if spacing:
        grids = [_tile_grid(shapes[k], float(spacing)) for k in keys]
        A = _design(n, grids, float(spacing), ia, ib, Ha, Hb, qa, qb)
    for it in range(int(a["robust_iterations"]) + 1):
        if it:   # the first solve uses every match: before it, a layout change is off by tens of px
            comp = r[is_match].ravel()
            sigma = 1.4826 * np.median(np.abs(comp - np.median(comp))) if len(comp) else 0.0
            keep = ~is_match | (np.linalg.norm(r, axis=1) <= max(float(a["reject_px"]), 3 * sigma))
        if spacing:
            U = _solve_warp(A, grids, ia, ib, is_match, r0, w * keep, float(a["local_regularization"]),
                            float(a["local_warp_smoothness"]))
            D = U[:3 * n].reshape(n, 3, 2).transpose(0, 2, 1)
            r = r0 + A @ U
        else:
            D = _solve(n, ia, ib, Ha, Hb, r0, w * keep, float(a["local_regularization"]))
            r = r0 + _apply(D, ia, Ha) - _apply(D, ib, Hb)
        log.debug("local solve iteration %d: %d of %d points kept", it, int(keep.sum()), len(keep))

    warps = {}
    if spacing:
        off = 3 * n
        for k, (nx, ny) in zip(keys, grids):
            g = U[off:off + nx * ny].reshape(ny, nx, 2)
            off += nx * ny
            warps[k] = {"x0": 0.0, "y0": 0.0, "spacing": float(spacing), "g": g.tolist()}

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
    rep["warp_max_px"] = (max(float(np.linalg.norm(np.asarray(f["g"]), axis=2).max()) for f in warps.values())
                          if warps else np.nan)
    return out, rep, (before[:n_match], after[:n_match], keep[:n_match], ok), warps


def _tile_grid(shape, spacing):
    """(nx, ny) of a warp grid over a tile of ``shape`` (height, width): nodes 0, spacing, ... past each edge."""
    h, w = shape
    return int(np.ceil(w / spacing)) + 1, int(np.ceil(h / spacing)) + 1


def _design(n, grids, spacing, ia, ib, Ha, Hb, qa, qb):
    """Sparse (points, unknowns) matrix of the corrections' effect on each point pair's mismatch:
    + side a's tile, - side b's (fixed tiles, index -1, contribute nothing). Unknowns: per tile its
    affine correction (basis u, v, 1), then per tile its warp grid (bilinear weights in tile px)."""
    off = np.r_[0, np.cumsum([nx * ny for nx, ny in grids])] + 3 * n
    rows, cols, vals = [np.zeros(0, int)], [np.zeros(0, int)], [np.zeros(0)]
    for sign, idx, H, q in ((1.0, ia, Ha, qa), (-1.0, ib, Hb, qb)):
        for i in np.unique(idx[idx >= 0]):
            pts = np.flatnonzero(idx == i)
            rows.append(np.repeat(pts, 3))
            cols.append(np.tile(3 * i + np.arange(3), len(pts)))
            vals.append(sign * H[pts].ravel())
            nx, ny = grids[i]
            W = _bilinear(q[pts], 0.0, 0.0, spacing, nx, ny).tocoo()
            rows.append(pts[W.row])
            cols.append(off[i] + W.col)
            vals.append(sign * W.data)
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(len(ia), int(off[-1])))


def _solve_warp(A, grids, ia, ib, is_match, r0, w, reg, smoothness):
    """Least squares over the unknowns of ``_design`` (n tiles' affine corrections, then their warp
    grids), both axes: r0 + A U = 0 at the points (weights w), a membrane penalty smoothness x
    (matched points per grid cell) on each tile's neighbouring grid values, and a pull reg x (point
    weight of the tile) toward 0, as the affine solve. Returns U (unknowns, 2)."""
    n = len(grids)
    off = np.r_[0, np.cumsum([nx * ny for nx, ny in grids])] + 3 * n
    weight = (np.bincount(ia[ia >= 0], w[ia >= 0], minlength=n) + np.bincount(ib[ib >= 0], w[ib >= 0], minlength=n))
    matched = (np.bincount(ia[(ia >= 0) & is_match], minlength=n) + np.bincount(ib[(ib >= 0) & is_match], minlength=n))
    # A grid value sees only the points around it: its pull scales with their weight, not the tile's.
    diag = reg * np.maximum(np.asarray(abs(A).T @ w).ravel(), 1.0)
    pen = []
    for i, (nx, ny) in enumerate(grids):
        diag[3 * i:3 * i + 3] = reg * max(weight[i], 1.0)
        idx = off[i] + np.arange(nx * ny).reshape(ny, nx)
        pairs = np.r_[np.column_stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()]),
                      np.column_stack([idx[:-1, :].ravel(), idx[1:, :].ravel()])]
        lam = smoothness * max(matched[i] / ((nx - 1) * (ny - 1)), 1.0)
        m = len(pairs)
        pen.append(sparse.csr_matrix((np.sqrt(lam) * np.r_[np.ones(m), -np.ones(m)],
                                      (np.r_[np.arange(m), np.arange(m)], pairs.T.ravel())), shape=(m, A.shape[1])))
    P = sparse.vstack(pen) if pen else sparse.csr_matrix((0, A.shape[1]))
    Aw = A.multiply(w[:, None]).tocsr()
    M = (A.T @ Aw + P.T @ P + sparse.diags(diag)).tocsc()
    return np.asarray(spsolve(M, -(Aw.T @ r0))).reshape(-1, 2)


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


def crosses(za, zb, zs, cuts):
    """True for matches between slices on different sides of an interface; ``cuts``: positions in
    ``zs`` of the first slice after each."""
    if not len(cuts):
        return np.zeros(len(za), bool)
    lo = np.searchsorted(zs, np.minimum(za, zb))[:, None]
    hi = np.searchsorted(zs, np.maximum(za, zb))[:, None]
    c = np.asarray(cuts)[None, :]
    return ((lo < c) & (hi >= c)).any(axis=1)


def solve(cfg, df, zs, mats, run, stitch, load_tile_matches, cuts=()):
    """Run every window; write align/local_tiles.csv and align/local_windows.csv.

    ``cuts``: the elastic interfaces (positions in zs of the first slice after each). Matches across
    them are left out: the field corrects that mismatch, fitted to the transforms before this solve,
    and render applies it on top of these tiles, so tiles moved here as well would be corrected twice.
    """
    a = cfg["align"]
    wins = windows(zs, run, a)
    rows, reports, all_warps = [], [], {}
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
        cut = sel & crosses(m["za"], m["zb"], zs, cuts)
        if cut.any():
            log.info("local tile solve z %d-%d: %d matches across elastic interfaces left out", int(zs[i0]),
                     int(zs[i1]), int(cut.sum()))
        m = {k: v[sel & ~cut] for k, v in m.items()}
        matched = None
        if a["local_seams"] == "matched":
            opts, cache, matched = seam_options(cfg), StackCache(cfg["raw_dir"]), {}
            for z in sorted(inside):
                rz = {r["tile"]: r for r in df[df["z"] == z].to_dict("records")}
                tz = {t: T0[(z, t)] for t in rz if (z, t) in T0}
                matched[z] = matched_seams(lambda t, rr, cc, rz=rz: cache.read(rz[t], rr, cc), tz,
                                           {t: shapes[(z, t)] for t in tz}, opts)
                log.info("local tile solve z %d: seams matched in the image: %s", z,
                         ", ".join(f"{ta}|{tb} {len(qa)}" for ta, qa, tb, _ in matched[z]) or "none")
        out, rep, (before, after, keep, ok), warps = solve_window(T0, shapes, m, inside, a, matched)
        all_warps.update(warps)
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
        if warps:
            log.info("local tile solve z %d-%d: warp grids %g px, largest displacement %.1f px", rep["z_first"],
                     rep["z_last"], float(a["local_warp_px"]), rep["warp_max_px"])
        reports.append(rep)
        rows += [({"z": int(z), "tile": t}, A) for (z, t), A in sorted(out.items())]
    tiles = transforms.to_frame(rows) if rows else pd.DataFrame(columns=["z", "tile", *transforms.COLUMNS])
    cli.atomic_write(step_dir(cfg, "align", "local_tiles.csv"), lambda tmp: tiles.to_csv(tmp, index=False))
    rep = pd.DataFrame(reports, columns=WINDOW_COLUMNS)
    cli.atomic_write(step_dir(cfg, "align", "local_windows.csv"), lambda tmp: rep.to_csv(tmp, index=False))
    data = {f"{z}/{t}": f for (z, t), f in sorted(all_warps.items())}
    cli.atomic_write(step_dir(cfg, "align", "local_warps.json"), lambda tmp: Path(tmp).write_text(json.dumps(data)))
    return tiles, rep


def load_warps(cfg):
    """align/local_warps.json as {"z/tile": field} (tile px -> displacement in aligned px), or None if
    there is none or it is empty."""
    path = step_path(cfg, "align", "local_warps.json")
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data or None


def warp_at(field, q):
    """Displacement (n, 2) in aligned px of a tile's warp at tile pixels q (n, 2)."""
    return field_at(field, q)
