"""Stitch: tile layout within each slice from SIFT point matches and a least-squares solve.

``run`` (array over sampled slices) measures every tile pair of a sampled slice in two passes:
coarse SIFT on whole downsampled tiles finds the unknown overlap, then fine SIFT on only the
overlap strips (read at full resolution) gives the point matches. A least-squares solve over
all pairs gives tile -> montage per slice (stitch/samples/z{z:06d}.json). ``merge`` turns the
samples of each segment into a transform for every selected (z, tile), fixed or per slice.
"""

import json
import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from . import features, transforms
from .cli import atomic_write, base_parser, my_chunks, setup, task_info
from .config import step_dir
from .slices import StackCache, load_slices, z_values

log = logging.getLogger(__name__)

DEFAULTS = {
    "stitch": {
        "sample_every": 50,          # stitch every n-th selected z of a segment (plus its ends and seams)
        "coarse_factor": 8,          # downsampling of whole tiles for the overlap search
        "min_coarse_factor": 2,      # if tiles stay unconnected, retry the search at half the factor down to this
        "fine_factor": 2,            # downsampling of the full-res overlap strips for the final matches
        "fine_margin_px": 200,       # full-res margin added around the coarse overlap for fine matching
        # Per-tile model. affine_rigid (default) is Janelia's montage model (mpicbg/TrakEM2/render): each
        # tile's transform is (1 - lambda) * affine + lambda * rigid, fitted iteratively over the λ
        # schedule below. Also: translation | rigid | similarity | affine (one linear least-squares solve).
        "model": "affine_rigid",
        "lambdas": [1.0, 0.5, 0.1],  # affine_rigid: rigid weight per round, decreasing (1 = rigid)
        "mode": "auto",              # fixed (one transform per tile and segment) | per_slice | auto
        "fixed_tolerance_px": 2.0,   # auto -> fixed if no tile's sampled offset is further than this from its median
        "ratio": 0.8,                # Lowe ratio test
        "ransac_px": 3.0,            # RANSAC inlier distance, in pixels of the image being matched
        "min_inliers": 12,           # fewer RANSAC inliers -> pair rejected
        "max_points_per_pair": 200,  # inlier correspondences per pair used in the solve
        "max_features": 20000,       # SIFT keypoints kept per image (strongest)
    },
}

MODES = ("auto", "fixed", "per_slice")
# Settings that change a sample's result: run redoes samples made with other values, merge refuses them.
_RUN_KEYS = ("coarse_factor", "min_coarse_factor", "fine_factor", "fine_margin_px", "model", "lambdas", "ratio", "ransac_px", "min_inliers",
             "max_points_per_pair", "max_features")
_NPARAM = {"translation": 2, "rigid": 3, "similarity": 4, "affine": 6}
MODELS = (*_NPARAM, "affine_rigid")
# Point-match filtering (RANSAC) model per tile model.
_PAIR_MODEL = {"affine_rigid": "affine"}
_IDENTITY = {"translation": [0, 0], "rigid": [0, 0, 0], "similarity": [1, 0, 0, 0], "affine": [1, 0, 0, 0, 1, 0]}


def tile_key(tile):
    """Numeric sort key for "r-c" tile ids ("2-0" before "10-0")."""
    return tuple(int(v) for v in tile.split("-"))


def sample_z(slices, every):
    """Sampled z: per segment every ``every``-th selected z, the first and last, and the last
    selected z before and the first after each seam (so interpolation never spans a seam).

    ``slices`` must include excluded rows, so a seam on an excluded slice still counts.
    """
    seams = np.array(sorted(slices.loc[slices["seam"], "z"].unique()), int)
    out = []
    for _, g in slices[~slices["excluded"]].groupby("segment"):
        zs = z_values(g)
        pick = set(zs[::every]) | {zs[0], zs[-1]}
        for i in np.searchsorted(zs, seams):  # zs[i]: first selected z at or after the seam
            if 0 < i < len(zs):
                pick |= {zs[i - 1], zs[i]}
        out += pick
    return sorted(out)


# ----- pairwise matching ------------------------------------------------------------

def _detect(img, factor, opts, offset=(0, 0)):
    """SIFT on ``img`` downsampled by ``factor``; keypoints in full-res pixels plus ``offset``."""
    small = features.downsample(img, factor)
    xy, desc = features.detect(features.to_uint8(small), nfeatures=opts["max_features"])
    scale = np.array([img.shape[1] / small.shape[1], img.shape[0] / small.shape[0]])
    return (xy + 0.5) * scale - 0.5 + np.asarray(offset, float), desc


def _fit(ka, da, kb, db, model, threshold, opts):
    """RANSAC ``model`` mapping tile_b keypoints -> tile_a keypoints: (A, pa, pb inliers) or None."""
    ia, ib = features.match(da, db, opts["ratio"])
    A, inl = features.fit_model(kb[ib], ka[ia], model, threshold=threshold, min_inliers=opts["min_inliers"])
    return None if A is None else (A, ka[ia][inl], kb[ib][inl])


def _overlap(t, shape_a, shape_b):
    """(x0, y0, x1, y1) in tile_a pixels of tile_b placed at offset t in tile_a's frame."""
    (ha, wa), (hb, wb) = shape_a, shape_b
    return max(0.0, t[0]), max(0.0, t[1]), min(wa, t[0] + wb), min(ha, t[1] + hb)


def _plausible(t, shape_a, shape_b):
    """Tiles overlap, by less than half a tile in at least one direction (not a duplicate)."""
    x0, y0, x1, y1 = _overlap(t, shape_a, shape_b)
    ox, oy = x1 - x0, y1 - y0
    return ox > 0 and oy > 0 and min(ox / shape_a[1], oy / shape_a[0]) < 0.5


def _region(box, offset, shape, margin):
    """(rows, cols) slices of ``box`` shifted by -offset, grown by ``margin``, clipped to ``shape``."""
    x0, y0, x1, y1 = box
    h, w = shape
    r0, r1 = (int(np.clip(v, 0, h)) for v in (np.floor(y0 - offset[1] - margin), np.ceil(y1 - offset[1] + margin)))
    c0, c1 = (int(np.clip(v, 0, w)) for v in (np.floor(x0 - offset[0] - margin), np.ceil(x1 - offset[0] + margin)))
    return slice(r0, r1), slice(c0, c1)


def _rms(A, pa, pb):
    return float(np.sqrt(np.mean(features.residuals(A, pb, pa) ** 2)))


# ----- per-slice solve --------------------------------------------------------------

def _linear(model, p):
    """(J (N, 2, k), c (N, 2)) with model(params) applied to points p == J @ params + c.

    rigid is linearised in the angle (exact to ~theta^2 * tile size, negligible for stage error).
    """
    x, y = p[:, 0], p[:, 1]
    o, z = np.ones_like(x), np.zeros_like(x)
    J = {"translation": [[o, z], [z, o]],
         "rigid": [[-y, o, z], [x, z, o]],
         "similarity": [[x, -y, o, z], [y, x, z, o]],
         "affine": [[x, y, o, z, z, z], [z, z, z, x, y, o]]}[model]
    c = p if model in ("translation", "rigid") else np.zeros_like(p)
    return np.moveaxis(np.array(J), -1, 0), c


def _matrix(model, q):
    if model == "translation":
        return transforms.translation(*q)
    if model == "rigid":
        th, tx, ty = q
        return np.array([[np.cos(th), -np.sin(th), tx], [np.sin(th), np.cos(th), ty]])
    if model == "similarity":
        p, s, tx, ty = q
        return np.array([[p, -s, tx], [s, p, ty]])
    return np.asarray(q, float).reshape(2, 3)


def solve(pairs, tiles, reference, model, lambdas=(1.0, 0.5, 0.1)):
    """Tile -> montage transforms with ``reference`` fixed at identity: ({tile: A}, {pair: rms px}).

    pairs: {(tile_a, tile_b): (pa, pb)}, pa (N, 2) in tile_a pixels matching pb in tile_b pixels.
    """
    if model == "affine_rigid":
        return solve_affine_rigid(pairs, tiles, reference, lambdas)
    return _solve_linear(pairs, tiles, reference, model)


def solve_affine_rigid(pairs, tiles, reference, lambdas, max_rounds=5000, tol=1e-3):
    """Janelia's montage solve (mpicbg TileConfiguration with InterpolatedAffineModel2D).

    Starting from the rigid solution, tiles are updated one at a time: each is refitted to its
    matches' current positions in the montage as (1 - λ) * affine fit + λ * rigid fit, until no
    point moves by more than ``tol`` px; then the next, smaller λ of the schedule. The rigid
    part keeps a tile's affine from overfitting its thin overlap strips.
    """
    T = _solve_linear(pairs, tiles, reference, "rigid")[0]
    free = [t for t in tiles if t != reference]
    for lam in lambdas:
        for _ in range(max_rounds):
            moved = 0.0
            for t in free:
                src, dst = [], []
                for (ta, tb), (pa, pb) in pairs.items():
                    if ta == t:
                        src.append(pa)
                        dst.append(transforms.apply(T[tb], pb))
                    elif tb == t:
                        src.append(pb)
                        dst.append(transforms.apply(T[ta], pa))
                if not src:
                    continue
                S, D = np.vstack(src), np.vstack(dst)
                new = (1 - lam) * features.estimate("affine", S, D) + lam * features.estimate("rigid", S, D)
                moved = max(moved, float(np.abs(transforms.apply(new, S) - transforms.apply(T[t], S)).max()))
                T[t] = new
            if moved < tol:
                break
    return T, residuals(T, pairs)


def _solve_linear(pairs, tiles, reference, model):
    """Least-squares tile -> montage transforms with ``reference`` fixed at identity.

    pairs: {(tile_a, tile_b): (pa, pb)}, pa (N, 2) in tile_a pixels matching pb in tile_b pixels.
    Returns ({tile: A}, {pair: rms residual px}).
    """
    k = _NPARAM[model]
    free = [t for t in tiles if t != reference]
    col = {t: i * k for i, t in enumerate(free)}
    ref = np.array(_IDENTITY[model], float)
    T = {reference: _matrix(model, ref)}
    if free:
        rows, rhs = [], []
        for (ta, tb), (pa, pb) in pairs.items():
            # T_a(pa) = T_b(pb)  ->  Ja qa - Jb qb = cb - ca
            (Ja, ca), (Jb, cb) = _linear(model, pa), _linear(model, pb)
            M = np.zeros((len(pa), 2, len(free) * k))
            r = cb - ca
            for t, J, sign in ((ta, Ja, 1), (tb, Jb, -1)):
                if t == reference:
                    r = r - sign * (J @ ref)
                else:
                    M[:, :, col[t]:col[t] + k] += sign * J
            rows.append(M.reshape(-1, M.shape[-1]))
            rhs.append(r.ravel())
        q = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)[0]
        T.update({t: _matrix(model, q[col[t]:col[t] + k]) for t in free})
    return T, residuals(T, pairs)


def residuals(T, pairs):
    """Per-pair rms distance between T_a(pa) and T_b(pb) (loop-closure consistency)."""
    return {(ta, tb): float(np.sqrt(np.mean(np.sum(
        (transforms.apply(T[ta], pa) - transforms.apply(T[tb], pb)) ** 2, axis=1))))
        for (ta, tb), (pa, pb) in pairs.items()}


def _without(pairs, p):
    return {q: v for q, v in pairs.items() if q != p}


def loop_error(pairs, tiles, reference, model, lambdas=(1.0, 0.5, 0.1)):
    """Worst loop-closure error: over pairs that close a loop, the residual of the pair when
    only the other pairs are solved. (The full solve's residuals understate it: one bad pair's
    error is spread around its loop, e.g. a quarter of it per pair in a 2x2 ring.)"""
    err = 0.0
    for p in pairs:
        rest = _without(pairs, p)
        if _component(reference, rest) == set(tiles):
            err = max(err, residuals(solve(rest, tiles, reference, model, lambdas)[0], {p: pairs[p]})[p])
    return err


def solve_robust(pairs, tiles, reference, model, tol, lambdas=(1.0, 0.5, 0.1)):
    """``solve`` after removing pairs that break loop closure (error > tol) one at a time.

    Each round removes the pair whose removal leaves the smallest loop error, keeping all tiles
    connected. If several removals would each fix it, the bad pair is ambiguous (e.g. one 2x2
    ring without diagonal matches) and the slice fails. Returns (T or None, kept pairs, reason).
    """
    err = loop_error(pairs, tiles, reference, model, lambdas)
    while err > tol:
        options = []
        for p in pairs:
            rest = _without(pairs, p)
            if _component(reference, rest) == set(tiles):
                options.append((loop_error(rest, tiles, reference, model, lambdas), p))
        options.sort()
        if len(options) > 1 and options[1][0] <= tol:
            return None, pairs, (f"inconsistent pair offsets (loop closure error {err:.1f} px) and "
                                 "no single bad pair can be identified")
        err, drop = options[0]
        pairs = _without(pairs, drop)
    return solve(pairs, tiles, reference, model, lambdas)[0], pairs, ""


def _component(start, pairs):
    """Tiles connected to ``start`` through ``pairs`` ((tile_a, tile_b) keys)."""
    seen, todo = {start}, [start]
    while todo:
        t = todo.pop()
        for a, b in pairs:
            u = b if t == a else a if t == b else None
            if u is not None and u not in seen:
                seen.add(u)
                todo.append(u)
    return seen


def stitch_slice(rows, cache, opts, reference):
    """Measure every tile pair of one slice and solve tile -> montage.

    rows: the slices.csv rows of one z. Returns the sample record (JSON-ready); ``ok`` is False,
    with a ``reason``, if ``reference`` is missing, the good pairs don't connect every tile to it,
    or a pair that breaks loop closure can't be identified.
    """
    by_tile = {r["tile"]: r for _, r in rows.iterrows()}
    tiles = sorted(by_tile, key=tile_key)
    first = by_tile[tiles[0]]
    model, cf, ff = opts["model"], opts["coarse_factor"], opts["fine_factor"]
    rec = {"z": int(first["z"]), "timestamp": first["timestamp"].isoformat(), "segment": int(first["segment"]),
           "model": model, "reference": reference, "settings": {k: opts[k] for k in _RUN_KEYS},
           "ok": False, "reason": "", "tiles": {}, "pairs": []}
    if reference not in by_tile:  # no gauge shared with the segment's other samples
        rec["reason"] = f"reference tile {reference} missing"
        return rec
    shapes = {t: (int(r["height"]), int(r["width"])) for t, r in by_tile.items()}

    # A small overlap (~1% of a tile) is only a few pixels wide at coarse_factor 8 and may yield too few
    # matches on noisy images: retry the whole search at twice the resolution before giving up.
    while True:
        rec["pairs"], points = _match_pairs(cache, by_tile, tiles, shapes, cf, opts)
        missing = sorted(set(tiles) - _component(reference, points), key=tile_key)
        if not missing or cf <= opts["min_coarse_factor"]:
            break
        new_cf = max(opts["min_coarse_factor"], cf // 2)
        log.info("z %d: tiles %s not connected at coarse_factor %d: retrying at %d", rec["z"], missing, cf, new_cf)
        cf = new_cf
    rec["coarse_factor_used"] = cf
    if missing:
        rec["reason"] = f"too few matches: tiles {missing} not connected to {reference}"
        return rec
    T, used, rec["reason"] = solve_robust(points, tiles, reference, model, opts["ransac_px"] * ff,
                                          opts["lambdas"])
    if T is None:
        return rec
    final = residuals(T, points)
    for pair in rec["pairs"]:
        key = (pair["tile_a"], pair["tile_b"])
        if key in final:
            pair.update(solve_residual_px=final[key], used=key in used)
    rec.update(ok=True, tiles={t: T[t].tolist() for t in tiles})
    return rec


def _match_pairs(cache, by_tile, tiles, shapes, cf, opts):
    """Coarse search over every tile pair at ``cf``, then fine matches in each found overlap.

    Returns (pair records, {(tile_a, tile_b): (points in a, points in b)} for the good pairs).
    """
    model, ff, margin = opts["model"], opts["fine_factor"], opts["fine_margin_px"]
    # Coarse: whole tiles, every pair (filename r/c say nothing reliable about the layout).
    coarse = {t: _detect(cache.read(r), cf, opts) for t, r in by_tile.items()}
    pairs, points = [], {}
    for ta, tb in combinations(tiles, 2):
        fit = _fit(*coarse[ta], *coarse[tb], "translation", opts["ransac_px"] * cf, opts)
        if fit is None or not _plausible(fit[0][:, 2], shapes[ta], shapes[tb]):
            # Expected for diagonal neighbours with little or no overlap; recorded for diagnosis.
            pairs.append({"tile_a": ta, "tile_b": tb, "coarse_tx": None, "coarse_ty": None,
                          "coarse_inliers": 0 if fit is None else len(fit[1]), "A": None, "n_inliers": 0,
                          "residual_px": None, "solve_residual_px": None, "used": False,
                          "note": f"no coarse match at coarse_factor {cf}"})
            continue
        t = fit[0][:, 2]
        pair = {"tile_a": ta, "tile_b": tb, "coarse_tx": float(t[0]), "coarse_ty": float(t[1]),
                "coarse_inliers": len(fit[1]), "A": None, "n_inliers": 0, "residual_px": None,
                "solve_residual_px": None, "used": False, "note": ""}
        pairs.append(pair)
        # Fine: only the overlap (+ margin) of each tile, at full resolution.
        box = _overlap(t, shapes[ta], shapes[tb])
        ra, ca = _region(box, (0, 0), shapes[ta], margin)
        rb, cb = _region(box, t, shapes[tb], margin)
        ka, da = _detect(cache.read(by_tile[ta], ra, ca), ff, opts, (ca.start, ra.start))
        kb, db = _detect(cache.read(by_tile[tb], rb, cb), ff, opts, (cb.start, rb.start))
        fit = _fit(ka, da, kb, db, _PAIR_MODEL.get(model, model), opts["ransac_px"] * ff, opts)
        if fit is None:
            pair["note"] = "fine matching failed"
            continue
        A, pa, pb = fit
        centre = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        if np.linalg.norm(transforms.apply(A, centre - t)[0] - centre) > margin:
            pair["note"] = "fine offset disagrees with coarse"
            continue
        pair.update(A=A.tolist(), n_inliers=len(pa), residual_px=_rms(A, pa, pb))
        keep = np.unique(np.linspace(0, len(pa) - 1, min(len(pa), opts["max_points_per_pair"])).round().astype(int))
        points[(ta, tb)] = (pa[keep], pb[keep])
    return pairs, points


def _sample_path(cfg, z):
    return step_dir(cfg, "stitch", "samples", f"z{z:06d}.json")


def _load(cfg):
    """(selected non-excluded slices, sampled z, {z: (segment, reference tile)})."""
    allrows = load_slices(cfg, include_excluded=True)
    slices = allrows[~allrows["excluded"]]
    if slices.empty:
        raise SystemExit("stitch: no selected, non-excluded slices in check/slices.csv")
    refs = {int(s): min(g["tile"], key=tile_key) for s, g in slices.groupby("segment")}
    gauge = {int(z): (int(s), refs[int(s)]) for z, s in slices.groupby("z")["segment"].first().items()}
    return slices, sample_z(allrows, cfg["stitch"]["sample_every"]), gauge


def _stale(rec, opts, segment, reference):
    """Why a stored sample doesn't fit slices.csv and the config any more ('' if it does).

    The reference tile depends on the selection; samples solved with different references are in
    different frames and must not be mixed.
    """
    if (rec["segment"], rec["reference"]) != (segment, reference):
        return (f"made for segment {rec['segment']} with reference tile {rec['reference']}, "
                f"now segment {segment} with {reference}")
    changed = [k for k in _RUN_KEYS if rec["settings"].get(k) != opts[k]]
    return f"made with different stitch.{', stitch.'.join(changed)}" if changed else ""


def run(cfg, task_id, num_tasks, overwrite=False):
    opts = cfg["stitch"]
    if opts["model"] not in MODELS:
        raise ValueError(f"stitch.model must be one of {list(MODELS)}")
    slices, samples, gauge = _load(cfg)
    mine = my_chunks(samples, task_id, num_tasks)
    log.info("%d sampled slices, task %d/%d takes %d", len(samples), task_id, num_tasks, len(mine))
    by_z = dict(tuple(slices.groupby("z")))
    cache = StackCache(cfg["raw_dir"])
    for z in mine:
        path = _sample_path(cfg, z)
        if path.exists() and not overwrite:
            why = _stale(json.loads(path.read_text()), opts, *gauge[z])
            if not why:
                continue
            log.info("z %d: redoing sample (%s)", z, why)
        rec = stitch_slice(by_z[z], cache, opts, gauge[z][1])
        if rec["ok"]:
            used = [p for p in rec["pairs"] if p["used"]]
            log.info("z %d: %d pairs, max solve residual %.2f px", z, len(used),
                     max((p["solve_residual_px"] for p in used), default=0.0))
        else:
            log.warning("z %d: %s; sample skipped", z, rec["reason"])
        atomic_write(path, lambda p: Path(p).write_text(json.dumps(rec, indent=1)))


# ----- merge ------------------------------------------------------------------------

def _grid_index(values, size):
    """Cluster 1-D tile positions: a new grid column starts where sorted positions jump by > size / 2."""
    values = np.asarray(values, float)
    order = np.argsort(values)
    idx = np.empty(len(values), int)
    idx[order] = np.concatenate([[0], np.cumsum(np.diff(values[order]) > size / 2)])
    return idx


def _row_axis(info):
    """'y' or 'x': which grid axis the filename row number follows (None if neither)."""
    def follows(src, dst):
        return all(len({i[dst] for i in info.values() if i[src] == v}) == 1 for v in {i[src] for i in info.values()})
    for axis, (r, c) in (("y", ("grid_y", "grid_x")), ("x", ("grid_x", "grid_y"))):
        if follows("tile_row", r) and follows("tile_col", c):
            return axis
    return None


def _layout(rows, origins, shape):
    """Grid position of each tile from its montage origin, filename-row axis, neighbour overlaps."""
    h, w = shape
    tiles = list(origins)
    xy = np.array([origins[t] for t in tiles])
    gx, gy = _grid_index(xy[:, 0], w), _grid_index(xy[:, 1], h)
    rc = rows.drop_duplicates("tile").set_index("tile")
    info = {t: {"grid_x": int(gx[i]), "grid_y": int(gy[i]), "tile_row": int(rc.at[t, "tile_row"]),
                "tile_col": int(rc.at[t, "tile_col"]), "x": round(float(xy[i, 0]), 2), "y": round(float(xy[i, 1]), 2)}
            for i, t in enumerate(tiles)}
    overlap = {"x": [], "y": []}
    for a, b in combinations(tiles, 2):
        ia, ib = info[a], info[b]
        if ia["grid_y"] == ib["grid_y"] and abs(ia["grid_x"] - ib["grid_x"]) == 1:
            overlap["x"].append(w - abs(ia["x"] - ib["x"]))
        if ia["grid_x"] == ib["grid_x"] and abs(ia["grid_y"] - ib["grid_y"]) == 1:
            overlap["y"].append(h - abs(ia["y"] - ib["y"]))
    return {"grid_shape": [int(gy.max()) + 1, int(gx.max()) + 1], "row_axis": _row_axis(info),
            "overlap_px": {k: round(float(np.median(v)), 1) if v else None for k, v in overlap.items()},
            "tiles": {t: {k: v for k, v in i.items() if k not in ("tile_row", "tile_col")} for t, i in info.items()}}


def merge_segment(rows, recs, opts):
    """Transforms for every selected (z, tile) of one segment, from its sampled-slice records.

    Returns (tiles frame, layout dict, sample points frame for the plot).
    """
    seg = int(rows["segment"].iloc[0])
    zs = np.array(z_values(rows))
    good = [r for r in recs if r["ok"]]
    if not good:
        why = "; ".join(f"z {r['z']}: {r['reason']}" for r in recs[:3])
        raise SystemExit(f"stitch merge: segment {seg} (z {zs[0]}-{zs[-1]}) has no usable sampled slice "
                         f"({len(recs)} sampled: {why}). See stitch/samples/*.json; try a smaller "
                         "stitch.sample_every or stitch.min_inliers, or exclude the bad slices.")
    tiles = sorted(rows["tile"].unique(), key=tile_key)
    samples = {}  # tile -> (sample z, params (n, 6))
    for t in tiles:
        have = [r for r in good if t in r["tiles"]]
        if not have:
            raise SystemExit(f"stitch merge: segment {seg}: tile {t} is in no usable sampled slice")
        samples[t] = (np.array([r["z"] for r in have]), np.array([np.ravel(r["tiles"][t]) for r in have]))
    dev = max(float(np.linalg.norm(p[:, [2, 5]] - np.median(p[:, [2, 5]], axis=0), axis=1).max())
              for _, p in samples.values())
    mode = opts["mode"] if opts["mode"] != "auto" else ("fixed" if dev <= opts["fixed_tolerance_px"] else "per_slice")

    params = {}  # tile -> (len(zs), 6)
    for t, (sz, p) in samples.items():
        if mode == "fixed":
            params[t] = np.repeat(np.median(p, axis=0)[None], len(zs), axis=0)
        else:
            p = ndimage.median_filter(p, size=(3, 1), mode="nearest")
            params[t] = np.column_stack([np.interp(zs, sz, v) for v in p.T])

    out = rows[["z", "timestamp", "tile", "height", "width"]].copy()
    mats = np.empty((len(out), 6))
    for t in tiles:
        sel = (out["tile"] == t).to_numpy()
        mats[sel] = params[t][np.searchsorted(zs, out.loc[sel, "z"])]
    # Shift the segment so the montage min corner over all its slices is (0, 0).
    hw = out[["height", "width"]].to_numpy(float)
    xs = [mats[:, 0] * cx * hw[:, 1] + mats[:, 1] * cy * hw[:, 0] + mats[:, 2] for cx in (0, 1) for cy in (0, 1)]
    ys = [mats[:, 3] * cx * hw[:, 1] + mats[:, 4] * cy * hw[:, 0] + mats[:, 5] for cx in (0, 1) for cy in (0, 1)]
    shift = np.array([np.min(xs), np.min(ys)])
    mats[:, [2, 5]] -= shift
    out[transforms.COLUMNS] = mats
    out["segment"] = seg
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    origins = {t: np.median(p[:, [2, 5]], axis=0) - shift for t, (_, p) in samples.items()}
    shape = (int(rows["height"].iloc[0]), int(rows["width"].iloc[0]))
    layout = {"segment": seg, "z_first": int(zs[0]), "z_last": int(zs[-1]), "tile_shape": list(shape),
              "reference_tile": good[0]["reference"], "model": opts["model"], "mode": mode,
              "max_deviation_px": round(dev, 2), "n_samples": len(recs), "n_good_samples": len(good),
              "failed_samples": [r["z"] for r in recs if not r["ok"]],
              **_layout(rows, origins, shape)}
    pts = pd.DataFrame([{"z": z, "tile": t, "tx": v[2] - shift[0], "ty": v[5] - shift[1]}
                        for t, (sz, p) in samples.items() for z, v in zip(sz, p)])
    log.info("segment %d: %s mode (max deviation %.2f px), %d/%d good samples, grid %s, filename row = %s",
             seg, mode, dev, len(good), len(recs), layout["grid_shape"], layout["row_axis"])
    return out.drop(columns=["height", "width"]), layout, pts


def _pair_rows(rec):
    for p in rec["pairs"]:
        if p["A"] is not None:
            yield {"z": rec["z"], "segment": rec["segment"], "tile_a": p["tile_a"], "tile_b": p["tile_b"],
                   "model": rec["model"], **transforms.to_row(p["A"]), "n_inliers": p["n_inliers"],
                   "residual_px": p["residual_px"], "solve_residual_px": p["solve_residual_px"], "used": p["used"]}


def merge(cfg):
    opts = cfg["stitch"]
    if opts["mode"] not in MODES:
        raise ValueError(f"stitch.mode must be one of {MODES}")
    slices, samples, gauge = _load(cfg)
    recs, missing, stale = {}, [], []
    for z in samples:
        path = _sample_path(cfg, z)
        if not path.exists():
            missing.append(z)
            continue
        recs[z] = json.loads(path.read_text())
        why = _stale(recs[z], opts, *gauge[z])
        if why:
            stale.append(f"z {z}: {why}")
    if missing:
        raise SystemExit(f"stitch merge: {len(missing)} of {len(samples)} sampled slices have no result "
                         f"(e.g. z {missing[:5]}); run 'stitch run' (all array tasks) first")
    if stale:
        raise SystemExit(f"stitch merge: {len(stale)} sampled slices are out of date ({stale[0]}); "
                         "run 'stitch run' (all array tasks) first, it redoes them")

    tiles, layouts, points = [], {}, []
    for seg, rows in slices.groupby("segment"):
        seg_recs = [recs[z] for z in sorted(set(rows["z"]) & set(recs))]
        out, layouts[str(seg)], pts = merge_segment(rows, seg_recs, opts)
        tiles.append(out)
        points.append(pts.assign(segment=seg))
    tiles = pd.concat(tiles, ignore_index=True)
    pairs = pd.DataFrame([row for z in samples for row in _pair_rows(recs[z])],
                         columns=["z", "segment", "tile_a", "tile_b", "model", *transforms.COLUMNS,
                                  "n_inliers", "residual_px", "solve_residual_px", "used"])
    failed = [z for z in samples if not recs[z]["ok"]]

    atomic_write(step_dir(cfg, "stitch", "tiles.csv"), lambda p: tiles.to_csv(p, index=False))
    atomic_write(step_dir(cfg, "stitch", "pairs.csv"), lambda p: pairs.to_csv(p, index=False))
    atomic_write(step_dir(cfg, "stitch", "layout.json"),
                 lambda p: Path(p).write_text(json.dumps({"segments": layouts}, indent=1)))
    plot(step_dir(cfg, "stitch", "stitch.png"), tiles, pd.concat(points, ignore_index=True), failed)
    log.info("wrote %d tile transforms for %d slices, %d pair measurements (%d failed samples)",
             len(tiles), tiles["z"].nunique(), len(pairs), len(failed))


def plot(path, tiles, points, failed):
    """Per tile, x/y offset relative to its segment median vs z; sampled slices as dots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ink, muted, grid, blue, orange, red = "#52514e", "#898781", "#e1e0d9", "#2a78d6", "#eb6834", "#d03b3b"
    names = sorted(tiles["tile"].unique(), key=tile_key)
    med = tiles.groupby(["segment", "tile"])[["tx", "ty"]].median()
    fig, axes = plt.subplots(len(names), 1, sharex=True, squeeze=False, figsize=(10, 0.6 + 1.3 * len(names)))
    seg_starts = tiles.groupby("segment")["z"].min().sort_values().to_numpy()[1:]
    for ax, t in zip(axes[:, 0], names):
        tz = set(tiles.loc[tiles["tile"] == t, "z"])
        for frame, style in ((tiles, "line"), (points, "dots")):
            d = frame[frame["tile"] == t].sort_values("z")
            m = med.loc[list(zip(d["segment"], d["tile"]))].to_numpy()
            for i, (col, color) in enumerate((("tx", blue), ("ty", orange))):
                if style == "line":
                    # One line per segment so segment changes don't draw connecting jumps.
                    for _, s in d.assign(v=d[col] - m[:, i]).groupby("segment"):
                        ax.plot(s["z"], s["v"], color=color, lw=1.5)
                else:
                    ax.plot(d["z"], d[col] - m[:, i], "o", ms=3.5, color=color, mec="white", mew=0.5)
        for z in seg_starts:
            ax.axvline(z, color=muted, lw=0.8, ls="--")
        fz = [z for z in failed if z in tz]
        ax.plot(fz, np.zeros(len(fz)), "x", color=red, ms=5)
        lo, hi = ax.get_ylim()  # at least +-1 px so sub-pixel jitter doesn't look like movement
        ax.set_ylim(min(lo, -1), max(hi, 1))
        ax.set_ylabel(f"tile {t}\npx", color=ink, fontsize=8)
        ax.grid(color=grid, lw=0.5)
        ax.tick_params(colors=muted, labelsize=8)
        for s in ax.spines.values():
            s.set_color(grid)
    axes[-1, 0].set_xlabel("z", color=ink)
    handles = [Line2D([], [], color=blue, lw=1.5, label="x offset"),
               Line2D([], [], color=orange, lw=1.5, label="y offset"),
               Line2D([], [], color=muted, marker="o", ls="", label="sampled slice"),
               Line2D([], [], color=red, marker="x", ls="", label="failed sample"),
               Line2D([], [], color=muted, ls="--", label="segment change")]
    axes[0, 0].legend(handles=handles, ncol=5, fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(0, 1))
    fig.suptitle("Tile offsets relative to each tile's segment median", color=ink, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    atomic_write(path, lambda p: fig.savefig(p, dpi=110))
    plt.close(fig)


def main(argv=None):
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("command", choices=["run", "merge"])
    args = p.parse_args(argv)
    cfg = setup(args, DEFAULTS)
    if args.command == "run":
        run(cfg, *task_info(args), overwrite=args.overwrite)
    else:
        merge(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
