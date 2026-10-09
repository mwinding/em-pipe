"""Slice-to-slice drift correction: SIFT point matches between nearby slices, one global solve.

``run`` (array over z-chunks) matches every selected slice to its next ``align.neighbors``
selected slices, tile by tile, and saves the RANSAC inliers in montage coordinates.
``solve`` finds one montage -> aligned transform per slice that agrees best with all matches
at once (sparse least squares with outlier rejection), as in Janelia's render pipeline.
Solving globally over several neighbours avoids the random-walk error of chaining adjacent
slices.

Chunk files also keep the points in tile pixels (``qa``, ``qb`` with tile indices ``ta``, ``tb``
into ``tiles``); ``solve`` maps those with the current stitch/tiles.csv, so re-running stitch
never requires re-matching.
"""

import json
import logging
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage import gaussian_filter1d
from scipy.sparse.linalg import spsolve

from . import cli, features, transforms
from .config import qc_path, step_dir, step_path
from .slices import StackCache, load_slices, z_values

log = logging.getLogger(__name__)

DEFAULTS = {
    "align": {
        "scale": 0.25,                 # detect features on tiles downsampled to this fraction
        "neighbors": 4,                # match each slice to this many following selected slices
        "chunk_slices": 100,           # global z per array-task chunk (bounds at multiples of this)
        "max_features": 8000,          # SIFT features per tile (at the detection scale)
        "ratio": 0.8,                  # Lowe ratio test
        "ransac_model": "rigid",       # translation | rigid | similarity | affine, per tile pair
        "ransac_px": 2.0,              # RANSAC inlier threshold in detection-scale pixels
        "min_inliers": 12,             # tile pairs with fewer RANSAC inliers are dropped
        "max_points_per_pair": 100,    # inliers kept per (slice pair, tile), spread over the tile
        "model": "translation",        # per-slice transform: translation | affine
        "affine_regularization": 0.1,  # affine: pull of the linear part toward identity, relative to the data
        "robust_iterations": 3,        # rounds of outlier rejection + re-solve
        "reject_px": 4.0,              # never reject points whose residual is below this (full-res px)
        "anchor": "first",             # first (first selected slice = identity) | mean (mean translation 0)
        "remove_trend": "none",        # none | linear | lowpass: remove slow drift (steps at seams are kept)
        "trend_window_slices": 2000,   # lowpass: Gaussian FWHM in slices
    }
}

# Weight of the prior "consecutive slices have equal transforms", relative to the median weight
# of a matched slice pair: negligible where there are matches, but defines slices without any.
SMOOTHNESS = 1e-3
# A pair is rejected as a whole when less than this fraction of its points survive rejection.
MIN_PAIR_FRACTION = 0.5
OPTIONS = {"model": ("translation", "affine"), "anchor": ("first", "mean"),
           "remove_trend": ("none", "linear", "lowpass"), "ransac_model": features.MODELS}
# Settings that change a chunk's matches: run redoes chunks made with other values, solve refuses them.
_RUN_KEYS = ("scale", "neighbors", "max_features", "ratio", "ransac_model", "ransac_px", "min_inliers",
             "max_points_per_pair")


def _check_options(a):
    for key, allowed in OPTIONS.items():
        if a[key] not in allowed:
            raise ValueError(f"align.{key} must be one of {allowed}, not {a[key]!r}")


# ----- run: point matches -------------------------------------------------------------

def run(cfg, task_id, num_tasks, overwrite=False):
    """Match this task's chunks and write ``align/matches/chunk_*.npz``."""
    a = cfg["align"]
    df = load_slices(cfg)
    stitch = _stitch_transforms(cfg, df)
    by_z = dict(tuple(df.groupby("z")))
    tiles_by_z = {z: g["tile"].tolist() for z, g in by_z.items()}
    out_dir = step_dir(cfg, "align", "matches")
    out_dir.mkdir(exist_ok=True)
    cache = StackCache(cfg["raw_dir"])
    size, settings = int(a["chunk_slices"]), _settings(a)
    all_chunks = _chunks(z_values(df), size, int(a["neighbors"]))
    todo = cli.my_chunks(all_chunks, task_id, num_tasks)
    log.info("task %d/%d: %d of %d chunks", task_id, num_tasks, len(todo), len(all_chunks))
    for core, ext in todo:
        path, sig = _chunk_path(out_dir, core, size), _signature(ext, tiles_by_z)
        if not overwrite and _chunk_is_current(path, sig, settings):
            log.info("%s exists, skipping", path.name)
            continue
        m = _match_chunk(core, ext, by_z, stitch, cache, a)
        cli.atomic_write(path, lambda tmp: np.savez(tmp, slices=np.array(sig), settings=json.dumps(settings), **m))
        log.info("%s: %d points, %d slice pairs", path.name, len(m["z_a"]),
                 len(set(zip(m["z_a"].tolist(), m["z_b"].tolist()))))


def _stitch_transforms(cfg, df):
    """stitch/tiles.csv as {(z, tile): A}; every selected (z, tile) in ``df`` must have one."""
    path = step_path(cfg, "stitch", "tiles.csv")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run the stitch step first")
    stitch = transforms.read_csv(path, ["z", "tile"])
    missing = [k for k in zip(df["z"].tolist(), df["tile"]) if k not in stitch]
    if missing:
        raise ValueError(f"stitch/tiles.csv has no transform for {len(missing)} selected (z, tile), "
                         f"e.g. {missing[:3]}: re-run stitch merge")
    return stitch


def _settings(a):
    return {k: a[k] for k in _RUN_KEYS}


def _signature(ext, tiles_by_z):
    """The (z, tile) a chunk reads, as 'z:tile': excluding or restoring a single tile redoes the chunk."""
    return [f"{z}:{t}" for z in ext for t in tiles_by_z[z]]


def _chunks(zs, size, neighbors):
    """(core, ext) per chunk: core = the selected z in one block [k*size, (k+1)*size) of global z,
    ext = core + the next ``neighbors`` selected z (neighbours by position, gaps allowed).

    Block bounds don't depend on the selection, so excluding a slice or appending a day only
    changes the chunks around it instead of shifting every later chunk.
    """
    zs = list(zs)
    out, start = [], 0
    for i in range(1, len(zs) + 1):
        if i == len(zs) or zs[i] // size != zs[start] // size:
            out.append((zs[start:i], zs[start:i + neighbors]))
            start = i
    return out


def _chunk_path(out_dir, core, size):
    z0 = core[0] // size * size
    return Path(out_dir) / f"chunk_{z0:06d}-{z0 + size:06d}.npz"


def _chunk_is_current(path, signature, settings):
    """True if ``path`` was matched over exactly these (z, tile) (appended data extends the last
    chunks) with these run settings."""
    if not path.exists():
        return False
    with np.load(path) as f:
        return ("settings" in f and "slices" in f and f["slices"].tolist() == signature
                and json.loads(str(f["settings"])) == settings)


def _match_chunk(core, ext, by_z, stitch, cache, a):
    """Stream through ``ext`` keeping features of the last ``neighbors`` slices only."""
    core = set(core)
    window = deque(maxlen=int(a["neighbors"]))
    tiles = sorted({t for z in ext for t in by_z[z]["tile"]})
    index = {t: i for i, t in enumerate(tiles)}
    out = {k: [] for k in ("z_a", "z_b", "ta", "tb", "qa", "qb")}
    for z_b in ext:
        rows_b = by_z[z_b]
        feats_b = {row["tile"]: _tile_features(cache, row, a) for _, row in rows_b.iterrows()}
        seg_b = rows_b["segment"].iloc[0]
        for z_a, seg_a, feats_a in window:
            if z_a not in core:
                continue
            for ta, tb in _tile_pairs(feats_a, feats_b, seg_a == seg_b):
                qa, qb = _match_tiles(feats_a[ta], feats_b[tb], a)
                for key, v in (("z_a", z_a), ("z_b", z_b), ("ta", index[ta]), ("tb", index[tb])):
                    out[key].append(np.full(len(qa), v))
                out["qa"].append(qa)
                out["qb"].append(qb)
            log.debug("z %d -> %d matched", z_a, z_b)
        window.append((z_b, seg_b, feats_b))
    empty = {"z_a": np.zeros(0, np.int32), "z_b": np.zeros(0, np.int32), "ta": np.zeros(0, np.int16),
             "tb": np.zeros(0, np.int16), "qa": np.zeros((0, 2), np.float32), "qb": np.zeros((0, 2), np.float32)}
    m = {k: np.concatenate([empty[k], *v]).astype(empty[k].dtype) for k, v in out.items()}
    m["tiles"] = np.array(tiles, str)
    m["pa"] = _to_montage(stitch, m["z_a"], m["ta"], m["tiles"], m["qa"]).astype(np.float32)
    m["pb"] = _to_montage(stitch, m["z_b"], m["tb"], m["tiles"], m["qb"]).astype(np.float32)
    m["w"] = np.ones(len(m["z_a"]), np.float32)
    return m


def _to_montage(stitch, z, t, tiles, q):
    """Tile-pixel points ``q`` of slices ``z`` and tiles ``tiles[t]`` -> montage coordinates."""
    key = z.astype(np.int64) * len(tiles) + t
    keys, inv = np.unique(key, return_inverse=True)
    A = np.array([stitch[(int(k) // len(tiles), str(tiles[k % len(tiles)]))] for k in keys])
    A = A.reshape(-1, 2, 3)[inv]
    return np.einsum("nij,nj->ni", A[:, :, :2], q) + A[:, :, 2]


def _tile_pairs(feats_a, feats_b, same_segment):
    """Same tile within a segment; every combination across a segment change (tile grids differ)."""
    if same_segment:
        return [(t, t) for t in feats_a if t in feats_b]
    return [(ta, tb) for ta in feats_a for tb in feats_b]


def _tile_features(cache, row, a):
    """SIFT on one tile at ``align.scale``; keypoints also mapped back to full-res tile pixels."""
    img = cache.read(row)
    small = features.downsample(img, 1.0 / float(a["scale"]))
    xy, desc = features.detect(features.to_uint8(small), nfeatures=int(a["max_features"]))
    f = np.array([img.shape[1] / small.shape[1], img.shape[0] / small.shape[0]])
    return {"xy": xy, "full": (xy + 0.5) * f - 0.5, "desc": desc, "shape": img.shape}


def _match_tiles(fa, fb, a):
    """RANSAC inlier pairs (full-res tile pixels) between two tiles, at most max_points_per_pair."""
    ia, ib = features.match(fa["desc"], fb["desc"], float(a["ratio"]))
    model, inl = features.fit_model(fa["xy"][ia], fb["xy"][ib], a["ransac_model"],
                                    threshold=float(a["ransac_px"]), min_inliers=int(a["min_inliers"]))
    if model is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    pa, pb = fa["full"][ia[inl]], fb["full"][ib[inl]]
    keep = _spread(pa, int(a["max_points_per_pair"]), fa["shape"])
    return pa[keep], pb[keep]


def _spread(pts, n, shape, seed=0):
    """Indices of at most ``n`` points spread over the image: one per grid cell first, then seconds, ..."""
    if len(pts) <= n:
        return np.arange(len(pts))
    k = int(np.ceil(np.sqrt(n)))
    h, w = shape
    cx = np.clip((pts[:, 0] * k // w).astype(int), 0, k - 1)
    cy = np.clip((pts[:, 1] * k // h).astype(int), 0, k - 1)
    perm = np.random.default_rng(seed).permutation(len(pts))
    cell = (cy * k + cx)[perm]
    order = np.argsort(cell, kind="stable")
    first = np.searchsorted(cell[order], cell[order])
    rank = np.empty(len(pts), int)
    rank[order] = np.arange(len(pts)) - first
    return np.sort(perm[np.argsort(rank, kind="stable")[:n]])


# ----- solve: global least squares ----------------------------------------------------

def solve(cfg):
    """Solve all chunks' matches for one transform per selected z; write CSVs and drift.png."""
    a = cfg["align"]
    df = load_slices(cfg)
    zs = np.array(z_values(df))
    if not len(zs):
        raise ValueError("no selected slices")
    m = _load_matches(cfg, df, zs, a)
    log.info("%d points in %d slice pairs over %d slices", len(m["pid"]), len(m["pairs"]), len(zs))

    mats, keep, rejected = solve_transforms(m, len(zs), a)
    # Residuals of the fit itself: trend removal deliberately breaks agreement between slices.
    res = _pair_residuals(m, mats, keep, rejected, zs)
    seams = sorted(df.loc[df["seam"], "z"].unique().tolist())
    seam_pos = np.searchsorted(zs, [z for z in seams if z > zs[0]]).astype(int)
    mats = _remove_trend(mats, zs, seam_pos, a["remove_trend"], float(a["trend_window_slices"]))
    mats = _apply_gauge(mats, a["anchor"])
    _report_gaps(m, keep, zs, int(a["min_inliers"]))

    ts = df.groupby("z")["timestamp"].first().loc[zs].dt.strftime("%Y-%m-%dT%H:%M:%S")
    frame = transforms.to_frame(({"z": int(z), "timestamp": t}, A) for z, t, A in zip(zs, ts, mats))
    cli.atomic_write(step_dir(cfg, "align", "transforms.csv"), lambda tmp: frame.to_csv(tmp, index=False))
    cli.atomic_write(step_dir(cfg, "align", "residuals.csv"), lambda tmp: res.to_csv(tmp, index=False))
    cli.atomic_write(qc_path(cfg, "align_drift.png"), lambda tmp: _plot(tmp, zs, mats, res, seams))
    used = res[~res["rejected"]]
    log.info("rejected %d of %d slice pairs and %d of %d points; median pair rms %.2f px",
             int(res["rejected"].sum()), len(res), int((~keep).sum()), len(keep),
             float(used["rms_px"].median()) if len(used) else float("nan"))


def _load_matches(cfg, df, zs, a):
    """Concatenate the chunk files expected for the selected slices ``df``, sorted by slice pair.

    Points are mapped to montage coordinates with the current stitch/tiles.csv.
    Returns dict with pa, pb (N, 2 float64), w (N), pid (N, pair index) and pairs (P, 2) of
    positions in ``zs``.
    """
    stitch = _stitch_transforms(cfg, df)
    tiles_by_z = df.groupby("z")["tile"].agg(list).to_dict()
    out_dir = step_path(cfg, "align", "matches")
    size, settings = int(a["chunk_slices"]), _settings(a)
    parts, missing = [], []
    for core, ext in _chunks(zs.tolist(), size, int(a["neighbors"])):
        path = _chunk_path(out_dir, core, size)
        if not _chunk_is_current(path, _signature(ext, tiles_by_z), settings):
            missing.append(path.name)
            continue
        with np.load(path) as f:
            parts.append({"z_a": f["z_a"], "z_b": f["z_b"], "w": f["w"],
                          "pa": _to_montage(stitch, f["z_a"], f["ta"], f["tiles"], f["qa"]),
                          "pb": _to_montage(stitch, f["z_b"], f["tb"], f["tiles"], f["qb"])})
    if missing:
        raise FileNotFoundError(f"{len(missing)} match chunks missing, out of date or made with other align "
                                "settings (run 'align run'): "
                                + ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""))
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    key = np.searchsorted(zs, cat["z_a"]).astype(np.int64) * len(zs) + np.searchsorted(zs, cat["z_b"])
    order = np.argsort(key, kind="stable")
    keys, pid = np.unique(key[order], return_inverse=True)
    return {"pa": cat["pa"][order].astype(np.float64), "pb": cat["pb"][order].astype(np.float64),
            "w": cat["w"][order].astype(np.float64), "pid": pid,
            "pairs": np.column_stack([keys // len(zs), keys % len(zs)])}


def solve_transforms(m, n, a):
    """Robust global solve. ``m`` as returned by ``_load_matches``; ``n`` = number of slices.

    Each point pair asks T[i_a](pa) = T[i_b](pb). With T(p) = p + D(p), D is linear in the
    unknowns and x / y separate into two systems sharing one sparse normal matrix.
    Returns (mats (n, 2, 3), per-point keep mask, per-pair rejected mask), anchored at slice 0.
    """
    pid, pairs = m["pid"], m["pairs"]
    keep = np.ones(len(pid), bool)
    rejected = np.zeros(len(pairs), bool)
    for it in range(int(a["robust_iterations"]) + 1):
        if it:
            keep, rejected = _reject(err, pid, len(pairs), float(a["reject_px"]))
        mats = _solve_once(m, keep, n, a)
        err = _point_errors(m, mats)
        rms = float(np.sqrt(np.mean((err[keep] ** 2).sum(1)))) if keep.any() else float("nan")
        log.info("iteration %d: rms %.3f px over %d points", it, rms, int(keep.sum()))
    return mats, keep, rejected


def _solve_once(m, keep, n, a):
    pa, pb, pid, pairs = m["pa"], m["pb"], m["pid"], m["pairs"]
    w = m["w"] * keep
    P = len(pairs)
    affine = a["model"] == "affine"
    # Affine unknowns use coordinates centred and scaled to ~1 so the regularisation is
    # dimensionless and the normal matrix well conditioned.
    centre, s = np.zeros(2), 1.0
    if affine and len(pa):
        centre = pa.mean(0)
        s = float(np.sqrt(((pa - centre) ** 2).sum(1).mean())) or 1.0
    k = 3 if affine else 1
    Ha, Hb = _basis(pa, affine, centre, s), _basis(pb, affine, centre, s)
    d = (pb - pa) / s

    rows, cols, vals = [], [], []

    def add(ri, ci, G):  # G: (len(ri), k, k) blocks at block rows ri, block cols ci
        r = ri[:, None, None] * k + np.arange(k)[None, :, None]
        c = ci[:, None, None] * k + np.arange(k)[None, None, :]
        rows.append(np.broadcast_to(r, G.shape).ravel())
        cols.append(np.broadcast_to(c, G.shape).ravel())
        vals.append(G.ravel())

    ia, ib = pairs[:, 0], pairs[:, 1]
    Gab = _pair_sums(pid, P, Ha, Hb, w)
    add(ia, ia, _pair_sums(pid, P, Ha, Ha, w))
    add(ib, ib, _pair_sums(pid, P, Hb, Hb, w))
    add(ia, ib, -Gab)
    add(ib, ia, -Gab.transpose(0, 2, 1))
    rhs = np.zeros((n * k, 2))
    for ax in range(2):
        for j in range(k):
            np.add.at(rhs[:, ax], ia * k + j, np.bincount(pid, w * Ha[:, j] * d[:, ax], minlength=P))
            np.add.at(rhs[:, ax], ib * k + j, -np.bincount(pid, w * Hb[:, j] * d[:, ax], minlength=P))

    pair_w = np.bincount(pid, w, minlength=P)
    lam = SMOOTHNESS * (np.median(pair_w[pair_w > 0]) if (pair_w > 0).any() else 1.0)
    i = np.arange(n - 1)
    eye = np.broadcast_to(lam * np.eye(k), (n - 1, k, k))
    add(i, i, eye)
    add(i + 1, i + 1, eye)
    add(i, i + 1, -eye)
    add(i + 1, i, -eye)
    if affine:
        # Tikhonov pull of the linear part toward identity (deviation 0), scaled by point count.
        count = np.bincount(ia, pair_w, minlength=n) + np.bincount(ib, pair_w, minlength=n)
        reg = float(a["affine_regularization"]) * np.maximum(count, 1.0)
        for j in range(2):
            rows.append(np.arange(n) * k + j)
            cols.append(np.arange(n) * k + j)
            vals.append(reg)

    M = sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n * k, n * k)).tocsr()
    # Pin slice 0 (all of it for anchor=first; only its translation for affine with anchor=mean,
    # whose linear gauge the regularisation fixes). The final gauge is applied later.
    pinned = np.zeros(n * k, bool)
    pinned[:k] = True
    if affine and a["anchor"] == "mean":
        pinned[:2] = False
    free = ~pinned
    U = np.zeros((n * k, 2))
    if free.any():
        U[free] = spsolve(M[free][:, free].tocsc(), rhs[free]).reshape(-1, 2)
    return _to_mats(U.reshape(n, k, 2), centre, s)


def _basis(p, affine, centre, s):
    if not affine:
        return np.ones((len(p), 1))
    return np.column_stack([(p - centre) / s, np.ones(len(p))])


def _pair_sums(pid, P, X, Y, w):
    """(P, k, k) per-pair sums of w * X[:, j] * Y[:, l]."""
    k = X.shape[1]
    G = np.empty((P, k, k))
    for j in range(k):
        for l in range(k):
            G[:, j, l] = np.bincount(pid, w * X[:, j] * Y[:, l], minlength=P)
    return G


def _to_mats(U, centre, s):
    """Deviation unknowns (n, k, 2) -> 2×3 matrices. Affine: D(p) = dL (p - centre) + s t."""
    n, k, _ = U.shape
    mats = np.tile(transforms.identity(), (n, 1, 1))
    if k == 1:
        mats[:, :, 2] = U[:, 0, :]
        return mats
    dL = U[:, :2, :].transpose(0, 2, 1)
    mats[:, :, :2] += dL
    mats[:, :, 2] = s * U[:, 2, :] - dL @ centre
    return mats


def _point_errors(m, mats, block=1 << 20):
    """T[i_a](pa) - T[i_b](pb) per point, in px (in blocks to bound memory)."""
    err = np.empty_like(m["pa"])
    for s in range(0, len(err), block):
        sl = slice(s, s + block)
        A, B = mats[m["pairs"][m["pid"][sl]].T]
        err[sl] = (np.einsum("nij,nj->ni", A[:, :, :2], m["pa"][sl]) + A[:, :, 2]
                   - np.einsum("nij,nj->ni", B[:, :, :2], m["pb"][sl]) - B[:, :, 2])
    return err


def _reject(err, pid, P, reject_px):
    """Points beyond max(reject_px, 3 sigma) (sigma from the MAD), and pairs that mostly fail."""
    comp = err.ravel()
    sigma = 1.4826 * np.median(np.abs(comp - np.median(comp))) if len(comp) else 0.0
    keep = np.linalg.norm(err, axis=1) <= max(reject_px, 3 * sigma)
    total = np.bincount(pid, minlength=P)
    rejected = np.bincount(pid, keep, minlength=P) < MIN_PAIR_FRACTION * total
    return keep & ~rejected[pid], rejected


def _remove_trend(mats, zs, seam_pos, mode, window):
    """Subtract slow drift (linear fit or Gaussian low-pass over z) from the translations.

    Steps at seams (``seam_pos``, positions > 0) are real stage jumps and are kept: the trend is
    fitted to the translations with those steps taken out.
    """
    if mode == "none" or len(zs) < 2:
        return mats
    t = mats[:, :, 2]
    steps = np.zeros_like(t)
    steps[seam_pos] = t[seam_pos] - t[seam_pos - 1]
    slow = t - np.cumsum(steps, axis=0)
    if mode == "linear":
        slope, offset = np.polyfit(zs.astype(float), slow, 1)
        trend = np.outer(zs, slope) + offset
    else:
        trend = gaussian_filter1d(slow, window / 2.355, axis=0, mode="nearest")
    out = mats.copy()
    out[:, :, 2] -= trend
    return out


def _apply_gauge(mats, anchor):
    """Shift the aligned frame: first slice's translation 0, or mean translation 0."""
    out = mats.copy()
    out[:, :, 2] -= mats[0, :, 2] if anchor == "first" else mats[:, :, 2].mean(0)
    return out


def _report_gaps(m, keep, zs, min_points):
    """Warn about slices, and consecutive-slice boundaries, that no surviving matches constrain."""
    n = len(zs)
    ia, ib = m["pairs"][:, 0], m["pairs"][:, 1]
    pair_n = np.bincount(m["pid"], keep, minlength=len(ia))
    count = np.bincount(ia, pair_n, minlength=n) + np.bincount(ib, pair_n, minlength=n)
    few = zs[count < min_points]
    if len(few):
        log.warning("%d slices have fewer than %d matched points; their transforms come only from the "
                    "smoothness prior: z %s", len(few), min_points, _ranges(few))
    used = pair_n > 0
    cover = np.cumsum(np.bincount(ia[used], minlength=n) - np.bincount(ib[used], minlength=n))[:-1]
    breaks = np.flatnonzero(cover == 0)
    if len(breaks):
        pairs = ", ".join(f"{zs[i]}|{zs[i + 1]}" for i in breaks[:50]) + (" ..." if len(breaks) > 50 else "")
        log.warning("no matches connect these consecutive slices (alignment across them is a guess): %s", pairs)


def _ranges(values):
    """'3-7, 12, 40-41' from sorted integers."""
    values = list(values)
    out, start = [], 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[i - 1] + 1:
            a, b = values[start], values[i - 1]
            out.append(str(a) if a == b else f"{a}-{b}")
            start = i
    return ", ".join(out)


def _pair_residuals(m, mats, keep, rejected, zs):
    """Per pair: n points, rms / max residual of the points used (all points if the pair was rejected)."""
    err = np.linalg.norm(_point_errors(m, mats), axis=1)
    pid, P = m["pid"], len(m["pairs"])
    use = keep | rejected[pid]
    n_use = np.bincount(pid, use, minlength=P)
    rms = np.sqrt(np.bincount(pid, use * err ** 2, minlength=P) / np.maximum(n_use, 1))
    starts = np.searchsorted(pid, np.arange(P))
    mx = np.maximum.reduceat(np.where(use, err, 0.0), starts) if P else np.zeros(0)
    return pd.DataFrame({"z_a": zs[m["pairs"][:, 0]], "z_b": zs[m["pairs"][:, 1]],
                         "n": np.bincount(pid, minlength=P), "rms_px": rms.round(4),
                         "max_px": mx.round(4), "rejected": rejected})


def _plot(path, zs, mats, res, seams):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, height_ratios=[2, 1])
    for ax in (ax1, ax2):
        for z in seams:
            ax.axvline(z, color="#b5b4ae", lw=0.8, ls="--", zorder=0)
        ax.grid(color="#e6e5e0", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    ax1.plot(zs, mats[:, 0, 2], color="#2a78d6", lw=1.5, label="tx")
    ax1.plot(zs, mats[:, 1, 2], color="#eb6834", lw=1.5, label="ty")
    ax1.set_ylabel("translation (px)")
    ax1.set_title("align: montage → aligned" + ("  (dashed: seams)" if seams else ""), loc="left", fontsize=10)
    ax1.legend(frameon=False, loc="best")
    ok, bad = res[~res["rejected"]], res[res["rejected"]]
    ax2.plot(ok["z_a"], ok["rms_px"], ".", ms=3, alpha=0.6, color="#2a78d6", label="pair rms")
    if len(bad):
        ax2.plot(bad["z_a"], bad["rms_px"], "x", ms=5, color="#d03b3b", label="rejected pair")
    ax2.set_ylabel("residual rms (px)")
    ax2.set_xlabel("z")
    ax2.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    p = cli.base_parser(__doc__.splitlines()[0])
    p.add_argument("command", choices=["run", "solve"])
    args = p.parse_args(argv)
    cfg = cli.setup(args, DEFAULTS)
    _check_options(cfg["align"])
    if args.command == "run":
        run(cfg, *cli.task_info(args), overwrite=args.overwrite)
    else:
        solve(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
