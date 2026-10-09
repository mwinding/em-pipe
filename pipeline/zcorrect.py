"""Axial (slice thickness) correction, after Hanslovsky et al. 2017, simplified for FIB-SEM.

Slices that are milled thinner look more alike. ``run`` measures the normalised
cross-correlation (NCC) between each slice and its next ``max_distance`` selected slices
on a few fixed crops of the aligned volume. ``solve`` estimates the similarity-vs-distance
curve f(d), inverts it to turn each NCC into a distance, and solves for slice positions
by weighted least squares; it re-estimates f from the positions and repeats. Seams are
breaks: NCC across them is not used and the spacing there is nominal.
"""

import logging
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

from . import transforms
from .cli import atomic_write, base_parser, chunks, my_chunks, setup, task_info
from .config import qc_path, step_dir, step_path
from .features import downsample
from .slices import StackCache, load_slices, voxel_size_nm, z_values

log = logging.getLogger(__name__)

DEFAULTS = {
    "zcorrect": {
        "enabled": False,           # render uses positions.csv only when true
        "chunk_slices": 200,        # selected slices per array chunk in `run`
        "max_distance": 8,          # compare each slice with its next N selected slices
        "crop_px": 1024,            # square crop per tile (full-resolution px), centred on the tile
        "factor": 1,                # downsample crops before NCC
        "iterations": 3,            # solve: re-estimate f(d) from positions and re-solve
        "regularization": 0.1,      # pull of each spacing toward nominal, relative to one adjacent-pair NCC
        "min_spacing_fraction": 0.25,  # minimum spacing as a fraction of nominal
        # Hold the rolling mean spacing over this many slices at nominal, so slow changes in
        # image texture are not read as thickness drift (0: only per block between seams).
        "nominal_window": 200,
    },
}


# ----- run --------------------------------------------------------------------------

def _crop_origins(df, stitch, align, size):
    """{segment: {tile: (x0, y0)}}: aligned-frame corner of a crop centred on each tile at the
    first selected z of the segment that has the tile. Fixed for the whole selection so every
    chunk measures the same areas."""
    origins = {}
    for seg, sdf in df.groupby("segment"):
        boxes, n_tiles = {}, sdf["tile"].nunique()
        for row in sdf.itertuples():   # sorted by (z, tile)
            if row.tile in boxes or row.z not in align or (row.z, row.tile) not in stitch:
                continue
            A = transforms.compose(align[row.z], stitch[(row.z, row.tile)])
            cx, cy = transforms.apply(A, [(row.width / 2, row.height / 2)])[0]
            boxes[row.tile] = (int(round(cx - size / 2)), int(round(cy - size / 2)))
            if len(boxes) == n_tiles:
                break
        if boxes:
            origins[seg] = boxes
    return origins


def _read_crop(cache, row, A, x0, y0, size, factor):
    """Crop [x0, x0+size) x [y0, y0+size) of the aligned frame from one tile (A: tile -> aligned),
    as a zero-mean unit-norm vector, or None if the crop leaves the tile."""
    h, w = int(row["height"]), int(row["width"])
    inv = transforms.invert(A)
    if transforms.is_translation(A):
        tx, ty = int(round(x0 + inv[0, 2])), int(round(y0 + inv[1, 2]))
        if tx < 0 or ty < 0 or tx + size > w or ty + size > h:
            return None
        img = cache.read(row, slice(ty, ty + size), slice(tx, tx + size)).astype(np.float32)
    else:
        pts = transforms.apply(inv, transforms.corners(size, size) + (x0, y0))
        bx0, by0 = np.floor(pts.min(axis=0)).astype(int) - 1
        bx1, by1 = np.ceil(pts.max(axis=0)).astype(int) + 2
        if bx0 < 0 or by0 < 0 or bx1 > w or by1 > h:
            return None
        block = cache.read(row, slice(by0, by1), slice(bx0, bx1)).astype(np.float32)
        M = transforms.compose(transforms.translation(-x0, -y0),
                               transforms.compose(A, transforms.translation(bx0, by0)))
        img = cv2.warpAffine(block, M, (size, size), flags=cv2.INTER_LINEAR)
    v = downsample(img, factor).ravel()
    v = v - v.mean()
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else None


def _chunk_ncc(core, ext, rows, segment, stitch, align, origins, cache, zc):
    """NCC (median over crops) of each z in ``core`` with its next max_distance z in ``ext``."""
    core = set(core)
    window = deque(maxlen=zc["max_distance"])
    z_a, z_b, ncc = [], [], []
    for z in ext:
        seg = segment[z]
        vecs = {}
        for tile, (x0, y0) in origins.get(seg, {}).items():
            if (z, tile) in rows and (z, tile) in stitch and z in align:
                v = _read_crop(cache, rows[(z, tile)], transforms.compose(align[z], stitch[(z, tile)]),
                               x0, y0, zc["crop_px"], zc["factor"])
                if v is not None:
                    vecs[tile] = v
        for zp, seg_p, vecs_p in window:
            if zp in core and seg_p == seg:
                vals = [float(vecs_p[t] @ vecs[t]) for t in vecs_p if t in vecs]
                z_a.append(zp)
                z_b.append(z)
                ncc.append(np.median(vals) if vals else np.nan)
        window.append((z, seg, vecs))
    return np.array(z_a, int), np.array(z_b, int), np.array(ncc, float)


def _chunk_path(cfg, core):
    return step_dir(cfg, "zcorrect", "ncc", f"chunk_{core[0]:06d}-{core[-1] + 1:06d}.npz")


def _chunk_is_current(path, ext, zc=None):
    """True if ``path`` exists and was measured over exactly ``ext`` (a growing dataset extends the
    last chunks) and, when ``zc`` is given, with its crop size and downsampling."""
    if not path.exists():
        return False
    with np.load(path) as f:
        if "zs" not in f or f["zs"].tolist() != list(ext):
            return False
        return zc is None or (int(f["crop_px"]), float(f["factor"])) == (int(zc["crop_px"]), float(zc["factor"]))


def _read_transforms(cfg):
    paths = {step: step_path(cfg, step, name) for step, name in (("stitch", "tiles.csv"), ("align", "transforms.csv"))}
    for step, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run the {step} step first")
    return transforms.read_csv(paths["stitch"], ["z", "tile"]), transforms.read_csv(paths["align"], ["z"])


def run(cfg, args):
    zc = cfg["zcorrect"]
    df = load_slices(cfg)
    zs = z_values(df)
    stitch, align = _read_transforms(cfg)
    origins = _crop_origins(df, stitch, align, zc["crop_px"])
    rows = {(r["z"], r["tile"]): r for r in df.to_dict("records")}
    segment = dict(zip(df["z"], df["segment"]))
    cache = StackCache(cfg["raw_dir"])
    task_id, num_tasks = task_info(args)
    all_chunks = chunks(zs, zc["chunk_slices"], zc["max_distance"])
    todo = my_chunks(all_chunks, task_id, num_tasks)
    log.info("task %d/%d: %d of %d chunks, crops per segment: %s", task_id, num_tasks, len(todo),
             len(all_chunks), {s: len(o) for s, o in origins.items()})
    for core, ext in todo:
        path = _chunk_path(cfg, core)
        if not args.overwrite and _chunk_is_current(path, ext, zc):
            log.info("skip %s (exists)", path.name)
            continue
        z_a, z_b, ncc = _chunk_ncc(core, ext, rows, segment, stitch, align, origins, cache, zc)
        atomic_write(path, lambda p: np.savez(p, z_a=z_a, z_b=z_b, ncc=ncc, zs=np.array(ext, int),
                                              crop_px=zc["crop_px"], factor=zc["factor"]))
        log.info("%s: %d pairs, %d without valid crops", path.name, len(ncc), int(np.isnan(ncc).sum()))


# ----- solve ------------------------------------------------------------------------

def _fit_curve(delta, ncc, min_count=5):
    """Knots (x, y) of a decreasing f(d) with f(0) = 1: medians of NCC and distance in half-slice bins.

    With a monotone f, the median distance and median NCC of a bin are one point on the curve,
    so bins mixing short and long pairs still give unbiased knots.
    """
    bins = np.round(delta * 2).astype(int)
    xs, ys = [0.0], [1.0]
    for b in np.unique(bins[bins > 0]):
        m = bins == b
        if m.sum() >= min_count:
            xs.append(float(np.median(delta[m])))
            ys.append(float(np.median(ncc[m])))
    if len(xs) < 2:
        raise RuntimeError("too few NCC pairs to estimate the similarity curve")
    # Strictly decreasing so it can be inverted; flat parts carry (almost) no weight.
    ys = np.minimum.accumulate(ys) - 1e-6 * np.arange(len(ys))
    return np.array(xs), ys


def _invert(xs, ys, ncc):
    """Distance at which f reaches each NCC, and the slope of f there (0 beyond the curve)."""
    d = np.interp(ncc, ys[::-1], xs[::-1])
    i = np.clip(np.searchsorted(xs, d, side="right") - 1, 0, len(xs) - 2)
    slope = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])
    slope[ncc < ys[-1]] = 0.0
    return d, slope


def _solve_spacings(ia, ib, dist, w, nominal, reg):
    """Weighted least squares for spacings s (s_i between positions i and i+1):
    sum(s[ia:ib]) = dist per pair, plus reg * (s - nominal)^2."""
    n, lengths = len(nominal), ib - ia
    rows = np.repeat(np.arange(len(ia)), lengths)
    starts = np.repeat(np.cumsum(lengths) - lengths, lengths)
    cols = np.repeat(ia, lengths) + np.arange(lengths.sum()) - starts
    A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ia), n))
    lhs = A.T @ sparse.diags(w) @ A + max(reg, 1e-9) * sparse.eye(n)
    rhs = A.T @ (w * dist) + max(reg, 1e-9) * nominal
    return spsolve(lhs.tocsc(), rhs)


def _constrain(s, nominal, brk, min_fraction, window):
    """Nominal spacing at breaks. Within each block between breaks: hold the rolling mean
    spacing over ``window`` slices at nominal, enforce the minimum spacing, and rescale the
    excess so the block keeps its nominal depth."""
    s = np.where(brk, nominal, s)
    lo = min_fraction * nominal
    block = np.cumsum(brk)
    for b in np.unique(block[~brk]):
        m = (block == b) & ~brk
        if window:
            ratio = pd.Series(s[m] / nominal[m]).rolling(window, center=True, min_periods=1).mean()
            s[m] = s[m] / ratio.to_numpy()
        excess = np.maximum(s[m] - lo[m], 0)
        room = (nominal[m] - lo[m]).sum()
        s[m] = lo[m] + (excess * room / excess.sum() if excess.sum() > 0 else nominal[m] - lo[m])
    return s


def _load_pairs(cfg, zs, zc):
    """(z_a, z_b, ncc) from the chunk files expected for the current selection (NaN rows dropped)."""
    frames, missing = [], []
    for core, ext in chunks(zs.tolist(), zc["chunk_slices"], zc["max_distance"]):
        path = _chunk_path(cfg, core)
        if not _chunk_is_current(path, ext):
            missing.append(path.name)
            continue
        with np.load(path) as d:
            frames.append(pd.DataFrame({k: d[k] for k in ("z_a", "z_b", "ncc")}))
    if missing:
        raise FileNotFoundError(f"{len(missing)} NCC chunks missing or out of date (run 'zcorrect run'): "
                                + ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""))
    return pd.concat(frames).dropna()


def solve(cfg, args):
    zc = cfg["zcorrect"]
    df = load_slices(cfg)
    zs = np.array(z_values(df))
    if len(zs) < 2:
        raise SystemExit("need at least two selected slices")
    pairs = _load_pairs(cfg, zs, zc)
    ia, ib = np.searchsorted(zs, pairs["z_a"]), np.searchsorted(zs, pairs["z_b"])
    ncc = pairs["ncc"].to_numpy(float)

    # Breaks (seams, segment changes) split the slices into blocks; drop pairs across them.
    per_z = df.groupby("z").agg(seam=("seam", "any"), segment=("segment", "first"),
                                timestamp=("timestamp", "first")).reindex(zs)
    brk = per_z["seam"].to_numpy()[1:] | (np.diff(per_z["segment"].to_numpy()) != 0)
    block = np.concatenate([[0], np.cumsum(brk)])
    keep = block[ia] == block[ib]
    ia, ib, ncc = ia[keep], ib[keep], ncc[keep]
    missing = len(zs) - len(np.union1d(ia, ib))
    log.info("%d pairs over %d slices, %d breaks, %d slices without pairs", len(ncc), len(zs),
             int(brk.sum()), missing)

    # Positions in nominal slice units: global z counts excluded slices too, so a hole in the
    # selection is a nominal gap of more than one slice.
    nominal = np.diff(zs).astype(float)
    d_nominal = (zs[ib] - zs[ia]).astype(float)
    s = nominal.copy()
    adjacent = ib - ia == 1
    for it in range(max(1, zc["iterations"])):
        pos = np.concatenate([[0.0], np.cumsum(s)])
        xs, ys = _fit_curve(pos[ib] - pos[ia], ncc)
        dist, slope = _invert(xs, ys, ncc)
        # Inverse variance of the distance estimate (NCC noise / slope^2), favouring short pairs;
        # scaled so a typical adjacent pair has weight 1.
        w = slope ** 2 / d_nominal
        if (adjacent & (w > 0)).any():
            w = w / np.median(w[adjacent & (w > 0)])
        s = _solve_spacings(ia, ib, dist, w, nominal, zc["regularization"])
        s = _constrain(s, nominal, brk, zc["min_spacing_fraction"], zc["nominal_window"])
        log.info("iteration %d: f(d) knots %s; spacing/nominal min %.2f max %.2f std %.3f", it + 1,
                 ", ".join(f"{x:.2f}:{y:.2f}" for x, y in zip(xs, ys)),
                 (s / nominal).min(), (s / nominal).max(), (s / nominal).std())

    vz = voxel_size_nm(cfg, df["file"].unique())[0]
    pos = np.concatenate([[0.0], np.cumsum(s)])
    out = step_dir(cfg, "zcorrect")
    # Same origin as uncorrected slices (z * voxel_z), so positions stay comparable across runs.
    frame = pd.DataFrame({"z": zs, "timestamp": per_z["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S").to_numpy(),
                          "position_nm": vz * (zs[0] + pos)})
    atomic_write(out / "positions.csv", lambda p: frame.to_csv(p, index=False))
    seams = zs[1:][brk]
    _plot(qc_path(cfg, "zcorrect.png"), zs, vz * s / nominal, vz, seams, pos[ib] - pos[ia], ncc, xs, ys)
    log.info("wrote %s (voxel z %.2f nm)", out / "positions.csv", vz)


def _plot(path, zs, spacing_nm, vz, seams, delta, ncc, xs, ys):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), gridspec_kw={"height_ratios": [3, 2]})
    window = max(1, min(51, len(spacing_nm) // 4 * 2 + 1))
    rolling = pd.Series(spacing_nm).rolling(window, center=True, min_periods=1).mean()
    for z in seams:
        ax1.axvline(z, color="#eb6834", lw=0.6, alpha=0.6)
    ax1.plot(zs[1:], spacing_nm, color="#b5b4ae", lw=0.6, label="spacing")
    ax1.plot(zs[1:], rolling, color="#2a78d6", lw=2, label=f"rolling mean ({window} slices)")
    ax1.axhline(vz, color="#52514e", lw=1, ls="--", label="nominal")
    ax1.set(xlabel="z (slice)", ylabel="spacing from previous slice (nm per slice)",
            title=f"Estimated slice spacing (orange lines: seams, {len(seams)})")
    ax1.legend(loc="upper right", frameon=False)
    if len(delta) > 20000:
        idx = np.random.default_rng(0).choice(len(delta), 20000, replace=False)
        delta, ncc = delta[idx], ncc[idx]
    ax2.scatter(delta, ncc, s=4, color="#8f8e88", alpha=0.5, linewidths=0, label="slice pairs")
    ax2.plot(xs, ys, color="#2a78d6", lw=2, marker="o", ms=4, label="f(d)")
    ax2.set(xlabel="estimated distance (nominal slices)", ylabel="NCC", title="Similarity vs distance")
    ax2.legend(loc="upper right", frameon=False)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    atomic_write(path, lambda p: fig.savefig(p, dpi=110))
    plt.close(fig)


def main(argv=None):
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("command", choices=["run", "solve"])
    args = p.parse_args(argv)
    cfg = setup(args, DEFAULTS)
    if not cfg["zcorrect"]["enabled"]:
        log.info("zcorrect.enabled is false: render will not use these results")
    {"run": run, "solve": solve}[args.command](cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
