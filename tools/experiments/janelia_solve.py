"""Janelia-style intensity matching, step 2: solve a smooth linear intensity map per tile from the
patch-pair relations of tools/experiments/janelia_bins.py, and the common display window per slice.

    python tools/experiments/janelia_solve.py BINS_DIR TILE_WINDOWS ZS OUT_PREFIX --variant A|B
        [--shape-zs ZS] [--grid 8] [--smooth 0.3] [--prior 0.02] [--gain-power 2] [--jump]

Model (as saalfeldlab/render's intensity-adjust, adapted): every tile is split into grid x grid
patches, patch p of tile t maps raw (flat-fielded) values v to g[t,p] (v - R) + R + c[t,p]; c is the
map's change at the fixed reference intensity R, so smoothness and priors act on values inside the
data range (lesson: a gain error times 37000 ruins any match made at v = 0). Applied in render with
bilinear interpolation of g and c between patch centres (clamped at the tile border).

Solve per slice, gains first, then c given the gains, each a weighted linear least squares:
  relation (bin j: v_a ~ gain_j v_b + offset_j between patch pa of a and pb of b)
      gains:  wg_j (gain_j g[a,pa] - g[b,pb]) = 0,  wg_j = sqrt(n_j) min(1, cnr_j / 10) ** gain_power
      c:      w_j (c[a,pa] - c[b,pb]) = w_j (g[b,pb] (m_j - R) - g[a,pa] (gain_j m_j + offset_j - R)),
              w_j = sqrt(n_j), m_j = the relation's centre (median of v_b): matched at its own data
  smoothness between 4-neighbour patches of a tile (Janelia's neighbour identity matches), weight
      smooth x the median relation weight; prior: each tile's mean g -> 1 (c: none by default; weight
      prior x the median relation weight per patch; Janelia pulls every patch towards Identity, which
      bends the maps where interior patches have no relation: --patch-prior); mean(g) = 1, mean(c) = 0
      over all patches with a large weight.

Variant A: that solve at every slice (Janelia-faithful). Variant B: each tile's within-tile shape
(g / mean g, c - mean c over its patches) from the variant-A solves of SHAPE_ZS, median per (segment,
tile); at each slice only one level per tile is solved (grid 1) from the relations corrected by the
shapes, and maps = shape followed by the tile level.

Display window per slice: C = median over tiles of the tile's mean map applied to its own window ends
(TILE_WINDOWS: intensity.compute_levels without balance; --jump: with the per-slice jump).

Writes OUT_PREFIX_maps.csv (z, tile, p, g, c), OUT_PREFIX_windows.csv (z, lo, hi) and
OUT_PREFIX_resid.csv (per bin: the predicted step at the seam, in display greys, before (per-tile
windows) and after).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix

R = 36000.0     # reference raw intensity at which c is defined (35h tiles: ~33000-40000)
GAIN_CNR = 10.0


# ----- geometry --------------------------------------------------------------------------

def interp_weights(x, size, grid):
    """(i0, t) of bilinear interpolation between patch centres along one axis, clamped at the border."""
    u = np.clip((np.asarray(x, float) + 0.5) / size * grid - 0.5, 0, grid - 1)
    if grid == 1:
        return np.zeros(u.shape, int), np.zeros(u.shape)
    i0 = np.minimum(np.floor(u).astype(int), grid - 2)
    return i0, u - i0


def interp_grid(vals, x, y, w, h):
    """Bilinear interpolation of a (grid, grid) patch array at tile pixels (x, y) (arrays of the same shape)."""
    G = vals.shape[0]
    cx, tx = interp_weights(x, w, G)
    ry, ty = interp_weights(y, h, G)
    if G == 1:
        return np.full(np.shape(x), float(vals[0, 0]))
    v00, v01 = vals[ry, cx], vals[ry, cx + 1]
    v10, v11 = vals[ry + 1, cx], vals[ry + 1, cx + 1]
    return (1 - ty) * ((1 - tx) * v00 + tx * v01) + ty * ((1 - tx) * v10 + tx * v11)


def map_rows(vals, rows, cols, h, w):
    """The bilinear field of a (grid, grid) patch array on tile pixels rows x cols (ranges), float32."""
    G = vals.shape[0]
    if G == 1:
        return np.full((len(rows), len(cols)), float(vals[0, 0]), np.float32)
    cx, tx = interp_weights(cols, w, G)
    across = (1 - tx)[None, :] * vals[:, cx] + tx[None, :] * vals[:, cx + 1]     # (G, len(cols))
    ry, ty = interp_weights(rows, h, G)
    return ((1 - ty)[:, None] * across[ry] + ty[:, None] * across[ry + 1]).astype(np.float32)


# ----- solve -----------------------------------------------------------------------------

def _connected_tiles(rels):
    tiles = sorted(set(rels["a"]) | set(rels["b"]))
    k = {t: i for i, t in enumerate(tiles)}
    e = coo_matrix((np.ones(len(rels)), (rels["a"].map(k), rels["b"].map(k))), shape=(len(tiles), len(tiles)))
    _, lab = connected_components(e, directed=False)
    big = np.bincount(lab).argmax()
    return [t for t in tiles if lab[k[t]] == big]


def _neighbours(grid):
    out = []
    for r in range(grid):
        for c in range(grid):
            if c + 1 < grid:
                out.append((r * grid + c, r * grid + c + 1))
            if r + 1 < grid:
                out.append((r * grid + c, (r + 1) * grid + c))
    return out


def solve_patches(rels, grid=8, smooth=0.3, prior=0.02, gain_power=2.0, smooth_c=None, patch_prior=0.0, prior_c=0.0):
    """{tile: (g (grid, grid), c (grid, grid))} from one slice's bin relations (pa, pb >= 0).

    rels columns: a, b, pa, pb, gain, offset, n, cnr, centre. smooth, prior and patch_prior are
    fractions of the median relation weight (gain system: of wg, c system: of w); smooth_c defaults
    to smooth. prior pulls each tile's mean g to 1 (prior_c: mean c to 0; one row per tile, weighted
    as if every patch had it); patch_prior is Janelia's per-patch Identity pull. Any pull that
    disagrees with what the seams say about a tile's level bends its map (the edge patches follow the
    relations, the interior follows the pull), so the priors must stay well below the smoothness:
    the gain prior only steadies poorly measured gains, c needs none (the global mean fixes it).
    """
    rels = rels[(rels["pa"] >= 0) & (rels["pb"] >= 0)]
    if rels.empty:
        return {}
    tiles = _connected_tiles(rels)
    rels = rels[rels["a"].isin(tiles) & rels["b"].isin(tiles)]
    P = grid * grid
    N = len(tiles) * P
    k = {t: i for i, t in enumerate(tiles)}
    ia = rels["a"].map(k).to_numpy() * P + rels["pa"].to_numpy(int)
    ib = rels["b"].map(k).to_numpy() * P + rels["pb"].to_numpy(int)
    gain, off = rels["gain"].to_numpy(float), rels["offset"].to_numpy(float)
    m = rels["centre"].to_numpy(float)
    w = np.sqrt(rels["n"].to_numpy(float))
    wg = w * np.minimum(1.0, rels["cnr"].to_numpy(float) / GAIN_CNR) ** gain_power
    nb = [(k[t] * P + p, k[t] * P + q) for t in tiles for p, q in _neighbours(grid)]
    nt = len(tiles)

    def system(wrel, coef_a, coef_b, rhs_rel, ws, wp, wpp, target):
        nr, ns = len(wrel), len(nb)
        A = np.zeros((nr + ns + nt + N + 1, N))
        b = np.zeros(len(A))
        j = np.arange(nr)
        np.add.at(A, (j, ia), wrel * coef_a)
        np.add.at(A, (j, ib), -wrel * coef_b)
        b[:nr] = wrel * rhs_rel
        for s, (p, q) in enumerate(nb):
            A[nr + s, p], A[nr + s, q] = ws, -ws
        r0 = nr + ns
        for i in range(nt):   # tile mean -> target, as strong as the same pull on each of its P patches
            A[r0 + i, i * P:(i + 1) * P] = wp * np.sqrt(P) / P
            b[r0 + i] = wp * np.sqrt(P) * target
        r0 += nt
        A[r0 + np.arange(N), np.arange(N)] = wpp
        b[r0:r0 + N] = wpp * target
        big = 100.0 * wrel.max()
        A[-1] = big / N
        b[-1] = big * target
        return np.linalg.lstsq(A, b, rcond=None)[0]

    sm_c = smooth if smooth_c is None else smooth_c
    mg, mw = np.median(wg[wg > 0]) if (wg > 0).any() else np.median(w), np.median(w)
    g = system(wg, gain, np.ones_like(gain), np.zeros_like(gain), smooth * mg, prior * mg, patch_prior * mg, 1.0)
    rhs = g[ib] * (m - R) - g[ia] * (gain * m + off - R)
    c = system(w, np.ones_like(gain), np.ones_like(gain), rhs, sm_c * mw, prior_c * mw, patch_prior * mw, 0.0)
    return {t: (g[k[t] * P:(k[t] + 1) * P].reshape(grid, grid), c[k[t] * P:(k[t] + 1) * P].reshape(grid, grid))
            for t in tiles}


def shape_corrected(rels, shapes, grid):
    """Relations between shape-corrected values u = sg (v - R) + R + sc (shapes: tile -> (sg, sc) grids),
    as whole-tile relations (pa = pb = 0) for a grid-1 solve."""
    out = rels[(rels["pa"] >= 0) & (rels["pb"] >= 0) & rels["a"].isin(shapes) & rels["b"].isin(shapes)].copy()
    sga = np.array([shapes[t][0].ravel()[p] for t, p in zip(out["a"], out["pa"])])
    sca = np.array([shapes[t][1].ravel()[p] for t, p in zip(out["a"], out["pa"])])
    sgb = np.array([shapes[t][0].ravel()[p] for t, p in zip(out["b"], out["pb"])])
    scb = np.array([shapes[t][1].ravel()[p] for t, p in zip(out["b"], out["pb"])])
    gain, off, m = out["gain"].to_numpy(), out["offset"].to_numpy(), out["centre"].to_numpy()
    g2 = sga * gain / sgb
    # u_a = R + sca + sga (gain v_b + off - R), v_b = R + (u_b - R - scb) / sgb
    off2 = R + sca + sga * (gain * R + off - R) - g2 * (R + scb)
    out["gain"], out["offset"], out["centre"] = g2, off2, sgb * (m - R) + R + scb
    out["pa"] = out["pb"] = 0
    return out


def compose(shape, level):
    """Patch maps of a shape (sg, sc) followed by a tile level (g_t, c_t) (grid-1 arrays)."""
    (sg, sc), (gt, ct) = shape, level
    return gt[0, 0] * sg, gt[0, 0] * sc + ct[0, 0]


# ----- windows and diagnostics -----------------------------------------------------------

def common_window(maps, win_z, lo_col="lo", hi_col="hi"):
    """(C_lo, C_hi): median over tiles of each tile's mean map applied to its own window ends."""
    lo, hi = [], []
    for r in win_z.itertuples():
        if r.tile not in maps:
            continue
        g, c = maps[r.tile]
        mg, mc = g.mean(), c.mean()
        lo.append(mg * (getattr(r, lo_col) - R) + R + mc)
        hi.append(mg * (getattr(r, hi_col) - R) + R + mc)
    return float(np.median(lo)), float(np.median(hi))


def predicted_steps(rels, maps, window, shapes_hw, levels_z=None):
    """Per bin relation: the step at its centre after the maps (greys; a minus b), and with the
    per-tile windows ``levels_z`` {tile: (lo, hi)} (the current pipeline) when given."""
    rels = rels[(rels["pa"] >= 0) & rels["a"].isin(maps) & rels["b"].isin(maps)].copy()
    span = window[1] - window[0]
    after = []
    for r in rels.itertuples():
        (ha, wa), (hb, wb) = shapes_hw[r.a], shapes_hw[r.b]
        ga, ca = (float(interp_grid(v, r.xa, r.ya, wa, ha)) for v in maps[r.a])
        gb, cb = (float(interp_grid(v, r.xb, r.yb, wb, hb)) for v in maps[r.b])
        va = r.gain * r.centre + r.offset
        after.append(((ga * (va - R) + ca) - (gb * (r.centre - R) + cb)) / span * 255)
    rels["after"] = after
    if levels_z is not None:
        va = rels["gain"] * rels["centre"] + rels["offset"]
        la = np.array([levels_z[t] for t in rels["a"]])
        lb = np.array([levels_z[t] for t in rels["b"]])
        rels["before"] = ((va - la[:, 0]) / (la[:, 1] - la[:, 0]) - (rels["centre"] - lb[:, 0]) / (lb[:, 1] - lb[:, 0])) * 255
    return rels


# ----- driver ----------------------------------------------------------------------------

def read_zs(text):
    text = Path(text[1:]).read_text() if text.startswith("@") else text
    return sorted({int(v) for v in text.replace("\n", ",").split(",") if v.strip()})


def load_bins(bins_dir, z):
    path = Path(bins_dir) / f"z{z}.csv"
    return pd.read_csv(path, dtype={"a": str, "b": str}) if path.exists() else None


def solve_variant(args, zs, wins):
    """{z: {tile: (g, c)}} for the requested variant."""
    kw = dict(grid=args.grid, smooth=args.smooth, prior=args.prior, gain_power=args.gain_power, smooth_c=args.smooth_c,
              patch_prior=args.patch_prior, prior_c=args.prior_c)
    if args.variant == "A":
        out = {}
        for z in zs:
            rels = load_bins(args.bins, z)
            if rels is not None:
                out[z] = solve_patches(rels, **kw)
        return out
    # B: static shapes per (segment, tile) from variant-A solves of the shape slices
    seg = wins.groupby("z")["segment"].first()
    acc = {}
    for z in read_zs(args.shape_zs):
        rels = load_bins(args.bins, z)
        if rels is None or z not in seg.index:
            continue
        for t, (g, c) in solve_patches(rels, **kw).items():
            acc.setdefault((int(seg[z]), t), []).append((g / g.mean(), c - c.mean()))
    shapes = {key: (np.median([s[0] for s in v], axis=0), np.median([s[1] for s in v], axis=0))
              for key, v in acc.items()}
    args.shapes = shapes
    out = {}
    for z in zs:
        rels = load_bins(args.bins, z)
        if rels is None:
            continue
        sh = {t: s for (sg, t), s in shapes.items() if sg == int(seg[z])}
        lev = solve_patches(shape_corrected(rels, sh, args.grid), grid=1, smooth=0.0, prior=args.prior,
                            gain_power=args.gain_power)
        out[z] = {t: compose(sh[t], lev[t]) for t in lev}
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("bins")
    p.add_argument("windows", help="tile_windows.csv from janelia_windows.py")
    p.add_argument("zs")
    p.add_argument("out")
    p.add_argument("--variant", choices=["A", "B"], default="A")
    p.add_argument("--shape-zs", help="B: slices whose variant-A solves give the static shapes")
    p.add_argument("--grid", type=int, default=8)
    p.add_argument("--smooth", type=float, default=0.3)
    p.add_argument("--smooth-c", type=float)
    p.add_argument("--prior", type=float, default=0.02)
    p.add_argument("--prior-c", type=float, default=0.0)
    p.add_argument("--gain-power", type=float, default=2.0)
    p.add_argument("--patch-prior", type=float, default=0.0)
    p.add_argument("--jump", action="store_true", help="windows with the per-slice jump (lo_s, hi_s)")
    p.add_argument("--levels", help="levels.csv of the current pipeline, for the 'before' predicted steps")
    p.add_argument("--slices", help="slices.csv (tile sizes); default: from the bins' positions")
    args = p.parse_args(argv)
    zs = read_zs(args.zs)
    wins = pd.read_csv(args.windows, dtype={"tile": str})
    sizes = pd.read_csv(args.slices, dtype={"tile": str}, usecols=["z", "tile", "height", "width"]) if args.slices else None
    levels = pd.read_csv(args.levels, dtype={"tile": str}) if args.levels else None
    maps_by_z = solve_variant(args, zs, wins)
    lo_col, hi_col = ("lo_s", "hi_s") if args.jump else ("lo", "hi")
    map_rows_out, win_rows, resid = [], [], []
    for z, maps in sorted(maps_by_z.items()):
        wz = wins[wins["z"] == z]
        C = common_window(maps, wz, lo_col, hi_col)
        win_rows.append({"z": z, "lo": C[0], "hi": C[1]})
        for t, (g, c) in maps.items():
            for q, (gv, cv) in enumerate(zip(g.ravel(), c.ravel())):
                map_rows_out.append({"z": z, "tile": t, "p": q, "g": gv, "c": cv})
        if sizes is not None:
            sz = sizes[sizes["z"] == z]
            hw = {r.tile: (r.height, r.width) for r in sz.itertuples()}
            lz = None
            if levels is not None:
                lv = levels[levels["z"] == z]
                lz = {r.tile: (r.lo, r.hi) for r in lv.itertuples()}
            resid.append(predicted_steps(load_bins(args.bins, z), maps, C, hw, lz))
    pd.DataFrame(map_rows_out).to_csv(f"{args.out}_maps.csv", index=False, float_format="%.6g")
    pd.DataFrame(win_rows).to_csv(f"{args.out}_windows.csv", index=False, float_format="%.3f")
    if resid:
        pd.concat(resid).to_csv(f"{args.out}_resid.csv", index=False, float_format="%.4g")
    if args.variant == "B":
        rows = [{"segment": s, "tile": t, "p": q, "sg": a, "sc": b}
                for (s, t), (sg, sc) in args.shapes.items() for q, (a, b) in enumerate(zip(sg.ravel(), sc.ravel()))]
        pd.DataFrame(rows).to_csv(f"{args.out}_shapes.csv", index=False, float_format="%.6g")
    print(f"{len(maps_by_z)} slices solved -> {args.out}_maps.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
