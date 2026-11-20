"""Offline variants of the z 2083 tile-solve window: per-tile match residuals and per-seam residuals."""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import sys

import numpy as np

from pipeline import align, tilesolve, transforms
from pipeline.config import load_config, step_path
from pipeline.slices import load_slices, StackCache

cfg = load_config(os.path.expandvars("configs/lab/${DEV_PREFIX}37i.yaml"), step_defaults=align.DEFAULTS)
a0 = cfg["align"]
df = load_slices(cfg)
zs = np.array(sorted(df["z"].unique()))
mats = transforms.read_csv(step_path(cfg, "align", "transforms.csv"), ["z"])
stitch = transforms.read_csv(step_path(cfg, "stitch", "tiles.csv"), ["z", "tile"])
inside = {2083}
around = [z for z in range(2079, 2088)]
sub = df[df["z"].isin(around)]
shapes = {(z, t): (int(h), int(w)) for z, t, h, w in zip(sub["z"], sub["tile"], sub["height"], sub["width"])}
T0 = {k: transforms.compose(mats[k[0]], stitch[k]) for k in shapes}
m = align._tile_matches(cfg, df, zs, a0, 2079, 2087)
sel = (np.isin(m["za"], list(inside)) | np.isin(m["zb"], list(inside)))
cut = tilesolve.crosses(m["za"], m["zb"], zs, [int(np.searchsorted(zs, 2083))])
m = {k: v[sel & ~cut] for k, v in m.items()}
print("matches:", len(m["za"]), "pairs:", sorted(set(zip(m["za"].tolist(), m["zb"].tolist()))), file=sys.stderr)

rz = {r["tile"]: r for r in df[df["z"] == 2083].to_dict("records")}
cache = StackCache(cfg["raw_dir"])
tz = {t: T0[(2083, t)] for t in rz}
seams = tilesolve.matched_seams(lambda t, rr, cc: cache.read(rz[t], rr, cc), tz, {t: shapes[(2083, t)] for t in tz},
                                tilesolve.seam_options(cfg))
matched = {2083: seams}
for ta, qa, tb, qb in seams:
    d = transforms.apply(tz[ta], qa) - transforms.apply(tz[tb], qb)
    print(f"matched seam {ta}|{tb}: {len(qa)} points; current transforms disagree there by median "
          f"({np.median(d[:, 0]):.1f}, {np.median(d[:, 1]):.1f}) px, spread x {np.percentile(d[:, 0], 5):.1f}..{np.percentile(d[:, 0], 95):.1f}"
          f" y {np.percentile(d[:, 1], 5):.1f}..{np.percentile(d[:, 1], 95):.1f}; along: "
          f"{np.round(np.percentile(transforms.apply(tz[ta], qa), [0, 100], axis=0).ravel(), 0).tolist()}", file=sys.stderr)


def per_tile(out, warps):
    """Residuals of the cross-slice matches per tile of z 2083 under the solution."""
    def place(z, t, q):
        T = out.get((int(z), t), T0[(int(z), t)])
        p = transforms.apply(T, q)
        w = warps.get((int(z), t))
        return p + (tilesolve.warp_at(w, q) if w is not None else 0)
    res = {}
    for za, ta, qa, zb, tb, qb in zip(m["za"], m["ta"], m["qa"], m["zb"], m["tb"], m["qb"]):
        tile = ta if za == 2083 else tb
        e = np.linalg.norm(place(za, ta, qa[None])[0] - place(zb, tb, qb[None])[0])
        res.setdefault(tile, []).append(e)
    return {t: (round(float(np.median(v)), 1), round(float(np.percentile(v, 90)), 1)) for t, v in sorted(res.items())}


def seam_res(out, warps, seams):
    r = {}
    for ta, qa, tb, qb in seams:
        def place(t, q):
            p = transforms.apply(out[(2083, t)], q)
            w = warps.get((2083, t))
            return p + (tilesolve.warp_at(w, q) if w is not None else 0)
        d = np.linalg.norm(place(ta, qa) - place(tb, qb), axis=1)
        r[f"{ta}|{tb}"] = (round(float(np.median(d)), 1), round(float(d.max()), 1))
    return r


print("\nbefore (current transforms): per tile median/p90", per_tile({}, {}), file=sys.stderr)
for label, opts, mt in [("affine, current seams 64", {}, None),
                        ("warp 2000, current seams 64", {"local_warp_px": 2000}, None),
                        ("warp 2000, matched seams 64", {"local_warp_px": 2000}, matched),
                        ("warp 2000, matched seams 4", {"local_warp_px": 2000, "local_seam_weight": 4.0}, matched),
                        ("warp 2000, matched seams 1", {"local_warp_px": 2000, "local_seam_weight": 1.0}, matched),
                        ("warp 2000, current seams 4", {"local_warp_px": 2000, "local_seam_weight": 4.0}, None),
                        ("warp 1000, matched seams 4", {"local_warp_px": 1000, "local_seam_weight": 4.0}, matched),
                        ("affine, matched seams 4", {"local_seam_weight": 4.0}, matched),
                        ("affine, matched seams 64", {}, matched)]:
    a = {**a0, **opts}
    out, rep, _, warps = tilesolve.solve_window(T0, shapes, m, inside, a, mt)
    nonaff = {}
    for k, f in warps.items():
        g = np.asarray(f["g"]); ny, nx, _ = g.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        X = np.column_stack([xx.ravel(), yy.ravel(), np.ones(nx * ny)])
        coef, *_ = np.linalg.lstsq(X, g.reshape(-1, 2), rcond=None)
        nonaff[k[1]] = round(float(np.linalg.norm(g.reshape(-1, 2) - X @ coef, axis=1).max()), 1)
    print(f"\n{label}: all median {rep['median_after_px']:.2f} p90 {rep['p90_after_px']:.2f}, rejected {rep['rejected_points']}\n"
          f"   per tile (median, p90): {per_tile(out, warps)}\n"
          f"   matched seams under it (median, max): {seam_res(out, warps, seams)}\n"
          f"   warp non-affine max per tile: {nonaff}", file=sys.stderr)
