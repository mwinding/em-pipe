"""Can a smooth displacement field explain the mismatch across a layout change?

Matches across the change are placed with the global solution (align o stitch); the displacement
d = (old-side position) - (new-side position) is fitted by a bilinear grid with a membrane
(neighbour-difference) penalty. Reports the fit over ALL points, per grid spacing and smoothness.
"""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import sys

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

sys.path.insert(0, os.path.expandvars("${EM_PIPE_ROOT}"))
from pipeline import align as al, transforms  # noqa: E402
from pipeline.config import load_config  # noqa: E402

cfg = load_config(sys.argv[1], step_defaults=al.DEFAULTS)
z_old = int(sys.argv[2])
df = al.load_slices(cfg)
zs = np.array(al.z_values(df))
m = al._tile_matches(cfg, df, zs, cfg["align"], z_old - 8, z_old + 8)
stitch = al._stitch_transforms(cfg, df)
T = transforms.read_csv(al.step_path(cfg, "align", "transforms.csv"), ["z"])
x = (m["za"] <= z_old) & (m["zb"] > z_old)
pa = np.array([transforms.apply(transforms.compose(T[z], stitch[(z, t)]), q)[0]
               for z, t, q in zip(m["za"][x], m["ta"][x], m["qa"][x])])
pb = np.array([transforms.apply(transforms.compose(T[z], stitch[(z, t)]), q)[0]
               for z, t, q in zip(m["zb"][x], m["tb"][x], m["qb"][x])])
d = pa - pb
print(f"{len(d)} points; mismatch as now: median {np.median(np.linalg.norm(d, axis=1)):.1f} "
      f"p90 {np.percentile(np.linalg.norm(d, axis=1), 90):.1f} px; field area "
      f"{np.ptp(pb[:, 0]):.0f} x {np.ptp(pb[:, 1]):.0f} px")


def fit(spacing, lam):
    x0, y0 = pb.min(0) - 1
    nx = int(np.ceil((pb[:, 0].max() - x0) / spacing)) + 1
    ny = int(np.ceil((pb[:, 1].max() - y0) / spacing)) + 1
    u, v = (pb[:, 0] - x0) / spacing, (pb[:, 1] - y0) / spacing
    i, j = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = u - i, v - j
    rows = np.repeat(np.arange(len(pb)), 4)
    cols = np.column_stack([j * nx + i, j * nx + i + 1, (j + 1) * nx + i, (j + 1) * nx + i + 1]).ravel()
    vals = np.column_stack([(1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv]).ravel()
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(len(pb), nx * ny))
    # membrane: differences between grid neighbours, weighted by lam * points per cell
    idx = np.arange(nx * ny).reshape(ny, nx)
    pairs = np.r_[np.column_stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()]),
                  np.column_stack([idx[:-1, :].ravel(), idx[1:, :].ravel()])]
    Dm = sparse.csr_matrix((np.r_[np.ones(len(pairs)), -np.ones(len(pairs))],
                            (np.r_[np.arange(len(pairs)), np.arange(len(pairs))], np.r_[pairs[:, 0], pairs[:, 1]])),
                           shape=(len(pairs), nx * ny))
    per_cell = len(pb) / (nx * ny)
    A = (W.T @ W + lam * per_cell * (Dm.T @ Dm) + 1e-9 * sparse.eye(nx * ny)).tocsc()
    G = np.column_stack([spsolve(A, W.T @ d[:, k]) for k in (0, 1)])
    r = np.linalg.norm(d - W @ G, axis=1)
    return r, G, (nx, ny)


for spacing in (2000, 1000, 500):
    for lam in (10.0, 1.0, 0.1):
        r, G, (nx, ny) = fit(spacing, lam)
        mag = np.linalg.norm(G, axis=1)
        print(f"grid {spacing:5d} px ({nx}x{ny}), smoothness {lam:5.1f}: residual median {np.median(r):4.1f} "
              f"p90 {np.percentile(r, 90):4.1f} max {r.max():5.1f} px; field up to {mag.max():5.1f} px", flush=True)
