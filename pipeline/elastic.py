"""Smooth (elastic) correction at problem interfaces between consecutive slices, run at the end of
align solve.

Where the slices on either side of an interface differ by a distortion that no transform per slice
or per tile captures (35h: the 3x3 -> 2x2 tile layout change, ~20 px in the middle of the field,
most likely scan distortion along the much longer 2x2 scan lines), align's matches across the
interface give a displacement at each matched point. A smooth field on a regular grid
(``align.elastic_grid_px`` apart, differences between neighbouring grid points penalised by
``align.elastic_smoothness``) is fitted to them. The distortion is a property of how each side was
imaged, so the whole field is applied to every slice of one side (``align.elastic_side``: before,
the earlier tile layout, or after), as a constant correction: that side is warped into the other
side's geometry and the other side is untouched. A side runs from the interface to the end of its
tile layout run or to the next corrected interface. Render warps each of those slices after
placing and blending its tiles, so seams within a slice stay intact.

Interfaces: every tile layout change whose frame fit (align/segment_changes.csv) leaves a median
mismatch above ``align.elastic_min_px``, and the first z after each mismatch listed in
``align.elastic_interfaces``. Writes align/elastic.json (fields and per-slice weights, read by
render) and align/elastic_interfaces.csv (mismatch over all matched points before and after).
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

from . import cli, transforms
from .config import step_dir, step_path

log = logging.getLogger(__name__)

REPORT_COLUMNS = ["z_before", "z_after", "reason", "points", "median_before_px", "p90_before_px",
                  "median_after_px", "p90_after_px", "max_field_px", "side", "z_first", "z_last"]


def interfaces(zs, run, changes, a):
    """[(position in zs of the first slice after the interface, reason)], sorted, without duplicates."""
    out = {}
    lim = a["elastic_min_px"]
    if lim is not None and len(changes):
        for _, c in changes.iterrows():
            if np.isfinite(c["median_px"]) and c["median_px"] > float(lim):
                out[int(np.searchsorted(zs, c["z_after"]))] = (
                    f"layout change z {c['z_before']}->{c['z_after']} (median {c['median_px']:.1f} px "
                    f"after the frame fit)")
    for z in a["elastic_interfaces"] or []:
        k = int(np.searchsorted(zs, z))
        if 0 < k < len(zs):
            out.setdefault(k, f"configured z {zs[k - 1]}->{zs[k]}")
    return sorted(out.items())


def weights(run, k, ks, side):
    """{position: weight} of the field of the interface at position k (first slice after it).

    The field moves the content after the interface onto the content before it. side "after":
    weight +1 for the slices from k to the end of k's tile layout run or the next interface in ks;
    "before": -1 (moving the earlier content the other way) for the slices back to the start of
    k - 1's run or the previous interface.
    """
    run = np.asarray(run)
    if side == "after":
        hi = min([len(run)] + [j for j in ks if j > k] + [int(np.flatnonzero(run == run[k])[-1]) + 1])
        return {i: 1.0 for i in range(k, hi)}
    lo = max([0] + [j for j in ks if j < k] + [int(np.flatnonzero(run == run[k - 1])[0])])
    return {i: -1.0 for i in range(lo, k)}


def fit_field(p, d, spacing, smoothness):
    """Bilinear grid (spacing px) fitted to displacements d at points p, with a membrane penalty
    smoothness * (points per grid cell) on differences between neighbouring grid values.
    Returns the field as a dict (x0, y0, spacing, g: (ny, nx, 2) list)."""
    x0, y0 = np.floor(p.min(0)) - spacing / 2
    nx = int(np.ceil((p[:, 0].max() - x0) / spacing)) + 2
    ny = int(np.ceil((p[:, 1].max() - y0) / spacing)) + 2
    W = _bilinear(p, x0, y0, spacing, nx, ny)
    idx = np.arange(nx * ny).reshape(ny, nx)
    pairs = np.r_[np.column_stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()]),
                  np.column_stack([idx[:-1, :].ravel(), idx[1:, :].ravel()])]
    m = len(pairs)
    D = sparse.csr_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[np.arange(m), np.arange(m)], pairs.T.ravel())),
                          shape=(m, nx * ny))
    lam = float(smoothness) * max(len(p) / (nx * ny), 1.0)
    A = (W.T @ W + lam * (D.T @ D) + 1e-9 * sparse.eye(nx * ny)).tocsc()
    g = np.column_stack([spsolve(A, W.T @ d[:, k]) for k in (0, 1)]).reshape(ny, nx, 2)
    return {"x0": float(x0), "y0": float(y0), "spacing": float(spacing), "g": g.tolist()}


def _bilinear(p, x0, y0, spacing, nx, ny):
    u = np.clip((p[:, 0] - x0) / spacing, 0, nx - 1 - 1e-9)
    v = np.clip((p[:, 1] - y0) / spacing, 0, ny - 1 - 1e-9)
    i, j = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = u - i, v - j
    rows = np.repeat(np.arange(len(p)), 4)
    cols = np.column_stack([j * nx + i, j * nx + i + 1, (j + 1) * nx + i, (j + 1) * nx + i + 1]).ravel()
    vals = np.column_stack([(1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv]).ravel()
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(p), nx * ny))


def field_at(field, p):
    """Displacement (n, 2) of a field at points p (n, 2); constant beyond the grid's edge."""
    g = np.asarray(field["g"], float)
    ny, nx, _ = g.shape
    W = _bilinear(np.asarray(p, float).reshape(-1, 2), field["x0"], field["y0"], field["spacing"], nx, ny)
    return W @ g.reshape(-1, 2)


def solve(cfg, df, zs, mats, run, changes, stitch, load_tile_matches):
    """Fit every interface's field; write align/elastic.json and align/elastic_interfaces.csv."""
    a = cfg["align"]
    nb = int(a["neighbors"])
    fields, per_z, reports = [], {}, []
    pos = {int(z): i for i, z in enumerate(zs)}
    found = interfaces(zs, run, changes, a)
    ks = [k for k, _ in found]
    for k, reason in found:
        z_old, z_new = int(zs[k - 1]), int(zs[k])
        m = load_tile_matches(int(zs[max(0, k - nb)]), int(zs[min(len(zs) - 1, k + nb - 1)]))
        x = (m["za"] <= z_old) & (m["zb"] >= z_new)
        rep = {"z_before": z_old, "z_after": z_new, "reason": reason, "points": int(x.sum())}
        if x.sum() < 50:
            log.warning("interface z %d->%d: only %d points matched across it; no elastic correction",
                        z_old, z_new, int(x.sum()))
            reports.append(rep)
            continue

        def place(zz, tt, qq):
            return np.array([transforms.apply(transforms.compose(mats[pos[int(z)]], stitch[(int(z), t)]), q)[0]
                             for z, t, q in zip(zz, tt, qq)])
        pa, pb = place(m["za"][x], m["ta"][x], m["qa"][x]), place(m["zb"][x], m["tb"][x], m["qb"][x])
        d = pa - pb   # where the new side's content must move to land on the old side's
        field = fit_field(pb, d, float(a["elastic_grid_px"]), float(a["elastic_smoothness"]))
        before = np.linalg.norm(d, axis=1)
        after = np.linalg.norm(d - field_at(field, pb), axis=1)
        w = weights(run, k, ks, a["elastic_side"])
        for i, wt in w.items():
            per_z.setdefault(int(zs[i]), []).append([len(fields), wt])
        fields.append(field)
        rep.update(median_before_px=float(np.median(before)), p90_before_px=float(np.percentile(before, 90)),
                   median_after_px=float(np.median(after)), p90_after_px=float(np.percentile(after, 90)),
                   max_field_px=float(np.linalg.norm(np.asarray(field["g"]), axis=2).max()),
                   side=a["elastic_side"], z_first=int(zs[min(w)]), z_last=int(zs[max(w)]))
        log.info("elastic correction at z %d->%d (%s): %d points, mismatch median %.2f -> %.2f px, p90 %.2f -> "
                 "%.2f px (all points); field up to %.1f px, applied to z %d-%d", z_old, z_new, reason,
                 rep["points"], rep["median_before_px"], rep["median_after_px"], rep["p90_before_px"],
                 rep["p90_after_px"], rep["max_field_px"], rep["z_first"], rep["z_last"])
        reports.append(rep)
    out = {"fields": fields, "weights": {str(z): v for z, v in sorted(per_z.items())}}
    cli.atomic_write(step_dir(cfg, "align", "elastic.json"), lambda tmp: Path(tmp).write_text(json.dumps(out)))
    rep = pd.DataFrame(reports, columns=REPORT_COLUMNS)
    cli.atomic_write(step_dir(cfg, "align", "elastic_interfaces.csv"), lambda tmp: rep.to_csv(tmp, index=False))
    return out, rep


def load(cfg):
    """align/elastic.json as written by solve, or None if there is none or it has no fields."""
    path = step_path(cfg, "align", "elastic.json")
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data if data.get("fields") else None


def displacement(elastic, z, xs, ys):
    """Displacement (dx, dy) of slice z's content at the grid of points xs (columns) x ys (rows),
    summed over the fields that apply to z; None if none does."""
    entries = elastic["weights"].get(str(int(z)))
    if not entries:
        return None
    X, Y = np.meshgrid(np.asarray(xs, float), np.asarray(ys, float))
    p = np.column_stack([X.ravel(), Y.ravel()])
    D = np.zeros_like(p)
    for i, w in entries:
        D += w * field_at(elastic["fields"][int(i)], p)
    return D[:, 0].reshape(X.shape), D[:, 1].reshape(X.shape)
