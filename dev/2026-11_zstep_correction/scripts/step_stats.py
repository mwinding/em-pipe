"""Statistics of the slice spacing the zcorrect step estimated, and how far independent areas agree on it.

    python dev/2026-11_zstep_correction/scripts/step_stats.py --config CONFIG [--window 64] [--top 8] [--tag TAG]

Reads ``work/zcorrect/positions.csv`` and ``ncc/`` (zcorrect run + solve), ``work/check/slices.csv`` and, when
present, ``qc/zstep_correction/tile_ncc/`` (tile_ncc.py). Prints, and writes to ``qc/zstep_correction/``:

- ``step_stats.json``: the spacing's distribution (nm; breaks left out, where zcorrect holds the spacing at
  nominal), counts of near-duplicate (< 0.5x) and thick (> 1.5x, > 2x) steps, the lag-1 autocorrelation of the
  spacing, the adjacent-slice NCC, a model with a factor per slice next to the spacing (is a low NCC a thick
  cut or a slice that looks different?), the most uneven ``window``-slice stretches and a typical one, the breaks with
  the cut across them read from the NCC of the slices before, and the agreement between tiles;
- ``spacing.csv``: per slice z, spacing_nm (from the previous slice), break, ncc1 (NCC with the previous slice)
  and, with tile_ncc, the spacing solved from each tile alone.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pipeline import zcorrect  # noqa: E402
from pipeline.cli import base_parser, setup  # noqa: E402
from pipeline.config import qc_path, step_path  # noqa: E402
from pipeline.slices import load_slices, voxel_size_nm, z_values  # noqa: E402


def solve_pairs(zs, brk, ia, ib, ncc, zc, full=False):
    """Spacing / nominal per step, as pipeline.zcorrect.solve computes it (same functions, same loop)."""
    nominal = np.diff(zs).astype(float)
    block = np.concatenate([[0], np.cumsum(brk)])
    keep = (block[ia] == block[ib]) & np.isfinite(ncc)
    ia, ib, ncc = ia[keep], ib[keep], ncc[keep]
    d_nominal = (zs[ib] - zs[ia]).astype(float)
    s, adjacent = nominal.copy(), ib - ia == 1
    for _ in range(max(1, zc["iterations"])):
        pos = np.concatenate([[0.0], np.cumsum(s)])
        xs, ys = zcorrect._fit_curve(pos[ib] - pos[ia], ncc)
        dist, slope = zcorrect._invert(xs, ys, ncc)
        w = slope ** 2 / d_nominal
        if (adjacent & (w > 0)).any():
            w = w / np.median(w[adjacent & (w > 0)])
        s = zcorrect._solve_spacings(ia, ib, dist, w, nominal, zc["regularization"])
        s = zcorrect._constrain(s, nominal, brk, zc["min_spacing_fraction"], zc["nominal_window"])
    covered = np.zeros(len(nominal), bool)
    covered[ia[adjacent]] = True
    if not full:
        return np.where(covered, s / nominal, np.nan)
    pos = np.concatenate([[0.0], np.cumsum(s)])
    xs, ys = zcorrect._fit_curve(pos[ib] - pos[ia], ncc)
    return np.where(covered, s / nominal, np.nan), keep, np.interp(pos[ib] - pos[ia], xs, ys)


def slice_factors(n, ia, ib, resid):
    """Per-slice u with u[a] + u[b] ~ resid for every pair (least squares, lightly damped towards 0)."""
    m = len(ia)
    A = sparse.csr_matrix((np.ones(2 * m), (np.r_[np.arange(m), np.arange(m)], np.r_[ia, ib])), shape=(m, n))
    return spsolve((A.T @ A + 1e-3 * sparse.eye(n)).tocsc(), A.T @ resid)


def quality_model(zs, brk, ia, ib, ncc, zc):
    """Is a low NCC a thick cut, or a slice that looks different (noise, focus, charging)?

    zcorrect reads every NCC as a distance. Hanslovsky et al. also give each section a scaling factor, so a slice
    that correlates poorly with all its neighbours is not read as two thick cuts. Two one-cause models of
    log NCC(a, b), each compared with the per-lag median alone: the spacing zcorrect solved (log f(distance)), and
    a factor per slice at nominal spacing (u_a + u_b + per-lag level). Returns the share of the variance about
    the per-lag median that each explains, and the SD of the slice factors exp(u).
    """
    _, keep, pred_t = solve_pairs(zs, brk, ia, ib, ncc, zc, full=True)
    ia, ib, ncc = ia[keep], ib[keep], ncc[keep]
    good = ncc > 0.02
    ia, ib, ncc, pred_t = ia[good], ib[good], ncc[good], pred_t[good]
    lag, y = ib - ia, np.log(ncc)
    base = np.zeros_like(y)
    for k in np.unique(lag):
        base[lag == k] = np.median(y[lag == k])
    total = float(((y - base) ** 2).sum())

    def share(pred_log):
        r = y - pred_log
        for k in np.unique(lag):                      # each model may keep its own per-lag level
            r[lag == k] -= np.median(r[lag == k])
        return round(1 - float((r ** 2).sum()) / total, 3)

    u = slice_factors(len(zs), ia, ib, y - base)
    return {"variance_explained_by_spacing": share(np.log(np.clip(pred_t, 1e-3, None))),
            "variance_explained_by_slice_factors": share(base + u[ia] + u[ib]),
            "slice_factor_sd": round(float(np.std(np.exp(u))), 4)}


def across(z, lag_ncc, before=60):
    """The cut before slice z, in slices of the ``before`` slices ahead of it: where NCC(z-1, z) sits on the
    curve of median NCC against lag measured in those slices. None when not measured; inf beyond the curve."""
    v = lag_ncc.get((z - 1, z))
    if v is None or not np.isfinite(v):
        return None, None
    lags, med = [], []
    for k in range(1, 9):
        vals = [lag_ncc[(a, a + k)] for a in range(z - before, z - k) if (a, a + k) in lag_ncc]
        vals = [x for x in vals if np.isfinite(x)]
        if len(vals) >= 10:
            lags.append(k)
            med.append(float(np.median(vals)))
    if len(lags) < 2:
        return float(v), None
    med = np.minimum.accumulate(med)
    if v >= med[0]:
        return float(v), float(lags[0] * (1 - v) / max(1 - med[0], 1e-6))
    if v < med[-1]:
        return float(v), float("inf")
    return float(v), float(np.interp(-v, -med, lags))


def main(argv=None):
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--window", type=int, default=64, help="slices per stretch when ranking uneven stretches")
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--tag", default="", help="suffix of the output names (step_stats_TAG.json, spacing_TAG.csv)")
    args = p.parse_args(argv)
    cfg = setup(args, zcorrect.DEFAULTS)
    zc = cfg["zcorrect"]
    df = load_slices(cfg)
    zs = np.array(z_values(df))
    vz = voxel_size_nm(cfg, df["file"].unique())[0]
    pos = pd.read_csv(step_path(cfg, "zcorrect", "positions.csv")).set_index("z")["position_nm"].reindex(zs).to_numpy()
    per_z = df.groupby("z").agg(seam=("seam", "any"), segment=("segment", "first")).reindex(zs)
    brk = per_z["seam"].to_numpy()[1:] | (np.diff(per_z["segment"].to_numpy()) != 0)
    nominal = np.diff(zs).astype(float)
    ratio = np.diff(pos) / (vz * nominal)                    # spacing / nominal, step i: zs[i] -> zs[i + 1]
    ok = ~brk
    pairs = zcorrect._load_pairs(cfg, zs, zc)
    lag_ncc = {(int(a), int(b)): float(v) for a, b, v in zip(pairs["z_a"], pairs["z_b"], pairs["ncc"])}
    ncc1 = np.array([lag_ncc.get((int(a), int(b)), np.nan) for a, b in zip(zs[:-1], zs[1:])])

    pia, pib = np.searchsorted(zs, pairs["z_a"]), np.searchsorted(zs, pairs["z_b"])
    qual = quality_model(zs, brk, pia, pib, pairs["ncc"].to_numpy(float), zc)
    r, nm = ratio[ok], ratio[ok] * vz
    dev = np.where(ok, ratio - 1, 0.0)
    out = {"slices": int(len(zs)), "voxel_z_nm": float(vz), "breaks": int(brk.sum()),
           "spacing_nm": {k: round(float(v), 3) for k, v in
                          [("median", np.median(nm)), ("mean", nm.mean()), ("sd", nm.std()), ("min", nm.min()),
                           ("p1", np.percentile(nm, 1)), ("p5", np.percentile(nm, 5)), ("p95", np.percentile(nm, 95)),
                           ("p99", np.percentile(nm, 99)), ("max", nm.max())]},
           "steps_below_0.5x": int((r < 0.5).sum()), "steps_at_minimum": int((r <= zc["min_spacing_fraction"] + 1e-3).sum()),
           "steps_above_1.5x": int((r > 1.5).sum()), "steps_above_2x": int((r > 2).sum()),
           "steps_within_10pct": round(float((np.abs(r - 1) <= 0.1).mean()), 4),
           "steps_within_20pct": round(float((np.abs(r - 1) <= 0.2).mean()), 4),
           "lag1_autocorrelation": round(float(np.corrcoef(dev[:-1][ok[:-1] & ok[1:]], dev[1:][ok[:-1] & ok[1:]])[0, 1]), 3),
           "ncc_adjacent": {"median": round(float(np.nanmedian(ncc1)), 4), "p5": round(float(np.nanpercentile(ncc1, 5)), 4),
                            "p95": round(float(np.nanpercentile(ncc1, 95)), 4)}}
    # Displacement: how far a slice sits from where the uncorrected volume puts it (z * voxel), in nm.
    disp = pos - vz * zs
    out["displacement_nm"] = {"abs_p50": round(float(np.percentile(np.abs(disp), 50)), 2),
                              "abs_p95": round(float(np.percentile(np.abs(disp), 95)), 2),
                              "abs_max": round(float(np.abs(disp).max()), 2), "at_z": int(zs[np.abs(disp).argmax()])}

    # Stretches of `window` slices ranked by the SD of spacing / nominal; non-overlapping picks.
    w = args.window
    sd = pd.Series(np.where(ok, ratio, np.nan)).rolling(w, center=True, min_periods=w // 2).std().to_numpy()
    order, picked = np.argsort(-np.nan_to_num(sd, nan=-1)), []
    for i in order:
        if len(picked) >= args.top:
            break
        if all(abs(i - j) >= w for j in picked):
            picked.append(int(i))

    def stretch(i):
        a, b = max(0, i - w // 2), min(len(ratio), i + w // 2)
        seg = ratio[a:b][ok[a:b]]
        return {"z_centre": int(zs[i]), "z_range": [int(zs[a]), int(zs[b])], "sd_ratio": round(float(sd[i]), 3),
                "min_nm": round(float(seg.min() * vz), 2), "max_nm": round(float(seg.max() * vz), 2),
                "disp_range_nm": round(float(np.ptp(disp[a:b + 1] - disp[a])), 1)}

    out["most_uneven_stretches"] = [stretch(i) for i in picked]
    finite = np.where(np.isfinite(sd))[0]
    typical = finite[np.argsort(np.abs(sd[finite] - np.nanmedian(sd)))]
    mid = [i for i in typical[:200] if abs(i - len(zs) // 2) < len(zs) // 4]
    out["typical_stretch"] = stretch(int(mid[0] if mid else typical[0]))
    out["stretch_sd_ratio"] = {"median": round(float(np.nanmedian(sd)), 3), "p95": round(float(np.nanpercentile(sd, 95)), 3)}

    out["slice_quality_model"] = qual
    out["breaks_cut"] = []
    for i in np.where(brk)[0]:
        z = int(zs[i + 1])
        v, d = across(z, lag_ncc)
        out["breaks_cut"].append({"z": z, "segment_change": bool(per_z["segment"].iloc[i] != per_z["segment"].iloc[i + 1]),
                                  "ncc_across": None if v is None else round(v, 4),
                                  "like_slices": None if d is None else (">8" if np.isinf(d) else round(d, 2))})

    table = pd.DataFrame({"z": zs[1:], "spacing_nm": ratio * vz, "break": brk, "ncc1": ncc1})
    tdir = qc_path(cfg, "zstep_correction") / "tile_ncc"
    files = sorted(tdir.glob("chunk_*.npz")) if tdir.exists() else []
    if files:
        z_a, z_b, ncc_t, tiles = [], [], [], None
        for f in files:
            with np.load(f) as d:
                z_a.append(d["z_a"]), z_b.append(d["z_b"]), ncc_t.append(d["ncc"])
                tiles = [str(t) for t in d["tiles"]]
        z_a, z_b, ncc_t = np.concatenate(z_a), np.concatenate(z_b), np.vstack(ncc_t)
        ia, ib = np.searchsorted(zs, z_a), np.searchsorted(zs, z_b)
        solved = {t: solve_pairs(zs, brk, ia, ib, ncc_t[:, k], zc) for k, t in enumerate(tiles)}
        for t in tiles:
            table[f"tile_{t}_nm"] = solved[t] * vz
        halves = [np.nanmedian(ncc_t[:, k::2], axis=1) for k in (0, 1)] if len(tiles) >= 4 else None

        def corr(a, b, smooth=1):
            m = ok & np.isfinite(a) & np.isfinite(b)
            a, b = (pd.Series(np.where(m, v, np.nan)).rolling(smooth, center=True, min_periods=1).mean().to_numpy() for v in (a, b))
            m &= np.isfinite(a) & np.isfinite(b)
            return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 10 else np.nan

        pair_r = [corr(solved[a], solved[b]) for i, a in enumerate(tiles) for b in tiles[i + 1:]]
        pair_r9 = [corr(solved[a], solved[b], 9) for i, a in enumerate(tiles) for b in tiles[i + 1:]]
        sds = [float(np.nanstd(solved[t][ok])) for t in tiles]
        agree = {"tiles": tiles, "tile_sd_ratio": [round(v, 3) for v in sds],
                 "pairwise_r_per_slice": {"median": round(float(np.nanmedian(pair_r)), 3), "min": round(float(np.nanmin(pair_r)), 3),
                                          "max": round(float(np.nanmax(pair_r)), 3)},
                 "pairwise_r_9_slice_mean": {"median": round(float(np.nanmedian(pair_r9)), 3)},
                 "r_tile_vs_all": {t: round(corr(solved[t], ratio), 3) for t in tiles}}
        if halves is not None:
            sa, sb = (solve_pairs(zs, brk, ia, ib, h, zc) for h in halves)
            m = ok & np.isfinite(sa) & np.isfinite(sb)
            agree["split_half"] = {"r_per_slice": round(corr(sa, sb), 3), "r_9_slice_mean": round(corr(sa, sb, 9), 3),
                                   "median_abs_difference_nm": round(float(np.median(np.abs(sa[m] - sb[m])) * vz), 3),
                                   "p95_abs_difference_nm": round(float(np.percentile(np.abs(sa[m] - sb[m]), 95) * vz), 3)}
            table["half_a_nm"], table["half_b_nm"] = sa * vz, sb * vz
            # What the two halves share (thickness, or anything else common to the whole slice): the square
            # root of their covariance; the rest of each half's variance is its own.
            cov = float(np.cov(sa[m], sb[m])[0, 1])
            agree["split_half"]["half_sd_nm"] = [round(float(sa[m].std() * vz), 3), round(float(sb[m].std() * vz), 3)]
            agree["split_half"]["shared_sd_nm"] = round(float(np.sqrt(max(cov, 0.0)) * vz), 3)
        out["tile_agreement"] = agree

    odir = qc_path(cfg, "zstep_correction")
    odir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    for name in (f"spacing{suffix}.csv", f"step_stats{suffix}.json"):
        if (odir / name).exists():
            sys.exit(f"{odir / name} exists: choose another --tag")
    table.round(4).to_csv(odir / f"spacing{suffix}.csv", index=False)
    (odir / f"step_stats{suffix}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
