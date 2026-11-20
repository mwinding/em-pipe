"""Compare stitch solvers on the real point matches dumped from the cluster."""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import sys
import numpy as np

sys.path.insert(0, os.path.expandvars("${EM_PIPE_ROOT}"))
from pipeline import stitch, transforms  # noqa: E402


def solve_joint(pairs, tiles, reference, lam):
    """Affine per tile, regularised toward rigid in one linear least-squares solve."""
    free = [t for t in tiles if t != reference]
    col = {t: 6 * i for i, t in enumerate(free)}
    n = 6 * len(free)
    rows, rhs = [], []
    npts = {t: 0 for t in free}
    spread = {t: [] for t in free}
    for (ta, tb), (pa, pb) in pairs.items():
        (Ja, ca), (Jb, cb) = stitch._linear("affine", pa), stitch._linear("affine", pb)
        M = np.zeros((len(pa), 2, n))
        r = cb - ca
        for t, J, p, sign in ((ta, Ja, pa, 1), (tb, Jb, pb, -1)):
            if t == reference:
                r = r - sign * (J @ np.array([1, 0, 0, 0, 1, 0], float))
            else:
                M[:, :, col[t]:col[t] + 6] += sign * J
                npts[t] += len(p)
                spread[t].append(p)
        rows.append(M.reshape(-1, n))
        rhs.append(r.ravel())
    if lam > 0:
        w = lam / max(1 - lam, 1e-9)
        for t in free:
            P = np.vstack(spread[t])
            r2 = float(np.mean(np.sum((P - P.mean(0)) ** 2, axis=1)))   # px^2: a unit change in L moves points by ~r
            s = np.sqrt(w * npts[t] * r2)
            a, b, tx, c, d, ty = range(col[t], col[t] + 6)
            for coeffs, target in (({a: 1, d: -1}, 0.0), ({b: 1, c: 1}, 0.0), ({a: 1, d: 1}, 2.0)):
                row = np.zeros(n)
                for k, v in coeffs.items():
                    row[k] = v * s
                rows.append(row[None])
                rhs.append(np.array([target * s]))
    q = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)[0]
    T = {reference: transforms.identity()}
    T.update({t: q[col[t]:col[t] + 6].reshape(2, 3) for t in free})
    return T, stitch.residuals(T, pairs)


def report(name, T, pairs, tiles):
    err = np.concatenate([np.linalg.norm(transforms.apply(T[a], pa) - transforms.apply(T[b], pb), axis=1)
                          for (a, b), (pa, pb) in pairs.items()])
    rms = stitch.residuals(T, pairs)
    lin = max(np.abs(T[t][:, :2] - np.eye(2)).max() for t in tiles)
    scale = max(abs(np.sqrt(abs(np.linalg.det(T[t][:, :2]))) - 1) for t in tiles)
    print(f"  {name:22s} median {np.median(err):5.2f} p95 {np.percentile(err, 95):6.2f}  "
          f"pair rms {sorted(round(v, 1) for v in rms.values())}  max|L-I| {lin:.4f} max|scale-1| {scale:.4f}")


d = np.load(sys.argv[1])
for z in (7926, 8593, 9136):
    for pm in ("translation", "affine"):
        pairs = {}
        for k in d.files:
            if k.startswith(f"z{z}_{pm}_") and k.endswith("_pa"):
                _, _, a, b, _ = k.split("_")
                pairs[(a, b)] = (d[k], d[k[:-3] + "_pb"])
        tiles = sorted({t for p in pairs for t in p})
        print(f"z {z}, pairs filtered with {pm} RANSAC: {[(p, len(v[0])) for p, v in pairs.items()]}")
        report("translation", stitch._solve_linear(pairs, tiles, "0-0", "translation")[0], pairs, tiles)
        report("rigid", stitch._solve_linear(pairs, tiles, "0-0", "rigid")[0], pairs, tiles)
        report("affine (free)", stitch._solve_linear(pairs, tiles, "0-0", "affine")[0], pairs, tiles)
        report("iterative λ=.1 (old)", stitch.solve_affine_rigid(pairs, tiles, "0-0", [1.0, 0.5, 0.1])[0], pairs, tiles)
        for lam in (0.9, 0.5, 0.1, 0.01):
            report(f"joint λ={lam}", solve_joint(pairs, tiles, "0-0", lam)[0], pairs, tiles)
