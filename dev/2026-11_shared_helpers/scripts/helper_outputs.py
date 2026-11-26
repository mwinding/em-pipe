"""Outputs of the functions whose own copy of a small helper was replaced by a shared one on 26 Nov 2026
(robust_sd in pipeline/stats.py, stretch_to_uint8 in pipeline/features.py), on fixed synthetic data, to
compare two checkouts of em-pipe (before and after the change) value for value.

    python dev/2026-11_shared_helpers/scripts/helper_outputs.py REPO OUT.npz       outputs of REPO's code
    python dev/2026-11_shared_helpers/scripts/helper_outputs.py --compare A.npz B.npz

REPO is the em-pipe checkout whose pipeline/ and tools/ are imported (e.g. a git worktree of the commit
before the change); run once per checkout, each in its own process. Values that are only logged
(intensity's slice_level and joint-balance messages) are taken from the log records' arguments, unrounded.
--compare reports, per output, whether both files hold the same array: same dtype, shape and bits.
"""
import argparse
import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_tool(repo, name):
    spec = importlib.util.spec_from_file_location(name, repo / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Records(logging.Handler):
    def __init__(self):
        super().__init__()
        self.args = []

    def emit(self, record):
        self.args.append(record.args)


def stripes(rng, n, amp, vertical=True, noise=4.0):
    """An n x n test window: smooth texture, vertical (or horizontal) stripes of amplitude amp, noise."""
    from scipy import ndimage
    tex = ndimage.gaussian_filter(rng.normal(0, 30, (n, n)), 3)
    prof = amp * np.sin(np.arange(n) * 2 * np.pi / 23.0) + amp * (rng.random(n) - 0.5)
    s = prof[None, :] if vertical else prof[:, None]
    return (1000 + tex + s + rng.normal(0, noise, (n, n))).astype(np.float32)


def textured(rng, h, w, mean, spread, noise):
    """Tissue-like texture: smooth blobs at 16 px scale (mean, spread) plus pixel noise."""
    from scipy import ndimage
    small = ndimage.gaussian_filter(rng.standard_normal((-(-h // 16), -(-w // 16))), 1.0)
    img = np.kron(mean + spread * small / small.std(), np.ones((16, 16)))[:h, :w]
    return img + rng.normal(0, noise, img.shape)


def outputs(repo):
    repo = Path(repo).resolve()
    sys.path.insert(0, str(repo))
    from pipeline import align, curtains, intensity, overlaps, tilesolve, transforms
    rec = Records()
    logging.getLogger("pipeline.intensity").addHandler(rec)
    logging.getLogger("pipeline.intensity").setLevel(logging.INFO)
    rng = np.random.default_rng(26)
    out = {}

    # align._reject: residuals of point matches (2-D), some far off, pairs of several sizes
    err = rng.normal(0, 1.5, (600, 2))
    err[::13] += 20
    pid = rng.integers(0, 15, 600)
    keep, rejected = align._reject(err, pid, 15, 2.0)
    out["align._reject keep"], out["align._reject rejected"] = keep, rejected
    out["align._reject empty"] = np.array([x.size for x in align._reject(np.zeros((0, 2)), np.zeros(0, int), 3, 2.0)])

    # tilesolve.solve_window: the robust rounds (as tests/test_align.py's warp test, with bad matches added)
    origin = {"0-0": np.array([0.0, 0.0]), "0-1": np.array([900.0, 0.0])}
    T0 = {(z, t): transforms.translation(*o) for z in (0, 1, 2) for t, o in origin.items()}
    shapes = {k: (800, 1000) for k in T0}
    m = {k: [] for k in ("za", "ta", "qa", "zb", "tb", "qb")}
    for z_other in (0, 2):
        for t in origin:
            q1 = rng.uniform([5, 5], [995, 795], (300, 2))
            qo = q1 + rng.normal(0, 0.7, q1.shape) + np.where(rng.random((300, 1)) < 0.08, 40.0, 0.0)
            ok = (qo[:, 0] > 0) & (qo[:, 0] < 1000) & (qo[:, 1] > 0) & (qo[:, 1] < 800)
            za, zb, qa, qb = (z_other, 1, qo, q1) if z_other == 0 else (1, z_other, q1, qo)
            for key, v in (("za", np.full(ok.sum(), za)), ("zb", np.full(ok.sum(), zb)), ("ta", np.full(ok.sum(), t)),
                           ("tb", np.full(ok.sum(), t)), ("qa", qa[ok]), ("qb", qb[ok])):
                m[key].append(v)
    m = {k: np.concatenate(v) for k, v in m.items()}
    A, rep, _, _ = tilesolve.solve_window(T0, shapes, m, {1}, {**align.DEFAULTS["align"]})
    for k in sorted(A):
        out[f"tilesolve.solve_window {k}"] = np.asarray(A[k])
    out["tilesolve.solve_window report"] = np.array([float(v) for k, v in sorted(rep.items())
                                                     if isinstance(v, (int, float, np.floating, np.integer))])

    # curtains.peak and measure_window: InLens with vertical curtains, EsB without (and EsB with them too)
    s = {**curtains.SETTINGS}
    for name, amp_e in (("esb clean", 0.0), ("esb striped", 6.0)):
        a, b = stripes(rng, 256, 6.0), stripes(rng, 256, 6.0)
        ea, eb = stripes(rng, 256, amp_e, noise=6.0), stripes(rng, 256, amp_e, noise=6.0)
        for with_esb in (False, True):
            res = curtains.measure_window(a, b, *((ea, eb) if with_esb else (None, None)), s=s)
            out[f"curtains.measure_window {name} esb={with_esb}"] = np.array(
                [str(res.get(k)) for k in sorted(res)])
        prof = curtains.angle_profile(curtains._spectrum(a), curtains._spectrum(b), s)
        out[f"curtains.peak {name}"] = np.array(curtains.peak(prof, s), float)

    # overlaps.fit_relation: pairs on a line with outliers
    vb = rng.uniform(1000, 5000, 4000)
    va = 1.3 * vb + 200 + rng.normal(0, 30, vb.size)
    va[::9] += rng.normal(0, 800, va[::9].size)
    out["overlaps.fit_relation"] = np.array(overlaps.fit_relation(va, vb), float)

    # intensity.sample_noise, follow_slice_level (logged robust SD), _log_joint (logged robust SDs)
    series = [(np.arange(40) * 5.0, 100 + np.cumsum(rng.normal(0, 1, 40))), (np.arange(30) * 3.0, rng.normal(0, 2, 30))]
    out["intensity.sample_noise"] = np.array([intensity.sample_noise(series)])
    zs = np.repeat(np.arange(60), 3)
    df = pd.DataFrame({"z": zs, "tile": np.tile(["0-0", "0-1", "1-0"], 60), "block": zs // 25,
                       "lo": 1000 + rng.normal(0, 5, zs.size), "hi": 9000 + rng.normal(0, 5, zs.size),
                       "t_p50": 5000 + rng.normal(0, 80, zs.size)})
    rec.args.clear()
    res = intensity.follow_slice_level(df, "t_p50", 9)
    out["intensity.follow_slice_level lo"], out["intensity.follow_slice_level hi"] = res["lo"].to_numpy(), res["hi"].to_numpy()
    out["intensity.follow_slice_level log"] = np.array([float(x) for x in rec.args[-1]])
    n_rel, n_t = 80, 30
    sub = pd.DataFrame({"z": np.repeat(np.arange(10), 3)})
    rb = pd.DataFrame({"gain": 1 + rng.normal(0, 0.02, n_rel), "offset": rng.normal(0, 20, n_rel),
                       "centre": rng.uniform(2000, 6000, n_rel)})
    g, o = 1 + rng.normal(0, 0.03, n_t), rng.normal(0, 30, n_t)
    rec.args.clear()
    intensity._log_joint(0, sub, rb, rng.integers(0, n_t, n_rel), rng.integers(0, n_t, n_rel), g, o, 4000.0,
                         rng.random(n_rel) < 0.4, np.arange(n_t))
    out["intensity._log_joint log"] = np.array([float(x) for x in rec.args[-1][1:]])
    rec.args.clear()
    intensity._log_joint(0, sub, rb, rng.integers(0, n_t, n_rel), rng.integers(0, n_t, n_rel), g, o, 4000.0,
                         np.zeros(n_rel, bool), np.arange(n_t))       # no z relations: NaN for them
    out["intensity._log_joint log, no z relations"] = np.array([float(x) for x in rec.args[-1][1:]])

    # tools
    zflicker, slice_quality = load_tool(repo, "zflicker"), load_tool(repo, "slice_quality")
    refmask_check, raw_stats = load_tool(repo, "refmask_check"), load_tool(repo, "raw_stats")
    clipping_check, seam_steps = load_tool(repo, "clipping_check"), load_tool(repo, "seam_steps")
    v = rng.normal(0, 2, 500)
    v[::50] = np.nan
    out["zflicker.robust_sd"] = np.array([zflicker.robust_sd(v), zflicker.robust_sd([]), zflicker.robust_sd([np.nan])])
    r = rng.standard_t(3, 300)
    r[::40] = np.nan
    out["slice_quality.robust_z"] = slice_quality.robust_z(r)
    out["slice_quality.robust_z constant"] = slice_quality.robust_z(np.ones(20))
    level = np.cumsum(rng.normal(0, 1, 200))
    level[[30, 90, 91]] += [15, -12, -12]
    jump, usual = slice_quality.lone_jump(level)
    out["slice_quality.lone_jump"], out["slice_quality.lone_jump usual"] = jump, np.array([usual, slice_quality.lone_jump(level[:4])[1]])
    imgs = [rng.normal(500 + 100 * k, 50, (64, 80)).astype(np.float32) for k in range(3)]
    for k, im in enumerate(slice_quality.to8(imgs)):
        out[f"slice_quality.to8 own {k}"] = im
    for k, im in enumerate(slice_quality.to8([i.astype(np.uint16) for i in imgs], together=True)):
        out[f"slice_quality.to8 together {k}"] = im
    crop = (120 + 30 * np.sin(np.arange(256)[None, :] / 5.0) + rng.normal(0, 8, (256, 256))).clip(0, 255).astype(np.uint8)
    out["refmask_check.metrics"] = np.array([v for k, v in sorted(refmask_check.metrics(crop).items())])
    # raw_stats.classify: textured tissue on the left, smooth bright resin reaching the border on the right (as
    # tests/test_raw_stats.py's tile)
    tile = textured(rng, 2048, 2048, 33000, 2500, 150)
    tile[:, 1228:] = 39000 + rng.normal(0, 150, (2048, 820))
    small, _ = raw_stats.downsample(np.clip(tile, 0, 65535).astype(np.uint16), 16)
    lab, strict = raw_stats.classify(small)
    out["raw_stats.classify labels"], out["raw_stats.classify strict"] = lab, strict
    # an 8-bit plane: tissue, bright resin, and a corner with no data
    plane = textured(rng, 3072, 3072, 110, 25, 6)
    plane[:, 2000:] = 215 + rng.normal(0, 4, (3072, 1072))
    plane = np.clip(plane, 1, 255).astype(np.uint8)
    plane[:400, :600] = 0
    out["clipping_check.tissue_mask"] = clipping_check.tissue_mask(plane)
    tissue, resin = seam_steps.classes(plane)
    out["seam_steps.classes tissue"], out["seam_steps.classes resin"] = tissue, resin
    return out


def compare(a, b):
    A, B = np.load(a), np.load(b)
    names = sorted(set(A.files) | set(B.files))
    same = 0
    for n in names:
        if n not in A.files or n not in B.files:
            print(f"only in one file: {n}")
            continue
        x, y = A[n], B[n]
        ok = x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
        same += ok
        print(f"{'same' if ok else 'DIFFERENT'}  {n}  {x.dtype} {x.shape}")
    print(f"{same} of {len(names)} outputs identical")
    return 0 if same == len(names) else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--compare", nargs=2, metavar=("A", "B"))
    p.add_argument("repo", nargs="?")
    p.add_argument("out", nargs="?")
    a = p.parse_args(argv)
    if a.compare:
        return compare(*a.compare)
    if not (a.repo and a.out):
        p.error("REPO and OUT.npz are required (or --compare A.npz B.npz)")
    out = outputs(a.repo)
    np.savez(a.out, **{k: np.asarray(v) for k, v in out.items()})
    print(f"{len(out)} outputs written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
