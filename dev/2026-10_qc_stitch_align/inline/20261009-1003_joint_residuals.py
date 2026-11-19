# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-09T10:03:25.787Z
# Ran: remote, via ssh
# Heredoc line: T=${DEV_USER_AREA}/em-pipe_test-M10-D01-D02/stitch-model-test; cat > $T/joint_residuals.py <<'EOF'
# Assistant note before it: The comparison is resubmitted as job 60338851; the first attempt failed because of an import-path mistake in my job command. It takes a few minutes, and I'll report the per-model loop closure as soon as it finishes.

"""Joint-solve residuals per tile pair on real slices: translation vs affine_rigid (and affine)."""
import argparse, sys
import numpy as np
from pipeline import stitch
from pipeline.cli import setup
from pipeline.slices import load_slices, StackCache

cfg = setup(argparse.Namespace(config=sys.argv[1], raw_dir=None, output_dir=None, verbose=False), stitch.DEFAULTS)
df = load_slices(cfg)
cache = StackCache(cfg["raw_dir"])
for z in (7926, 8593, 9136):
    rows = df[df["z"] == z]
    by_tile = {r["tile"]: r for _, r in rows.iterrows()}
    tiles = sorted(by_tile)
    shapes = {t: (int(r["height"]), int(r["width"])) for t, r in by_tile.items()}
    for model in ("translation", "affine_rigid", "affine"):
        opts = {**cfg["stitch"], "model": model}
        pairs, points = stitch._match_pairs(cache, by_tile, tiles, shapes, opts["coarse_factor"], opts)
        T, rms = stitch.solve(points, tiles, "0-0", model, opts["lambdas"])
        per = {f"{a}/{b}": round(v, 2) for (a, b), v in rms.items()}
        res = np.concatenate([np.linalg.norm(
            (pa @ T[a][:, :2].T + T[a][:, 2]) - (pb @ T[b][:, :2].T + T[b][:, 2]), axis=1) for (a, b), (pa, pb) in points.items()])
        lin = {t: np.round(T[t][:, :2].ravel() - [1, 0, 0, 1], 5).tolist() for t in tiles if t != "0-0"}
        print(f"z {z} {model:12s} joint rms per pair {per}  all-match median {np.median(res):.2f} p95 {np.percentile(res, 95):.2f}", flush=True)
        if model != "translation":
            print(f"      linear part - I: {lin}", flush=True)
