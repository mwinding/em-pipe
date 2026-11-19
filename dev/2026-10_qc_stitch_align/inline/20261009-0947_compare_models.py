# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-09T09:47:42.539Z
# Ran: remote, via ssh
# Heredoc line: T=${DEV_USER_AREA}/em-pipe_test-M10-D01-D02; mkdir -p $T/stitch-model-test && cat > $T/stitch-model-test/compare_models.py <<'EOF'
# Assistant note before it: The full suite is running in the background. Meanwhile, `configs/example.yaml` needs the new stitch defaults; a test checks that it matches the code:

"""Stitch three real slices with each tile model; report loop closure and overlap residuals."""
import sys
import numpy as np
from pipeline import stitch
from pipeline.cli import setup
from pipeline.slices import load_slices, StackCache
import argparse

args = argparse.Namespace(config=sys.argv[1], raw_dir=None, output_dir=None, verbose=False)
cfg = setup(args, stitch.DEFAULTS)
df = load_slices(cfg)
cache = StackCache(cfg["raw_dir"])
for z in (7926, 8593, 9136):
    rows = df[df["z"] == z]
    for model in ("translation", "rigid", "similarity", "affine", "affine_rigid"):
        opts = {**cfg["stitch"], "model": model}
        rec = stitch.stitch_slice(rows, cache, opts, "0-0")
        used = [p for p in rec["pairs"] if p["A"] is not None]
        pts = {(p["tile_a"], p["tile_b"]): p for p in used}
        sol = [p["solve_residual_px"] for p in used if p["solve_residual_px"] is not None]
        print(f"z {z} {model:13s} ok={rec['ok']!s:5} pairs={len(used)} pair_fit_rms={np.mean([p['residual_px'] for p in used]):5.2f} "
              f"solve_rms_max={max(sol) if sol else float('nan'):6.2f}  {rec['reason'][:70]}", flush=True)
