# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-09T10:08:58.187Z
# Ran: remote, via ssh
# Heredoc line: T=${DEV_USER_AREA}/em-pipe_test-M10-D01-D02/stitch-model-test; cat > $T/dump_points.py <<'EOF'
# Assistant note before it: Stitch and end-to-end tests pass with the new solve path. I'm now waiting on the real-data joint residuals (job 60342343). They'll show whether Janelia's model fits your tiles jointly to near the matching noise (about 2 px), and I'll set the acceptance threshold from them.

"""Save the real point matches (affine-RANSAC filtered, as for affine_rigid) of three slices."""
import argparse, sys
import numpy as np
from pipeline import stitch
from pipeline.cli import setup
from pipeline.slices import load_slices, StackCache

cfg = setup(argparse.Namespace(config=sys.argv[1], raw_dir=None, output_dir=None, verbose=False), stitch.DEFAULTS)
df = load_slices(cfg)
cache = StackCache(cfg["raw_dir"])
out = {}
for z in (7926, 8593, 9136):
    rows = df[df["z"] == z]
    by_tile = {r["tile"]: r for _, r in rows.iterrows()}
    tiles = sorted(by_tile)
    shapes = {t: (int(r["height"]), int(r["width"])) for t, r in by_tile.items()}
    for pm in ("translation", "affine"):
        opts = {**cfg["stitch"], "model": pm}
        _, points = stitch._match_pairs(cache, by_tile, tiles, shapes, opts["coarse_factor"], opts)
        for (a, b), (pa, pb) in points.items():
            out[f"z{z}_{pm}_{a}_{b}_pa"] = pa
            out[f"z{z}_{pm}_{a}_{b}_pb"] = pb
    print("done", z, flush=True)
np.savez(sys.argv[2], **out)
