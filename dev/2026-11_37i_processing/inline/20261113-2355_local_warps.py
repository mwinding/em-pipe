# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-11-13T23:55:37.153Z
# Ran: local
# Heredoc line: ${EM_PIPE_ENV}/bin/python - <<'EOF'
# Assistant note before it: The per-tile warp is built and running on z 2083. **em-pipe 686d9a2** (pushed and pulled on the cluster; all 385 tests pass, 3 of them new): - **Tile solve:** with `align.local_warp_px` set, each tile in a tile-solve window gets a smooth displacement grid on top of its affine, solved together with the affi

import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import json, numpy as np
d = json.load(open(os.path.expandvars("${DEV_37I}/work/align/local_warps.json")))
for k, f in d.items():
    g = np.asarray(f["g"])
    ny, nx, _ = g.shape
    m = g.reshape(-1, 2).mean(0)
    # best affine fit of the grid (in node coords) and the non-affine remainder
    yy, xx = np.mgrid[0:ny, 0:nx]
    X = np.column_stack([xx.ravel(), yy.ravel(), np.ones(nx * ny)])
    coef, *_ = np.linalg.lstsq(X, g.reshape(-1, 2), rcond=None)
    rest = g.reshape(-1, 2) - X @ coef
    print(f"{k}: grid {nx}x{ny}, mean ({m[0]:.1f}, {m[1]:.1f}) px, max |g| {np.linalg.norm(g, axis=2).max():.1f}, "
          f"non-affine part max {np.linalg.norm(rest, axis=1).max():.1f} px, rms {np.sqrt((rest**2).sum(1).mean()):.1f} px")
    if k.endswith("0-0"):
        print("  dx by row (mean over columns):", np.round(g[..., 0].mean(1) - g[..., 0].mean(), 1).tolist())
        print("  dx by column (mean over rows):", np.round(g[..., 0].mean(0) - g[..., 0].mean(), 1).tolist())
