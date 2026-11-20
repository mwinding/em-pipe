# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-25T23:18:57.734Z
# Ran: remote, via ssh
# Heredoc line: cat > ${DEV_WORK}/cluster/planes_37i.py <<'EOF'
# Assistant note before it: That's worth checking: the first 1,914 planes are the single-tile approach, which is roughly the first 60% of the stack. There are two possibilities: - **Expected:** the single tile images a smaller field than the 2×2 grid, so in parts of the 2×2 area the early planes have no data. - **A real proble

"""37i: per plane (s3, 64 nm; z every 8 slices), the share of pixels with data and their brightness, in both volumes."""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import sys
sys.path.insert(0, os.path.expandvars("${EM_PIPE_ROOT}"))
import numpy as np
from pipeline import omezarr
O = os.path.expandvars("${DEV_37I}/")
for v in (os.path.expandvars("${DEV_PREFIX}37i_8nm"), os.path.expandvars("${DEV_PREFIX}37i_8nm_raw")):
    a = omezarr.open_scale(O + v + ".ome.zarr", 3)
    print(v, a.shape, flush=True)
    for k in list(range(0, 240, 20)) + [238, 239, 240, 241, 260, 300, 340, 380]:
        if k >= a.shape[0]:
            continue
        p = a[k].read().result()
        nz = p > 0
        print(f"  z~{8 * k:5d}: data {100 * nz.mean():5.1f} % of plane, mean {p[nz].mean() if nz.any() else 0:5.1f}, "
              f"p1/p99 {np.percentile(p[nz], [1, 99]).round().tolist() if nz.any() else '-'}", flush=True)
