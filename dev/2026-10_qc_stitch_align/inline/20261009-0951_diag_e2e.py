# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-09T09:51:11.440Z
# Ran: local
# Heredoc line: cat > ${DEV_WORK}/diag_e2e.py <<'EOF'
# Assistant note before it: One footprint check still fails at z = 1, so I'll measure how far outside the tile footprint the stray pixels are and how bright they are:

import sys, json
sys.path.insert(0, "tests")
import numpy as np
from scipy import ndimage
import test_e2e as e
import pytest

def test_diag(main_run):
    truth, out = main_run
    meta = json.loads((out / "render" / "volume" / "render.json").read_text())
    s0 = e.read_scales(out)[0]
    present = {}
    import pandas as pd
    sl = pd.read_csv(out / "check" / "slices.csv", dtype={"tile": str})
    for z, g in sl[~sl.excluded].groupby("z"):
        present[z] = set(g.tile)
    for z in range(4):
        montage, feet, corner = e.masks(truth, meta, z)
        covered = np.any([feet[t] for t in present[z]], axis=0)
        img = s0[z].astype(float)
        d = ndimage.distance_transform_edt(~covered)
        bad = (img > 0) & ~covered
        print(z, "outside nonzero:", bad.sum(), "max dist px:", d[bad].max() if bad.any() else 0, "max val:", img[bad].max() if bad.any() else 0)
        al = pd.read_csv(out / "align" / "transforms.csv").set_index("z")
        print("   align t:", al.loc[z, ["tx", "ty"]].values - al.loc[0, ["tx", "ty"]].values, "true:", truth.drift[z] - truth.drift[0])
