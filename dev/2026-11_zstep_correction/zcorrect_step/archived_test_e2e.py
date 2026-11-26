"""The zcorrect part of tests/test_e2e.py, taken out when the step was removed from the pipeline (26 Nov 2026).
A record: pytest does not collect this file, and it cannot pass until the step is a pipeline module again
(``python -m pipeline.zcorrect``) and render reads positions.csv (render_plane_plan.py, next to this file).

In tests/test_e2e.py the second run (``corrected_run``) had decurtaining and zcorrect switched on together:
``run_pipeline(cfg, zcorrect=True)`` ran ``zcorrect run`` (two tasks) and ``zcorrect solve`` between intensity
and render init, as run_pipeline.sh did. Three tests used that run: the one below, and
``test_destreak_removes_stripes_in_render`` and ``test_corrected_pyramid``, which are still in tests/test_e2e.py
and now run on the same data without zcorrect (one plane per slice).

To use it again: in tests/test_e2e.py give ``run_pipeline`` its ``zcorrect`` argument back (below), add the
zcorrect settings to the ``corrected_run`` fixture's config (below) and put the test back.
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import synth
from test_e2e import SMALL, run_step   # tests/ on sys.path


def run_pipeline(cfg, zcorrect=False):
    run_step(cfg, "check")
    run_step(cfg, "preview", "run", tasks=2)
    run_step(cfg, "preview", "merge")
    run_step(cfg, "stitch", "run", tasks=2)
    run_step(cfg, "stitch", "merge")
    run_step(cfg, "align", "run", tasks=2)
    run_step(cfg, "align", "solve")
    run_step(cfg, "intensity")
    if zcorrect:
        run_step(cfg, "zcorrect", "run", tasks=2)
        run_step(cfg, "zcorrect", "solve")
    run_step(cfg, "render", "init")
    run_step(cfg, "render", "run", tasks=2)
    for s in range(1, SMALL["render"]["num_scales"]):
        run_step(cfg, "pyramid", "run", "--scale", str(s), tasks=2)


# ----- second run: destreak and zcorrect inside render -----------------------------------------

@pytest.fixture(scope="module")
def corrected_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e_corrected")
    n = 16
    # Larger tiles: the wavelet-FFT filter needs a few hundred px to tell stripes from structure.
    truth = synth.make_dataset(root / "raw", n_slices=n, tile_shape=(256, 288), overlap=(48, 56),
                               start=datetime(2026, 9, 24, 12), streaks=0.03, tile_gain={"0-1": 1.15},
                               z_positions=np.arange(n) * 0.25, seed=5)
    cfg = synth.write_config(root / "config.yaml", truth.raw_dir, root / "out", **SMALL,
                             destreak={"enabled": True},
                             zcorrect={"enabled": True, "chunk_slices": 8, "max_distance": 4, "crop_px": 128})
    run_pipeline(cfg, zcorrect=True)
    return truth, root / "out"


def test_zcorrect_positions_drive_the_planes(corrected_run):
    truth, out = corrected_run
    n = len(truth.timestamps)
    pos = pd.read_csv(out / "work" / "zcorrect" / "positions.csv")
    assert list(pos["z"]) == list(range(n))
    p = pos["position_nm"].to_numpy()
    assert p[0] == 0 and (np.diff(p) > 0).all()
    assert p[-1] == pytest.approx(8.0 * (n - 1), rel=0.02)   # mean spacing is held at nominal
    meta = json.loads((out / "work" / "render" / "synthetic" / "render.json").read_text())
    assert meta["zcorrected"] and meta["voxel_nm"] == [8.0, 8.0, 8.0]
    assert len(meta["planes"]) == int(np.floor((p[-1] - p[0]) / 8.0 + 1e-6)) + 1
    for k, plane in enumerate(meta["planes"]):
        q = p[0] + 8.0 * k   # plane k sits at q and interpolates the slices that bracket it
        zs = [z for z, _ in plane]
        assert p[min(zs)] <= q + 1e-6 and (len(zs) == 1 or p[max(zs)] >= q)
        assert sum(w for _, w in plane) == pytest.approx(1.0)
