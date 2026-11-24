"""tools/zflicker.py: a shared brightness step on single planes is found as common flicker; a step in one box
only as that box's own flicker; empty boxes are left out."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import zflicker  # noqa: E402
from pipeline import omezarr  # noqa: E402


def _volume(path, data):
    omezarr.create(path, data.shape, (8.0, 8.0, 8.0), num_scales=1, chunk=(4, 32, 32), shard=(8, 64, 64))
    omezarr.open_scale(path, 0)[...].write(data).result()


def test_common_and_own_flicker(tmp_path):
    rng = np.random.default_rng(0)
    nz, ny, nx = 40, 64, 64
    base = np.full((nz, ny, nx), 120.0) + np.linspace(0, 10, nz)[:, None, None]     # slow change: not flicker
    common = rng.normal(0, 4.0, nz)                                                  # whole-plane jumps
    a = base + common[:, None, None]
    b = base.copy()
    b[::2, :32, :32] += 6.0                                                          # one box, every other plane
    a[:, :, :20] = 0                                                                 # no data: boxes left out
    _volume(tmp_path / "a.ome.zarr", a.astype(np.uint8))
    _volume(tmp_path / "b.ome.zarr", b.astype(np.uint8))
    zflicker.main(["--out", str(tmp_path / "out"), "--scale", "0", "--grid", "2", "--window", "5",
                   f"A={tmp_path / 'a.ome.zarr'}", f"B={tmp_path / 'b.ome.zarr'}"])
    s = json.loads((tmp_path / "out" / "zflicker.json").read_text())["series"]
    assert s["A"]["boxes"] == 2                       # the left column of boxes is under half covered
    assert 2.0 < s["A"]["flicker_common_sd"] < 6.0
    assert s["A"]["flicker_own_sd"] < 0.6
    assert s["B"]["flicker_common_sd"] < 0.6          # one box of four moving leaves the median alone
    rows = pd.read_csv(tmp_path / "out" / "zflicker.csv").query("series == 'B'")
    own = rows.groupby("box")["resid"].std()
    assert own["0-0"] > 2.5 and own.drop("0-0").max() < 0.6
    assert (tmp_path / "out" / "zflicker.png").exists()


def test_box_px_reads_a_centred_square(tmp_path):
    nz, ny, nx = 12, 64, 64
    data = np.full((nz, ny, nx), 100, np.uint8)
    data[:, 8:24, 8:24] = 200                         # the centre of grid cell 0-0 (grid 2: cells of 32 px)
    data[5] += 10                                     # one plane brighter everywhere
    _volume(tmp_path / "v.ome.zarr", data)
    df = zflicker.box_means(tmp_path / "v.ome.zarr", 0, 2, box_px=16)
    assert len(df) == 4 * nz
    v = df.set_index(["z", "box"])["value"]
    assert v[(0, "0-0")] == 200 and v[(0, "1-1")] == 100 and v[(5, "0-1")] == 110
