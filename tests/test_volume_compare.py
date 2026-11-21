"""tools/volume_compare.py: a zarr and a precomputed copy read as the same (z, y, x) place; crops are picked in
textured tissue; the figures and statistics are written."""

import json
import sys
from pathlib import Path

import numpy as np
import tensorstore as ts

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import volume_compare as vc  # noqa: E402
from pipeline import omezarr  # noqa: E402


def _pair(tmp_path, shape=(8, 96, 80)):
    rng = np.random.default_rng(0)
    data = rng.integers(1, 256, shape, dtype=np.uint8)
    data[:, :, :10] = 0                                   # no data along one side
    zroot = tmp_path / "a.ome.zarr"
    omezarr.create(zroot, shape, (8.0, 8.0, 8.0), num_scales=1, chunk=(4, 32, 32), shard=(8, 64, 64))
    omezarr.open_scale(zroot, 0)[...].write(data).result()
    proot = tmp_path / "b.precomputed"
    store = ts.open({"driver": "neuroglancer_precomputed", "kvstore": {"driver": "file", "path": str(proot) + "/"},
                     "multiscale_metadata": {"type": "image", "data_type": "uint8", "num_channels": 1},
                     "scale_metadata": {"size": [shape[2], shape[1], shape[0]], "resolution": [8, 8, 8],
                                        "encoding": "raw", "chunk_size": [32, 32, 4]}}, create=True).result()
    store[...].write(data.transpose(2, 1, 0)[..., None]).result()
    return data, zroot, proot


def test_zarr_and_precomputed_are_the_same_place(tmp_path):
    data, zroot, proot = _pair(tmp_path)
    a, b = vc.Volume(zroot), vc.Volume(proot)
    assert a.shape() == b.shape() == data.shape
    np.testing.assert_array_equal(a.plane(0, 5, 10, 40, 20, 70), data[5, 10:40, 20:70])
    np.testing.assert_array_equal(b.plane(0, 5, 10, 40, 20, 70), data[5, 10:40, 20:70])
    np.testing.assert_array_equal(b.plane(0, 3), data[3])


def test_stats_leave_no_data_out():
    img = np.array([[0, 0, 10, 20], [30, 255, 255, 40]], np.uint8)
    st = vc.stats(img)
    assert st["n"] == 6 and st["median"] == 35 and abs(st["at_255_pct"] - 100 / 3) < 1e-9
    assert vc.stats(np.zeros((3, 3), np.uint8)) == {"n": 0}


def test_crops_are_picked_in_textured_data_and_spread_out():
    ov = np.zeros((100, 100), np.uint8)
    ov[:, 50:] = 120                                       # flat data (resin-like)
    ov[:, 50:] += (np.random.default_rng(1).integers(0, 60, (100, 50))).astype(np.uint8) * (np.arange(100)[:, None] < 50)
    picks = vc.pick_crops(ov, 4, 32, 3, np.random.default_rng(0))
    assert 1 <= len(picks) <= 3
    for y, x in picks:
        assert x >= 50 * 4 and y < 50 * 4 + 16              # in the data, in its textured half


def test_main_writes_a_figure_per_z_and_stats(tmp_path):
    data, zroot, proot = _pair(tmp_path)
    spec = {"volumes": [{"label": "zarr", "source": str(zroot)}, {"label": "precomputed", "source": str(proot)}],
            "z": [2, 6], "overview_scale": 0, "crop_px": 32, "crops": {"6": [[48, 50]]}}
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps(spec))
    assert vc.main([str(sp), str(tmp_path / "cmp.png"), "--json", str(tmp_path / "cmp.json")]) == 0
    assert (tmp_path / "cmp_z2.png").exists() and (tmp_path / "cmp_z6.png").exists()
    out = json.loads((tmp_path / "cmp.json").read_text())
    crop = out["z"]["6"]["crops"][0]
    assert crop["yx"] == [48, 50] and crop["zarr"] == crop["precomputed"]
    assert out["z"]["2"]["overview"]["zarr"]["median"] == float(np.median(data[2][data[2] > 0]))
