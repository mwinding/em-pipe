"""tools/ortho_views.py: the x-y, x-z and y-z views are the volume's own voxels through the point, kept inside
the volume at its edges, and written one PNG per view with the volumes stacked."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import ortho_views as ov  # noqa: E402
from pipeline import omezarr  # noqa: E402
from volume_compare import Volume  # noqa: E402


def _volume(path, data):
    omezarr.create(path, data.shape, (8.0, 8.0, 8.0), num_scales=1, chunk=(4, 32, 32), shard=(8, 64, 64))
    omezarr.open_scale(path, 0)[...].write(data).result()


def test_views_are_the_voxels_through_the_point(tmp_path):
    data = np.random.default_rng(0).integers(1, 256, (12, 64, 80), dtype=np.uint8)
    _volume(tmp_path / "a.ome.zarr", data)
    v, (y0, x0, za) = ov.views(Volume(tmp_path / "a.ome.zarr"), 5, 30, 70, 32)
    assert (y0, x0, za) == (14, 48, 0)                       # x window pushed inside the volume
    np.testing.assert_array_equal(v["xy"], data[5, 14:46, 48:80])
    np.testing.assert_array_equal(v["xz"], data[:, 30, 48:80])
    np.testing.assert_array_equal(v["yz"], data[:, 14:46, 70])
    v, _ = ov.views(Volume(tmp_path / "a.ome.zarr"), 5, 30, 70, 32, z_range=[2, 9])
    np.testing.assert_array_equal(v["xz"], data[2:9, 30, 48:80])


def test_one_png_per_view_with_volumes_stacked(tmp_path):
    data = np.full((6, 40, 40), 100, np.uint8)
    _volume(tmp_path / "a.ome.zarr", data)
    _volume(tmp_path / "b.ome.zarr", data + 50)
    spec = {"volumes": [{"label": "A", "source": str(tmp_path / "a.ome.zarr")},
                        {"label": "B", "source": str(tmp_path / "b.ome.zarr")}],
            "points": [[3, 20, 20]], "size": 16}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    ov.main([str(tmp_path / "spec.json"), str(tmp_path / "out")])
    xz = cv2.imread(str(tmp_path / "out" / "ortho_z3_y20_x20_xz.png"), cv2.IMREAD_GRAYSCALE)
    assert xz.shape == (2 * (ov.BAR + 6) + ov.GAP, 16)
    assert (xz[ov.BAR:ov.BAR + 6] == 100).all() and (xz[-6:] == 150).all()
    assert (tmp_path / "out" / "ortho_z3_y20_x20_xy.png").exists()
    assert (tmp_path / "out" / "ortho_z3_y20_x20_yz.png").exists()
