"""export: quick-look TIFF of the rendered volume, from the finest scale under export.max_gb."""

import numpy as np
import tifffile

from pipeline import export, omezarr


def _volume(out, name="synthetic.ome.zarr"):
    root = out / name
    shapes = omezarr.create(root, (32, 96, 128), (8.0, 8.0, 8.0), num_scales=3, chunk=(8, 32, 32),
                            shard=(16, 64, 64))
    rng = np.random.default_rng(0)
    data = {}
    for s, shape in enumerate(shapes):
        data[s] = rng.integers(0, 255, shape, dtype=np.uint8)
        omezarr.open_scale(root, s)[...].write(data[s]).result()
    return root, data


def test_exports_finest_scale_that_fits(make_config, tmp_path):
    out = tmp_path / "out"
    root, data = _volume(out)
    # s0 is 393 kB, s1 49 kB: a 0.1 MB cap picks s1 (16 nm)
    cfg = make_config(tmp_path / "raw", export={"max_gb": 0.0001})
    assert export.main(["--config", str(cfg)]) == 0
    tif = out / "synthetic_16nm.tif"
    with tifffile.TiffFile(tif) as t:
        np.testing.assert_array_equal(t.asarray(), data[1])
        assert t.imagej_metadata["spacing"] == 0.016 and t.imagej_metadata["unit"] == "micron"
    # a larger cap gives s0 and replaces the coarser export
    cfg = make_config(tmp_path / "raw", export={"max_gb": 1.0})
    assert export.main(["--config", str(cfg)]) == 0
    assert sorted(p.name for p in out.glob("*.tif")) == ["synthetic_8nm.tif"]
    np.testing.assert_array_equal(tifffile.imread(out / "synthetic_8nm.tif"), data[0])


def test_missing_volume_fails(make_config, tmp_path):
    cfg = make_config(tmp_path / "raw")
    assert export.main(["--config", str(cfg)]) == 1


def test_name_uses_rounded_voxel_size(make_config, tmp_path):
    """Headers give 7.99986 nm: the file is called ..._8nm.tif, not ..._7.99986nm.tif."""
    out = tmp_path / "out"
    root = out / "synthetic.ome.zarr"
    omezarr.create(root, (8, 64, 64), (7.99986, 7.99986, 7.99986), num_scales=2, chunk=(8, 32, 32),
                   shard=(8, 64, 64))
    cfg = make_config(tmp_path / "raw")
    assert export.main(["--config", str(cfg)]) == 0
    assert [p.name for p in out.glob("*.tif")] == ["synthetic_8nm.tif"]
