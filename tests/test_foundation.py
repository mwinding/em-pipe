"""Tests for the shared modules: config, cli, imagej_tiff, transforms, features, synth."""

import numpy as np
import pytest
import tifffile

from pipeline import cli, features, transforms
from pipeline.config import load_config
from pipeline.imagej_tiff import ImageJStack


def test_synth_files_and_reader(synth_2x2):
    names = sorted(synth_2x2.files)
    # 24 slices from 23:35 at ~142 s cross midnight -> two days x 4 tiles
    assert len(names) == 8
    assert {n[:7] for n in names} == {"M09_D24", "M09_D25"}
    name = names[0]
    st = ImageJStack(synth_2x2.raw_dir / name)
    zs = synth_2x2.files[name]
    assert st.shape == (len(zs), *synth_2x2.tile_shape)
    assert st.contiguous and not st.truncated
    assert st.byteorder == ">"
    assert len(st.labels) == len(zs) and st.labels[0].startswith("G460-0186_26-09-")
    vx, vy, vz = st.voxel_size_nm
    assert vx == pytest.approx(8.0) and vy == pytest.approx(8.0) and vz == pytest.approx(8.0)
    ref = tifffile.imread(synth_2x2.raw_dir / name)
    np.testing.assert_array_equal(st.read(1), ref[1])
    np.testing.assert_array_equal(st.read(2, slice(10, 20), slice(5, 9)), ref[2, 10:20, 5:9])
    assert st.read(0).dtype == np.uint16 and st.read(0).dtype.isnative


def test_truncated_detection(tmp_path):
    import synth
    truth = synth.make_dataset(tmp_path / "raw", n_slices=4, faults={"truncate": [("M09_D24", "1-1", 1000)]})
    name = [n for n in truth.files if "tile1-1" in n][0]
    st = ImageJStack(truth.raw_dir / name)
    assert st.truncated
    assert st.memmap().shape[0] == st.n - 1   # only complete slices are mapped


def test_single_slice_labels(tmp_path):
    import synth
    synth.write_imagej(tmp_path / "one.tif", np.zeros((1, 8, 8), np.uint16), ["only-label.tif"])
    assert ImageJStack(tmp_path / "one.tif").labels == ["only-label.tif"]


def test_config_overrides(make_config, synth_2x2, tmp_path):
    path = make_config(synth_2x2, selection={"z_start": 3})
    cfg = load_config(path, output_dir=str(tmp_path / "other"), step_defaults={"stitch": {"model": "translation"}})
    assert cfg["output_dir"] == str(tmp_path / "other")
    assert cfg["stitch"]["model"] == "translation"
    assert cfg["selection"] == {"z_start": 3}
    assert "file_pattern" in cfg["raw"]


def test_chunks_and_tasks():
    ch = cli.chunks(range(10), 4, overlap=2)
    assert [c for c, _ in ch] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert [e for _, e in ch] == [[0, 1, 2, 3, 4, 5], [4, 5, 6, 7, 8, 9], [8, 9]]
    assert cli.my_chunks(list("abcdefg"), 1, 3) == ["b", "e"]


def test_transforms_roundtrip():
    A = np.array([[0.99, -0.02, 5.0], [0.03, 1.01, -2.0]])
    B = transforms.translation(3, 4)
    pts = np.array([[0, 0], [10, 20], [100, 5]], float)
    np.testing.assert_allclose(transforms.apply(transforms.compose(A, B), pts),
                               transforms.apply(A, transforms.apply(B, pts)))
    np.testing.assert_allclose(transforms.apply(transforms.invert(A), transforms.apply(A, pts)), pts, atol=1e-9)
    df = transforms.to_frame([({"z": 0, "tile": "0-1"}, A)])
    assert list(df.columns) == ["z", "tile", *transforms.COLUMNS]


@pytest.mark.parametrize("model", features.MODELS)
def test_fit_model_recovers_known_transform(model):
    rng = np.random.default_rng(1)
    pa = rng.uniform(0, 500, (200, 2))
    theta = 0.01 if model != "translation" else 0.0
    s = 1.02 if model in ("similarity", "affine") else 1.0
    A = np.array([[s * np.cos(theta), -s * np.sin(theta), 12.3], [s * np.sin(theta), s * np.cos(theta), -7.8]])
    if model == "affine":
        A[0, 1] += 0.01
    pb = transforms.apply(A, pa) + rng.normal(0, 0.2, pa.shape)
    pb[:40] = rng.uniform(0, 500, (40, 2))  # 20 % outliers
    est, inl = features.fit_model(pa, pb, model=model, threshold=1.5)
    assert est is not None
    assert inl[40:].mean() > 0.95 and inl[:40].mean() < 0.1
    np.testing.assert_allclose(transforms.apply(est, pa[40:]), transforms.apply(A, pa[40:]), atol=0.3)


def test_sift_finds_true_tile_offset(synth_2x2):
    """SIFT + RANSAC on the overlap of two neighbouring synthetic tiles recovers the true offset."""
    t = synth_2x2
    name_a = [n for n in t.files if n.startswith("M09_D24") and "tile0-0" in n][0]
    name_b = name_a.replace("tile0-0", "tile0-1")
    a = ImageJStack(t.raw_dir / name_a).read(0)
    b = ImageJStack(t.raw_dir / name_b).read(0)
    ka, da = features.detect(features.to_uint8(a))
    kb, db = features.detect(features.to_uint8(b))
    ia, ib = features.match(da, db)
    # model maps tile_b pixels -> tile_a pixels
    A, inl = features.fit_model(kb[ib], ka[ia], "translation", threshold=2.0)
    assert A is not None and inl.sum() >= 10
    true = np.subtract(t.tile_origin["0-1"], t.tile_origin["0-0"])
    np.testing.assert_allclose(A[:, 2], true, atol=0.5)


@pytest.mark.parametrize("compression", ["gzip", "blosc", "none"])
def test_omezarr_create_write_read(tmp_path, compression):
    import json
    from pipeline import omezarr
    root = tmp_path / "v.ome.zarr"
    shapes = omezarr.create(root, (70, 300, 260), (8.0, 8.0, 8.0), num_scales=3, chunk=(16, 32, 32),
                            shard=(32, 128, 128), compression=compression)
    assert shapes == [(70, 300, 260), (35, 150, 130), (18, 75, 65)]
    meta = json.loads((root / "zarr.json").read_text())
    ds = meta["attributes"]["ome"]["multiscales"][0]["datasets"]
    assert [d["path"] for d in ds] == ["s0", "s1", "s2"]
    assert ds[1]["coordinateTransformations"][0]["scale"] == [16.0, 16.0, 16.0]
    arr = omezarr.open_scale(root, 0)
    assert omezarr.shard_shape(arr) == (32, 128, 128)
    data = np.random.default_rng(0).integers(0, 255, (32, 300, 260), dtype=np.uint8)
    arr[32:64, :, :].write(data).result()
    np.testing.assert_array_equal(omezarr.open_scale(root, 0)[32:64].read().result(), data)
    assert omezarr.open_scale(root, 0)[0:1, 0:1, 0:1].read().result()[0, 0, 0] == 0
    boxes = omezarr.shard_boxes(shapes[0], (32, 128, 128))
    assert len(boxes) == 3 * 3 * 3 and boxes[-1] == ((64, 70), (256, 300), (256, 260))


def test_selection_is_local_time_when_timezone_set():
    """check stores UTC when check.timezone is set; selection start/end are still label-clock time."""
    import pandas as pd
    from pipeline.slices import select
    # 2026-10-01 00:30 BST = 2026-09-30 23:30 UTC; 2026-10-25 01:30 is the repeated hour (BST pass first).
    utc = pd.to_datetime(["2026-09-30 22:59:59", "2026-09-30 23:30:00", "2026-10-01 22:59:59",
                          "2026-10-01 23:00:00", "2026-10-25 00:30:00", "2026-10-25 01:30:00"])
    df = pd.DataFrame({"z": range(6), "timestamp": utc})
    sel = {"start": "2026-10-01 00:00:00", "end": "2026-10-01 23:59:59"}
    assert select(df, sel, "Europe/London")["z"].tolist() == [1, 2]
    assert select(df, sel)["z"].tolist() == [2, 3]          # no timezone: times compared as written
    assert select(df, {"start": "2026-10-25 01:30:00"}, "Europe/London")["z"].tolist() == [4, 5]


def test_read_blocks_match_tifffile(tmp_path):
    """Every read path (whole slice, wide block, narrow strip, steps, ints) equals tifffile's array."""
    import synth
    data = np.random.default_rng(3).integers(0, 65535, (3, 50, 64), dtype=np.uint16)
    path = tmp_path / "s.tif"
    synth.write_imagej(path, data, [f"l{i}" for i in range(3)])
    st = ImageJStack(path)
    for rows, cols in [(slice(None), slice(None)), (slice(5, 40), slice(2, 60)), (slice(0, 50), slice(10, 14)),
                       (slice(3, 30, 2), slice(1, 63, 3)), (slice(10, 10), slice(0, 5)), (slice(45, 99), slice(60, 70)),
                       (slice(None, None, -1), slice(5, 9)), (7, slice(0, 64))]:
        np.testing.assert_array_equal(st.read(2, rows, cols), data[2][rows, cols])
    assert st.read(1, slice(0, 50), slice(10, 14)).dtype.isnative


def test_read_truncated_slice_raises(tmp_path):
    import synth
    truth = synth.make_dataset(tmp_path / "raw", n_slices=4, faults={"truncate": [("M09_D24", "1-1", 1000)]})
    name = [n for n in truth.files if "tile1-1" in n][0]
    st = ImageJStack(truth.raw_dir / name)
    st.read(st.n - 2)
    with pytest.raises(EOFError):
        st.read(st.n - 1)


def test_atomic_write_uses_normal_permissions(tmp_path):
    """Outputs get a normal write's permissions (e.g. group-readable), not mkstemp's private 0600."""
    import os
    old = os.umask(0o027)   # set after import: the process's current umask must be honoured
    try:
        path = tmp_path / "x.csv"
        cli.atomic_write(path, lambda tmp: open(tmp, "w").write("a"))
        open(tmp_path / "plain.csv", "w").close()
    finally:
        os.umask(old)
    assert (path.stat().st_mode & 0o777) == (tmp_path / "plain.csv").stat().st_mode & 0o777 == 0o640
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []


def test_atomic_write_follows_default_acl(tmp_path):
    """In a folder with a default ACL (NEMO lab folders) outputs get the ACL's rw-rw----, not 0666 & ~umask."""
    import os
    import shutil
    import subprocess
    if not shutil.which("setfacl") or subprocess.run(
            ["setfacl", "-d", "-m", "u::rwx,g::rwx,o::---", str(tmp_path)], capture_output=True).returncode:
        pytest.skip("POSIX default ACLs not available here")
    old = os.umask(0o022)
    try:
        path = tmp_path / "x.csv"
        cli.atomic_write(path, lambda tmp: open(tmp, "w").write("a"))
    finally:
        os.umask(old)
    assert (path.stat().st_mode & 0o777) == 0o660
