"""Tests for pipeline.pyramid: scales against a float/NaN reference, task split, markers, resume."""

import os

import numpy as np
import pytest

from pipeline import omezarr, pyramid

CHUNK, SHARD = (4, 8, 8), (8, 16, 16)
NAME = "test.ome.zarr"


def reference(a):
    """Mean of each 2×2×2 block over the voxels that exist, rounded half up."""
    out = [-(-n // 2) for n in a.shape]
    p = np.full([2 * n for n in out], np.nan)
    p[:a.shape[0], :a.shape[1], :a.shape[2]] = a
    m = np.nanmean(p.reshape(out[0], 2, out[1], 2, out[2], 2), axis=(1, 3, 5))
    return np.floor(m + 0.5).astype(np.uint8)  # counts are powers of two, so halves are exact


def structured(shape, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 256, shape, dtype=np.uint8)
    data[:, :5, :] = 0        # no-data border: averaged in like any other value
    data[-2:, :, -7:] = 255
    return data


@pytest.fixture
def volume(make_config, tmp_path):
    """volume(shape, num_scales) -> (config path, zarr root, s0 data)."""
    def _make(shape, num_scales=4, seed=0):
        cfg = make_config(tmp_path / "raw", render={"name": NAME})
        root = tmp_path / "out" / NAME
        omezarr.create(root, shape, (8.0, 8.0, 8.0), num_scales=num_scales, chunk=CHUNK, shard=SHARD)
        data = structured(shape, seed)
        omezarr.open_scale(root, 0).write(data).result()
        return cfg, root, data
    return _make


def run(cfg, scale, tasks=1, *extra):
    for t in range(tasks):
        args = ["run", "--scale", str(scale), "--config", str(cfg), "--task-id", str(t), "--num-tasks", str(tasks)]
        assert pyramid.main(args + list(extra)) == 0


def read(root, scale):
    return omezarr.open_scale(root, scale).read().result()


def markers(root, scale):
    return sorted(p.name for p in (root.parent / "work" / "render" / "test" / "done").glob(f"s{scale}_*"))


@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 2, 2), (3, 5, 7), (6, 9, 4), (5, 1, 11)])
def test_downsample_matches_reference(shape):
    a = structured(shape, seed=sum(shape))
    np.testing.assert_array_equal(pyramid.downsample(a), reference(a))


def test_downsample_rounding():
    a = np.zeros((2, 2, 2), np.uint8)
    a[0, 0, 0] = 4                      # mean 0.5 -> 1 (halves up)
    assert pyramid.downsample(a)[0, 0, 0] == 1
    a[0, 0, 0] = 3                      # mean 0.375 -> 0
    assert pyramid.downsample(a)[0, 0, 0] == 0
    assert pyramid.downsample(np.full((3, 3, 3), 255, np.uint8)).tolist() == [[[255] * 2] * 2] * 2


@pytest.mark.parametrize("shape", [(37, 70, 53), (3, 33, 17)])
def test_all_scales_match_reference(volume, shape):
    cfg, root, data = volume(shape)
    expected = data
    for s in (1, 2, 3):
        run(cfg, s, tasks=3)
        expected = reference(expected)
        got = read(root, s)
        assert got.shape == expected.shape
        np.testing.assert_array_equal(got, expected)
        n = len(omezarr.shard_boxes(got.shape, SHARD))
        assert markers(root, s) == [f"s{s}_{i:06d}" for i in range(n)]


def test_task_takes_strided_shards(volume):
    cfg, root, data = volume((37, 70, 53))
    run_args = ["run", "--scale", "1", "--config", str(cfg), "--task-id", "1", "--num-tasks", "3"]
    assert pyramid.main(run_args) == 0
    boxes = omezarr.shard_boxes(read(root, 1).shape, SHARD)
    mine = list(range(1, len(boxes), 3))
    assert markers(root, 1) == [f"s1_{i:06d}" for i in mine]
    got, expected = read(root, 1), reference(data)
    for i, box in enumerate(boxes):
        sl = tuple(slice(a, b) for a, b in box)
        np.testing.assert_array_equal(got[sl], expected[sl] if i in mine else 0)


def marker_times(root):
    return {p.name: p.stat().st_mtime for p in (root.parent / "work" / "render" / "test" / "done").iterdir()}


def test_skip_done_and_overwrite(volume):
    cfg, root, data = volume((20, 40, 36))
    run(cfg, 1, tasks=2)
    expected = reference(data)
    omezarr.open_scale(root, 1).write(np.full(expected.shape, 7, np.uint8)).result()   # tamper with s1

    run(cfg, 1, tasks=2)                       # all done: nothing recomputed
    assert (read(root, 1) == 7).all()

    boxes = omezarr.shard_boxes(expected.shape, SHARD)
    (root.parent / "work" / "render" / "test" / "done" / "s1_000002").unlink()
    run(cfg, 1)                                # only the shard without a marker is redone
    got = read(root, 1)
    for i, box in enumerate(boxes):
        sl = tuple(slice(a, b) for a, b in box)
        np.testing.assert_array_equal(got[sl], expected[sl] if i == 2 else 7)

    run(cfg, 1, 2, "--overwrite")
    np.testing.assert_array_equal(read(root, 1), expected)


def test_rewritten_source_shard_redoes_dependent_shards(volume):
    """Regression: re-rendering s0 in place used to leave s1.. stale unless --overwrite."""
    cfg, root, data = volume((37, 70, 53))
    for s in (1, 2):
        run(cfg, s, tasks=2)
    before = marker_times(root)
    # Re-render the s0 shard at grid (z 0, y 2, x 1) = s0[0:8, 32:48, 16:32]. It feeds s1[0:4, 16:24, 8:16],
    # i.e. s1 shard 2 ((0, 8), (16, 32), (0, 16)), which feeds s2 shard 0 only.
    data[0:8, 32:48, 16:32] = 255 - data[0:8, 32:48, 16:32]
    omezarr.open_scale(root, 0)[0:8, 32:48, 16:32].write(data[0:8, 32:48, 16:32]).result()
    for s in (1, 2):
        run(cfg, s, tasks=2)
    np.testing.assert_array_equal(read(root, 1), reference(data))
    np.testing.assert_array_equal(read(root, 2), reference(reference(data)))
    after = marker_times(root)
    assert sorted(k for k in after if after[k] != before[k]) == ["s1_000002", "s2_000000"]


def test_recreated_volume_invalidates_markers(volume):
    cfg, root, _ = volume((20, 40, 36))
    run(cfg, 1)
    stamp = (root.parent / "work" / "render" / "test" / "done" / "s1_000000").stat().st_mtime
    _, _, data = volume((20, 40, 36), seed=5)   # render init again, new s0
    assert (root / "s1" / "zarr.json").stat().st_mtime > stamp
    run(cfg, 1)
    np.testing.assert_array_equal(read(root, 1), reference(data))


def test_requires_previous_scale_complete(volume):
    cfg, root, data = volume((20, 40, 36))
    args = ["run", "--config", str(cfg), "--scale"]
    assert pyramid.main(args + ["2"]) == 1     # s1 not built yet
    assert markers(root, 2) == []
    run(cfg, 1)
    os.remove(root.parent / "work" / "render" / "test" / "done" / "s1_000001")
    assert pyramid.main(args + ["2"]) == 1     # s1 incomplete
    run(cfg, 1)
    assert pyramid.main(args + ["2"]) == 0
    np.testing.assert_array_equal(read(root, 2), reference(reference(data)))
    assert pyramid.main(args + ["0"]) == 1
    assert pyramid.main(args + ["4"]) == 1     # only s0..s3 exist


def test_task_checks_only_the_shards_it_reads(volume):
    # s1 (19, 35, 27) has a 3x3x2 shard grid; s2 (10, 18, 14) has 2x2x1. s2 shard 3 ((8, 10), (16, 18), (0, 14))
    # reads s1 grid (2, 2, 0..1) = shards 16, 17; s2 shard 0 reads s1 grid (0..1, 0..1, 0..1).
    cfg, root, data = volume((37, 70, 53))
    run(cfg, 1)
    os.remove(root.parent / "work" / "render" / "test" / "done" / "s1_000016")
    task = ["run", "--scale", "2", "--config", str(cfg), "--num-tasks", "4", "--task-id"]
    assert pyramid.main(task + ["3"]) == 1
    for t in (0, 1, 2):
        assert pyramid.main(task + [str(t)]) == 0
    assert markers(root, 2) == ["s2_000000", "s2_000001", "s2_000002"]
    run(cfg, 1)
    assert pyramid.main(task + ["3"]) == 0
    np.testing.assert_array_equal(read(root, 2), reference(reference(data)))
