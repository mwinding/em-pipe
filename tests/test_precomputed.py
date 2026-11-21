"""precomputed: Neuroglancer precomputed export of a rendered OME-Zarr volume."""

import json
import math
import os
import shutil

import numpy as np
import pytest
import tensorstore as ts

from pipeline import omezarr, precomputed


def _volume(out, shape=(40, 150, 130), voxel=(8.0, 8.0, 8.0), num_scales=3):
    root = out / "synthetic.ome.zarr"
    shapes = omezarr.create(root, shape, voxel, num_scales=num_scales, chunk=(8, 32, 32), shard=(16, 64, 64))
    rng = np.random.default_rng(0)
    data = {}
    for s, sh in enumerate(shapes):
        data[s] = rng.integers(0, 255, sh, dtype=np.uint8)
        omezarr.open_scale(root, s)[...].write(data[s]).result()
    return root, data


def _read(dest, i):
    store = ts.open({"driver": "neuroglancer_precomputed", "kvstore": {"driver": "file", "path": str(dest) + "/"},
                     "scale_index": i}, open=True).result()
    return np.transpose(store[..., 0].read().result(), (2, 1, 0))   # x y z -> z y x


@pytest.mark.parametrize("sharded", [True, False])
def test_export_matches_every_scale(make_config, tmp_path, sharded):
    out = tmp_path / "out"
    root, data = _volume(out)
    cfg = make_config(tmp_path / "raw", precomputed={"chunk": [32, 32, 16], "sharded": sharded,
                                                      "shard_chunks_log2": 3, "minishard_chunks_log2": 1,
                                                      "owners": ["me@example.org"]})
    assert precomputed.main(["init", "--config", str(cfg)]) == 0
    dest = out / "synthetic.precomputed"
    for task in range(2):
        assert precomputed.main(["run", "--config", str(cfg), "--task-id", str(task), "--num-tasks", "3"]) == 0
    assert precomputed.main(["verify", "--config", str(cfg)]) == 1       # a task is missing
    assert not (dest / "info").exists()                                # so the volume doesn't look complete
    assert precomputed.main(["run", "--config", str(cfg), "--task-id", "2", "--num-tasks", "3"]) == 0
    assert precomputed.main(["verify", "--config", str(cfg)]) == 0
    info = json.loads((dest / "info").read_text())                     # written last, by verify
    assert [sc["size"] for sc in info["scales"]] == [list(d.shape[::-1]) for d in data.values()]
    assert info["scales"][0]["resolution"] == [8.0, 8.0, 8.0] and info["scales"][1]["key"] == "16_16_16"
    assert ("sharding" in info["scales"][0]) == sharded
    assert json.loads((dest / "provenance").read_text())["owners"] == ["me@example.org"]
    for s, d in data.items():
        np.testing.assert_array_equal(_read(dest, s), d)
    if sharded:
        sc = info["scales"][0]
        files = sorted(p.name for p in (dest / sc["key"]).iterdir())
        grid = [math.ceil(a / b) for a, b in zip(sc["size"], sc["chunk_sizes"][0])]
        want = sorted(precomputed.shard_file(sc["key"], sc["sharding"], s).split("/")[1]
                      for s, _, _ in precomputed.shard_blocks(grid, 3))
        assert files == want
        # a re-run leaves existing shards alone; a missing one is written again and verify notices it first
        first = dest / sc["key"] / files[0]
        stamp = first.stat().st_mtime_ns
        (dest / sc["key"] / files[1]).unlink()
        assert precomputed.main(["verify", "--config", str(cfg)]) == 1
        assert precomputed.main(["run", "--config", str(cfg)]) == 0
        assert first.stat().st_mtime_ns == stamp
        assert precomputed.main(["verify", "--config", str(cfg)]) == 0


def test_empty_regions_are_not_stored_and_verify_accepts_them(make_config, tmp_path):
    """Shards with no data (all zero: outside the imaged area) get no file; that is fine where the
    zarr is empty too, and still an error where it has data."""
    out = tmp_path / "out"
    root, data = _volume(out)
    data[0][:32, :64, :64] = 0          # exactly the first s0 shard (64 x 64 x 32 voxels in x, y, z)
    omezarr.open_scale(root, 0)[...].write(data[0]).result()
    cfg = make_config(tmp_path / "raw", precomputed={"chunk": [32, 32, 16], "shard_chunks_log2": 3,
                                                      "minishard_chunks_log2": 1})
    assert precomputed.main(["init", "--config", str(cfg)]) == 0
    assert precomputed.main(["run", "--config", str(cfg)]) == 0
    assert precomputed.main(["verify", "--config", str(cfg)]) == 0
    dest = out / "synthetic.precomputed"
    sc = json.loads((dest / "info").read_text())["scales"][0]
    empty = dest / precomputed.shard_file(sc["key"], sc["sharding"], 0)
    assert not empty.exists()
    np.testing.assert_array_equal(_read(dest, 0), data[0])       # read back as zeros
    data[0][5, 5, 5] = 9                # the zarr gains data there: now the missing shard is an error
    omezarr.open_scale(root, 0)[...].write(data[0]).result()
    assert precomputed.main(["verify", "--config", str(cfg)]) == 1


@pytest.mark.parametrize("grid", [(5, 3, 7), (1, 1, 9), (13, 1, 2), (4, 4, 4), (1, 1, 1)])
@pytest.mark.parametrize("k", [0, 2, 3, 6, 20])
def test_shard_blocks_cover_every_chunk_once(grid, k):
    seen = np.zeros(grid, int)
    for _, origin, block in precomputed.shard_blocks(list(grid), k):
        seen[origin[0]:origin[0] + block[0], origin[1]:origin[1] + block[1], origin[2]:origin[2] + block[2]] += 1
    assert (seen == 1).all()


def test_init_refuses_changed_settings(make_config, tmp_path):
    out = tmp_path / "out"
    _volume(out)
    cfg = make_config(tmp_path / "raw", precomputed={"chunk": [32, 32, 16]})
    assert precomputed.main(["init", "--config", str(cfg)]) == 0
    assert precomputed.main(["init", "--config", str(cfg)]) == 0      # same: kept
    cfg = make_config(tmp_path / "raw", precomputed={"chunk": [64, 64, 16]})
    assert precomputed.main(["init", "--config", str(cfg)]) == 1
    assert precomputed.main(["run", "--config", str(cfg)]) == 1       # info doesn't match the settings
    assert precomputed.main(["init", "--config", str(cfg), "--overwrite"]) == 0
    cfg = make_config(tmp_path / "raw", precomputed={"encoding": "png"})
    with pytest.raises(ValueError, match="encoding"):
        precomputed.main(["init", "--config", str(cfg)])


def _render_record(out, digest):
    """A render.json for the synthetic volume: which render of it this is."""
    w = out / "work" / "render" / "synthetic"
    w.mkdir(parents=True, exist_ok=True)
    (w / "render.json").write_text(json.dumps({"digest": digest, "created": digest}))


def test_url_destination_is_written_directly_and_replaced_in_place(make_config, tmp_path):
    """A file:/// destination stands in for a bucket: written through the key-value store, info last; a re-render
    of the same geometry replaces it in place (--overwrite), and a shard that has no data now is deleted."""
    out = tmp_path / "out"
    root, data = _volume(out)
    _render_record(out, "first")
    bucket = tmp_path / "bucket" / "vol"
    pc = {"chunk": [32, 32, 16], "shard_chunks_log2": 3, "minishard_chunks_log2": 1,
          "destination": f"file://{bucket}"}
    cfg = make_config(tmp_path / "raw", precomputed=pc)
    args = ["--config", str(cfg)]
    assert precomputed.main(["init", *args]) == 0
    assert precomputed.main(["run", *args]) == 0
    assert precomputed.main(["verify", *args]) == 0
    assert not (out / "synthetic.precomputed").exists()                  # no local copy
    np.testing.assert_array_equal(_read(bucket, 0), data[0])
    sc = json.loads((bucket / "info").read_text())["scales"][0]
    first = bucket / precomputed.shard_file(sc["key"], sc["sharding"], 0)
    assert first.exists()

    # re-rendered (new contrast, same geometry), and the first shard's region now has no data
    new = {s: (d // 2 + 1).astype(np.uint8) for s, d in data.items()}
    new[0][:32, :64, :64] = 0
    for s, d in new.items():
        omezarr.open_scale(root, s)[...].write(d).result()
    _render_record(out, "second")
    assert precomputed.main(["run", *args]) == 1                         # the plan was for the first render
    assert precomputed.main(["init", *args]) == 1                        # replacing needs --overwrite
    assert precomputed.main(["init", *args, "--overwrite"]) == 0
    assert precomputed.main(["run", *args]) == 0
    assert not first.exists()                                            # deleted: no data there now
    assert precomputed.main(["verify", *args]) == 0
    for s, d in new.items():
        np.testing.assert_array_equal(_read(bucket, s), d)
    # a re-run after a lost shard writes only that one again
    second = bucket / precomputed.shard_file(sc["key"], sc["sharding"], 1)
    stamp = second.stat().st_mtime_ns
    third = bucket / precomputed.shard_file(sc["key"], sc["sharding"], 2)
    third.unlink()
    assert precomputed.main(["verify", *args]) == 1
    assert precomputed.main(["run", *args]) == 0
    assert third.exists() and second.stat().st_mtime_ns == stamp
    assert precomputed.main(["verify", *args]) == 0


def test_destination_with_another_volume_is_left_alone(make_config, tmp_path):
    out = tmp_path / "out"
    _volume(out)
    bucket = tmp_path / "bucket" / "vol"
    pc = {"chunk": [32, 32, 16], "destination": f"file://{bucket}"}
    cfg = make_config(tmp_path / "raw", precomputed=pc)
    assert precomputed.main(["init", "--config", str(cfg)]) == 0
    assert precomputed.main(["run", "--config", str(cfg)]) == 0
    assert precomputed.main(["verify", "--config", str(cfg)]) == 0
    before = sorted(p.relative_to(bucket) for p in bucket.rglob("*"))
    # other export settings: a bucket volume isn't replaced in place, even with --overwrite
    cfg = make_config(tmp_path / "raw", precomputed={**pc, "chunk": [64, 64, 16]})
    assert precomputed.main(["init", "--config", str(cfg), "--overwrite"]) == 1
    assert sorted(p.relative_to(bucket) for p in bucket.rglob("*")) == before
    # another destination: the plan is for the bucket, so switching needs --overwrite
    cfg = make_config(tmp_path / "raw", precomputed={"chunk": [32, 32, 16]})
    assert precomputed.main(["init", "--config", str(cfg)]) == 1
    assert precomputed.main(["init", "--config", str(cfg), "--overwrite"]) == 0
    # a plain folder that holds a volume nobody planned here: refused without --overwrite
    assert precomputed.main(["run", "--config", str(cfg)]) == 0
    assert precomputed.main(["verify", "--config", str(cfg)]) == 0
    shutil.rmtree(out / "work" / "precomputed")
    assert precomputed.main(["init", "--config", str(cfg)]) == 1
    assert precomputed.main(["init", "--config", str(cfg), "--overwrite"]) == 0
    assert not (out / "synthetic.precomputed" / "info").exists()          # deleted first, written again by verify


def test_bucket_destination_spec_and_credentials(tmp_path, monkeypatch):
    """gs:// destinations: the tensorstore spec, and only the key file's path handed on (no network needed)."""
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    key = tmp_path / "key.json"
    key.write_text("{}")
    os.chmod(key, 0o600)
    d = precomputed.Destination("gs://bucket/prefix/", credentials=str(key))
    assert d.spec == {"driver": "gcs", "bucket": "bucket", "path": "prefix/"} and d.local is None
    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == str(key.resolve())
    assert precomputed.Destination("gs://bucket").spec["path"] == ""
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS")
    with pytest.raises(FileNotFoundError):
        precomputed.Destination("gs://bucket/x", credentials=str(tmp_path / "missing.json"))
    with pytest.raises(ValueError, match="destination"):
        precomputed.Destination("http://host/x")
