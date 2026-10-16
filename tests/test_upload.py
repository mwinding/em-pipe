"""upload: copy a finished output folder to a bucket (here a local folder standing in for one)."""

import os

import pytest

from pipeline import upload


def _folder(out):
    pc = out / "vol.precomputed"
    (pc / "8_8_8").mkdir(parents=True)
    (pc / "16_16_16").mkdir()
    (pc / "info").write_text('{"@type": "neuroglancer_multiscale_volume"}')
    (pc / "provenance").write_text("{}")
    for i in range(7):
        (pc / "8_8_8" / f"{i:x}.shard").write_bytes(bytes([i]) * (100 + i))
    (pc / "16_16_16" / "0.shard").write_bytes(b"x" * 50)
    return pc


def test_upload_runs_resumes_and_finishes(make_config, tmp_path):
    out = tmp_path / "out"
    _folder(out)
    bucket = tmp_path / "bucket"
    cfg = make_config(tmp_path / "raw", upload={"source": "vol.precomputed", "destination": f"file://{bucket}"})
    assert upload.main(["run", "--config", str(cfg), "--dry-run"]) == 0
    assert not bucket.exists()
    for task in range(3):
        assert upload.main(["run", "--config", str(cfg), "--task-id", str(task), "--num-tasks", "3"]) == 0
    got = upload.local_files(bucket)
    assert len(got) == 8 and "info" not in got and "provenance" not in got     # metadata waits for finish
    assert upload.main(["finish", "--config", str(cfg)]) == 0
    assert upload.local_files(bucket) == upload.local_files(out / "vol.precomputed")
    # a re-run sends nothing; a file of another size (e.g. cut short) is sent again
    stamp = (bucket / "8_8_8" / "0.shard").stat().st_mtime_ns
    (bucket / "8_8_8" / "3.shard").write_bytes(b"short")
    assert upload.main(["run", "--config", str(cfg)]) == 0
    assert (bucket / "8_8_8" / "0.shard").stat().st_mtime_ns == stamp
    assert (bucket / "8_8_8" / "3.shard").read_bytes() == bytes([3]) * 103


def test_finish_refuses_until_every_data_file_is_there(make_config, tmp_path):
    out = tmp_path / "out"
    _folder(out)
    bucket = tmp_path / "bucket"
    cfg = make_config(tmp_path / "raw", upload={"source": "vol.precomputed", "destination": f"file://{bucket}"})
    assert upload.main(["run", "--config", str(cfg), "--task-id", "0", "--num-tasks", "2"]) == 0
    assert upload.main(["finish", "--config", str(cfg)]) == 1
    assert not (bucket / "info").exists()


def test_credentials_checks(tmp_path, caplog):
    key = tmp_path / "key.json"
    key.write_text("{}")
    os.chmod(key, 0o600)
    assert upload.credentials_path({"credentials": str(key)}) == key.resolve()
    os.chmod(key, 0o644)
    upload.credentials_path({"credentials": str(key)})
    assert "chmod 600" in caplog.text
    with pytest.raises(FileNotFoundError):
        upload.credentials_path({"credentials": str(tmp_path / "missing.json")})
    with pytest.raises(ValueError, match="inside the repository"):
        upload.credentials_path({"credentials": str(upload.REPO / "configs" / "example.yaml")})


def test_destination_must_be_set_and_known(make_config, tmp_path):
    _folder(tmp_path / "out")
    cfg = make_config(tmp_path / "raw", upload={"source": "vol.precomputed"})
    with pytest.raises(ValueError, match="destination"):
        upload.main(["run", "--config", str(cfg)])
    cfg = make_config(tmp_path / "raw", upload={"source": "vol.precomputed", "destination": "s3://b/x"})
    with pytest.raises(ValueError, match="gs://"):
        upload.main(["run", "--config", str(cfg)])
