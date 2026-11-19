"""tools/copy_verify.py: copy, checksum, verify, manifest; refuses nested folders; never deletes."""
import importlib.util
import stat
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("copy_verify", ROOT / "tools" / "copy_verify.py")
cv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cv)


@pytest.fixture
def src(tmp_path):
    s = tmp_path / "src"
    (s / "sub").mkdir(parents=True)
    (s / "a.tif").write_bytes(b"A" * 1000)
    (s / "sub" / "b.tif").write_bytes(bytes(range(256)) * 40)
    (s / "notes.txt").write_text("not copied by the default pattern")
    return s


def test_run_copies_verifies_and_writes_manifest(src, tmp_path):
    dst = tmp_path / "copies" / "dst"
    assert cv.main(["run", str(src), str(dst), "--group-rw"]) == 0
    assert (dst / "a.tif").read_bytes() == (src / "a.tif").read_bytes()
    assert (dst / "sub" / "b.tif").read_bytes() == (src / "sub" / "b.tif").read_bytes()
    assert not (dst / "notes.txt").exists()
    manifest = cv.manifest_path(dst)
    assert manifest == dst.parent / "dst.md5"
    lines = manifest.read_text().splitlines()
    assert [ln.split("  ")[1] for ln in lines] == ["./a.tif", "./sub/b.tif"]
    assert stat.S_IMODE((dst / "a.tif").stat().st_mode) == 0o660
    assert (src / "notes.txt").exists()                                # source untouched


def test_mismatch_is_detected_and_no_manifest(src, tmp_path):
    dst = tmp_path / "dst"
    cv.main(["plan", str(src), str(dst)])
    cv.main(["copy", str(src), str(dst)])
    (dst / "a.tif").write_bytes(b"B" * 1000)                           # same size, different content
    cv.main(["md5", str(src), str(dst)])
    assert cv.main(["verify", str(src), str(dst)]) == 1
    assert not cv.manifest_path(dst).exists()


def test_missing_checksum_and_extra_file_fail_without_deleting(src, tmp_path):
    dst = tmp_path / "dst"
    cv.main(["plan", str(src), str(dst)])
    cv.main(["copy", str(src), str(dst)])
    cv.main(["md5", str(src), str(dst), "--index", "0"])               # file 1 never checksummed
    assert cv.main(["verify", str(src), str(dst)]) == 1
    cv.main(["md5", str(src), str(dst), "--index", "1"])
    (dst / "stray.bin").write_bytes(b"x")
    assert cv.main(["verify", str(src), str(dst)]) == 1
    assert (dst / "stray.bin").exists()                                # flagged, not deleted
    (dst / "stray.bin").rename(tmp_path / "stray.bin")
    assert cv.main(["verify", str(src), str(dst)]) == 0


@pytest.mark.parametrize("make_dst", [lambda s: s / "inside", lambda s: s, lambda s: s.parent])
def test_refuses_nested_or_same_folders(src, make_dst):
    with pytest.raises(SystemExit):
        cv.main(["plan", str(src), str(make_dst(src))])


def test_submit_dry_run_prints_chain(src, tmp_path, capsys):
    dst = tmp_path / "dst"
    assert cv.main(["submit", str(src), str(dst), "--dry-run", "--parallel", "3"]) == 0
    out = capsys.readouterr().out
    assert "--array=0-1%3" in out and "copy_verify.sbatch copy" in out and "copy_verify.sbatch verify" in out
    assert not dst.exists() or not any(dst.iterdir())                  # dry run copies nothing
