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
    assert not cv.work_dir(dst).exists()                               # bookkeeping removed after a PASS


def test_manifest_elsewhere_and_bookkeeping_kept_after_a_fail(src, tmp_path):
    dst = tmp_path / "raw" / "Backscatter"
    listed = tmp_path / "raw_backscatter.md5"                          # next to the raw folder, not inside it
    m = ["--manifest", str(listed)]
    cv.main(["plan", str(src), str(dst), *m])
    assert cv.work_dir(dst, listed) == tmp_path / "raw_backscatter.copy_verify"
    cv.main(["copy", str(src), str(dst), *m])
    cv.main(["md5", str(src), str(dst), "--index", "0", *m])
    assert cv.main(["verify", str(src), str(dst), *m]) == 1
    assert cv.work_dir(dst, listed).is_dir() and not listed.exists()   # kept to show what went wrong
    cv.main(["md5", str(src), str(dst), "--index", "1", *m])
    assert cv.main(["verify", str(src), str(dst), *m]) == 0
    assert [ln.split("  ")[1] for ln in listed.read_text().splitlines()] == ["./a.tif", "./sub/b.tif"]
    assert not cv.manifest_path(dst).exists() and not cv.work_dir(dst, listed).exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["raw", "raw_backscatter.md5", "src"]


def test_one_list_per_detector_in_one_folder(tmp_path):
    raw = tmp_path / "facility" / "VOL"
    for rel in ("M01_D01_tile0-0.tif", "Approach/M01_D02_tile0-0.tif", "Backscatter/M01_D01_EsB_tile0-0.tif",
                "Approach/Backscatter/M01_D02_EsB_tile0-0.tif", "uCT datasets/scan.tif"):
        (raw / rel).parent.mkdir(parents=True, exist_ok=True)
        (raw / rel).write_bytes(rel.encode() * 50)
    dst, inlens, esb = tmp_path / "lab" / "VOL", tmp_path / "lab" / "VOL.md5", tmp_path / "lab" / "VOL_backscatter.md5"
    runs = [(r"^(?!(.*/)?Backscatter/).*\.tif$", inlens), (r"(^|/)Backscatter/[^/]*\.tif$", esb)]
    for pattern, listed in runs:                                       # both planned before either verifies
        cv.main(["plan", str(raw), str(dst), "--pattern", pattern, "--manifest", str(listed)])
    for _, listed in runs:
        cv.main(["copy", str(raw), str(dst), "--manifest", str(listed)])
        cv.main(["md5", str(raw), str(dst), "--manifest", str(listed)])
    for _, listed in runs:
        assert cv.main(["verify", str(raw), str(dst), "--manifest", str(listed)]) == 0
    names = lambda p: sorted(ln.split("  ")[1] for ln in p.read_text().splitlines())
    assert names(inlens) == ["./Approach/M01_D02_tile0-0.tif", "./M01_D01_tile0-0.tif", "./uCT datasets/scan.tif"]
    assert names(esb) == ["./Approach/Backscatter/M01_D02_EsB_tile0-0.tif", "./Backscatter/M01_D01_EsB_tile0-0.tif"]
    assert sorted(p.name for p in dst.parent.iterdir()) == ["VOL", "VOL.md5", "VOL_backscatter.md5"]   # no folders left
    (dst / "Backscatter" / "M01_D03_EsB_tile0-0.tif").write_bytes(b"x")   # a stray EsB file fails only the EsB list
    cv.main(["plan", str(raw), str(dst), "--pattern", runs[1][0], "--manifest", str(esb)])
    cv.main(["md5", str(raw), str(dst), "--manifest", str(esb)])
    assert cv.main(["verify", str(raw), str(dst), "--manifest", str(esb)]) == 1


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
    (dst / "stray.tif").write_bytes(b"x")
    assert cv.main(["verify", str(src), str(dst)]) == 1
    assert (dst / "stray.tif").exists()                                # flagged, not deleted
    (dst / "stray.tif").rename(tmp_path / "stray.tif")
    (dst / "notes.txt").write_text("not a .tif: outside this list's pattern")
    assert cv.main(["verify", str(src), str(dst)]) == 0
    assert (dst / "notes.txt").exists()


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
